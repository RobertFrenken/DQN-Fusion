import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
import random

from models.models import GATWithJK, GraphAutoencoderNeighborhood
from preprocessing import graph_creation, build_id_mapping_from_normal
from training_utils import DistillationTrainer, distillation_loss_fn, FocalLoss
from torch_geometric.data import Batch

def extract_latent_vectors(pipeline, loader):
    """Extract latent vectors (graph embeddings) and labels from a data loader."""
    pipeline.autoencoder.eval()
    zs, labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(pipeline.device)
            _, _, _, z, _ = pipeline.autoencoder(batch.x, batch.edge_index, batch.batch)
            
            graphs = Batch.to_data_list(batch)
            start = 0
            for graph in graphs:
                n = graph.x.size(0)
                z_graph = z[start:start+n].mean(dim=0).cpu().numpy()
                zs.append(z_graph)
                labels.append(int(graph.y.flatten()[0]))
                start += n
                
    return np.array(zs), np.array(labels)

def create_teacher_student_models(num_ids, embedding_dim, device):
    """Create teacher (large) and student (small) models."""
    
    # Teacher models (large, complex)
    teacher_autoencoder = GraphAutoencoderNeighborhood(
        num_ids=num_ids, 
        in_channels=11, 
        embedding_dim=embedding_dim,
        hidden_dim=32,           # Larger hidden dimension
        latent_dim=32,           # Larger latent dimension
        num_encoder_layers=3,    # More layers
        num_decoder_layers=3,
        encoder_heads=4,         # More attention heads
        decoder_heads=4
    ).to(device)
    
    teacher_classifier = GATWithJK(
        num_ids=num_ids, 
        in_channels=11, 
        hidden_channels=32,      # Smaller hidden channels
        out_channels=1, 
        num_layers=5,            # More layers
        heads=8,                # More attention heads
        embedding_dim=embedding_dim
    ).to(device)
    
    # Student models (small, efficient)
    student_autoencoder = GraphAutoencoderNeighborhood(
        num_ids=num_ids, 
        in_channels=11, 
        embedding_dim=embedding_dim,
        hidden_dim=16,           # Smaller hidden dimension
        latent_dim=16,           # Smaller latent dimension
        num_encoder_layers=2,    # Fewer layers
        num_decoder_layers=2,
        encoder_heads=2,         # Fewer attention heads
        decoder_heads=2
    ).to(device)
    
    student_classifier = GATWithJK(
        num_ids=num_ids, 
        in_channels=11, 
        hidden_channels=16,      # Smaller hidden channels
        out_channels=1, 
        num_layers=2,            # Fewer layers
        heads=4,                 # Fewer attention heads
        embedding_dim=embedding_dim
    ).to(device)
    
    return teacher_autoencoder, teacher_classifier, student_autoencoder, student_classifier

class KnowledgeDistillationPipeline:
    """Knowledge distillation pipeline for CAN bus anomaly detection."""
    
    def __init__(self, teacher_autoencoder, teacher_classifier, 
                 student_autoencoder, student_classifier, device='cpu'):
        self.device = device
        
        # Teacher models (pre-trained)
        self.teacher_autoencoder = teacher_autoencoder.to(device)
        self.teacher_classifier = teacher_classifier.to(device)
        
        # Student models (to be trained)
        self.student_autoencoder = student_autoencoder.to(device)
        self.student_classifier = student_classifier.to(device)
        
        # Set teachers to eval mode
        self.teacher_autoencoder.eval()
        self.teacher_classifier.eval()
        
        # Freeze teacher parameters
        for param in self.teacher_autoencoder.parameters():
            param.requires_grad = False
        for param in self.teacher_classifier.parameters():
            param.requires_grad = False
            
        self.threshold = 0.0

    def load_teacher_models(self, autoencoder_path, classifier_path):
        """Load pre-trained teacher models with proper checkpoint handling."""
        print(f"Loading teacher autoencoder from: {autoencoder_path}")
        
        # Load autoencoder checkpoint
        autoencoder_checkpoint = torch.load(autoencoder_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(autoencoder_checkpoint, dict):
            if 'state_dict' in autoencoder_checkpoint:
                # Checkpoint contains metadata
                autoencoder_state_dict = autoencoder_checkpoint['state_dict']
                if 'threshold' in autoencoder_checkpoint:
                    self.threshold = autoencoder_checkpoint['threshold']
                    # print(f"Loaded threshold: {self.threshold}")
            else:
                # Dictionary but direct state dict
                autoencoder_state_dict = autoencoder_checkpoint
        else:
            # Direct state dict
            autoencoder_state_dict = autoencoder_checkpoint
        
        # Load classifier checkpoint
        print(f"Loading teacher classifier from: {classifier_path}")
        classifier_checkpoint = torch.load(classifier_path, map_location=self.device)
        
        if isinstance(classifier_checkpoint, dict):
            if 'state_dict' in classifier_checkpoint:
                classifier_state_dict = classifier_checkpoint['state_dict']
            else:
                classifier_state_dict = classifier_checkpoint
        else:
            classifier_state_dict = classifier_checkpoint
        
        # Load state dicts
        try:
            self.teacher_autoencoder.load_state_dict(autoencoder_state_dict)
            print("Teacher autoencoder loaded successfully!")
            
            self.teacher_classifier.load_state_dict(classifier_state_dict)
            print("Teacher classifier loaded successfully!")
            
        except RuntimeError as e:
            print(f"Architecture mismatch: {str(e)[:200]}...")
            print("The saved model has a different architecture than expected.")
            print("Please check if the saved models were created with different parameters.")
            raise e
        
        # Set to eval mode
        self.teacher_autoencoder.eval()
        self.teacher_classifier.eval()
        
        print("Teacher models loaded and set to evaluation mode!")

    def distill_autoencoder(self, train_loader, epochs=20, alpha=0.5, temperature=5.0):
        """Knowledge distillation for autoencoder with dimension alignment."""
        print(f"Distilling autoencoder for {epochs} epochs...")
        
        self.student_autoencoder.train()
        optimizer = torch.optim.Adam(self.student_autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Add projection layer to align student latent space with teacher
        teacher_latent_dim = self.teacher_autoencoder.latent_dim  # 32
        student_latent_dim = self.student_autoencoder.latent_dim  # 16
        
        if teacher_latent_dim != student_latent_dim:
            projection_layer = nn.Linear(student_latent_dim, teacher_latent_dim).to(self.device)
            proj_optimizer = torch.optim.Adam(projection_layer.parameters(), lr=1e-3)
            print(f"Added projection layer: {student_latent_dim} -> {teacher_latent_dim}")
        else:
            projection_layer = None
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                # Teacher forward pass (compute fresh each time to avoid caching issues)
                with torch.no_grad():
                    teacher_cont_out, teacher_canid_logits, teacher_neighbor_logits, teacher_z, _ = \
                        self.teacher_autoencoder(batch.x, batch.edge_index, batch.batch)
                
                # Student forward pass
                student_outputs = self.student_autoencoder(batch.x, batch.edge_index, batch.batch)
                student_cont_out, student_canid_logits, student_neighbor_logits, student_z, student_kl_loss = student_outputs
                
                # Standard reconstruction losses (vectorized operations)
                cont_loss = nn.MSELoss()(student_cont_out, batch.x[:, 1:])
                canid_loss = nn.CrossEntropyLoss()(student_canid_logits, batch.x[:, 0].long())
                
                neighbor_targets = self.student_autoencoder.create_neighborhood_targets(
                    batch.x, batch.edge_index, batch.batch)
                neighbor_loss = nn.BCEWithLogitsLoss()(student_neighbor_logits, neighbor_targets)
                
                # Knowledge distillation losses
                # 1. Feature distillation with projection
                if projection_layer is not None:
                    student_z_projected = projection_layer(student_z)
                    feature_distill_loss = nn.MSELoss()(student_z_projected, teacher_z)
                else:
                    feature_distill_loss = nn.MSELoss()(student_z, teacher_z)
                
                # 2. Output distillation (soft targets)
                cont_distill_loss = nn.MSELoss()(student_cont_out, teacher_cont_out.detach())
                canid_distill_loss = distillation_loss_fn(student_canid_logits, teacher_canid_logits.detach(), T=temperature)
                neighbor_distill_loss = distillation_loss_fn(student_neighbor_logits, teacher_neighbor_logits.detach(), T=temperature)
                
                # Combined loss
                reconstruction_loss = cont_loss + canid_loss + neighbor_loss + 0.1 * student_kl_loss
                distillation_loss = feature_distill_loss + cont_distill_loss + canid_distill_loss + neighbor_distill_loss
                
                total_loss = (1 - alpha) * reconstruction_loss + alpha * distillation_loss
                
                # Backward pass
                optimizer.zero_grad()
                if projection_layer is not None:
                    proj_optimizer.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.student_autoencoder.parameters(), max_norm=1.0)
                if projection_layer is not None:
                    torch.nn.utils.clip_grad_norm_(projection_layer.parameters(), max_norm=1.0)
                
                optimizer.step()
                if projection_layer is not None:
                    proj_optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Clear cache periodically to manage memory
                if num_batches % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                print(f"Autoencoder Distillation Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Set threshold using student model
        self._set_threshold_student(train_loader)
        print(f"Set student anomaly threshold: {self.threshold:.4f}")

    def distill_classifier(self, full_loader, epochs=20, alpha=0.7, temperature=5.0):
        """Knowledge distillation for classifier using filtered graphs."""
        print(f"Distilling classifier for {epochs} epochs...")
        
        # Create balanced dataset using student autoencoder
        balanced_graphs = self._create_balanced_dataset_student(full_loader)
        if not balanced_graphs:
            print("No graphs available for classifier distillation.")
            return
        
        self.student_classifier.train()
        
        # Use DistillationTrainer for classifier
        trainer = DistillationTrainer(
            teacher=self.teacher_classifier,
            student=self.student_classifier,
            device=self.device,
            teacher_epochs=0,  # Teacher already trained
            student_epochs=epochs,
            distill_alpha=alpha,
            warmup_epochs=5,
            lr=1e-3,
            use_focal_loss=True
        )
        
        # Create data loader for balanced graphs with better GPU utilization
        balanced_loader = DataLoader(
            balanced_graphs, 
            batch_size=32, 
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=4 if torch.cuda.is_available() else 0
        )
        test_loader = DataLoader(
            balanced_graphs[:min(100, len(balanced_graphs))], 
            batch_size=32, 
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=4 if torch.cuda.is_available() else 0
        )
        
        # Train student classifier with knowledge distillation
        trainer.train_student(balanced_loader, test_loader)

    def _set_threshold_student(self, train_loader, percentile=50):
        """Set anomaly detection threshold using student autoencoder."""
        errors = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.device)
                cont_out, _, _, _, _ = self.student_autoencoder(batch.x, batch.edge_index, batch.batch)
                errors.append((cont_out - batch.x[:, 1:]).pow(2).mean(dim=1))
        self.threshold = torch.cat(errors).quantile(percentile / 100.0).item()

    def _create_balanced_dataset_student(self, loader):
        """Create balanced dataset using student autoencoder."""
        print("Computing composite errors using student model...")
        
        all_graphs = []
        all_composite_errors = []
        all_is_attack = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                # Use student autoencoder for filtering
                cont_out, canid_logits, neighbor_logits, _, _ = self.student_autoencoder(
                    batch.x, batch.edge_index, batch.batch)
                
                # Vectorized composite error computation
                node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
                
                neighbor_targets = self.student_autoencoder.create_neighborhood_targets(
                    batch.x, batch.edge_index, batch.batch)
                neighbor_recon_errors = nn.BCEWithLogitsLoss(reduction='none')(
                    neighbor_logits, neighbor_targets).mean(dim=1)
                
                canid_pred = canid_logits.argmax(dim=1)
                
                graphs = Batch.to_data_list(batch)
                start = 0
                
                for graph in graphs:
                    num_nodes = graph.x.size(0)
                    is_attack = int(graph.y.flatten()[0]) == 1
                    
                    graph_node_error = node_errors[start:start+num_nodes].max().item()
                    graph_neighbor_error = neighbor_recon_errors[start:start+num_nodes].max().item()
                    
                    true_canids = graph.x[:, 0].long()
                    pred_canids = canid_pred[start:start+num_nodes]
                    canid_error = 1.0 - (pred_canids == true_canids).float().mean().item()
                    
                    composite_error = (1.0 * graph_node_error + 
                                     20.0 * graph_neighbor_error + 
                                     0.3 * canid_error)
                    
                    all_graphs.append(graph.cpu())
                    all_composite_errors.append(composite_error)
                    all_is_attack.append(is_attack)
                    start += num_nodes
        
        # Same filtering logic as original
        graph_data = list(zip(all_graphs, all_composite_errors, all_is_attack))
        attack_graphs = [(graph, error) for graph, error, is_attack in graph_data if is_attack]
        normal_graphs = [(graph, error) for graph, error, is_attack in graph_data if not is_attack]
        
        selected_attack_graphs = [graph for graph, _ in attack_graphs]
        num_attacks = len(selected_attack_graphs)
        
        if num_attacks == 0:
            return []
        
        max_normal_graphs = num_attacks * 4
        
        if len(normal_graphs) <= max_normal_graphs:
            selected_normal_graphs = [graph for graph, _ in normal_graphs]
        else:
            normal_graphs_sorted = sorted(normal_graphs, key=lambda x: x[1])
            selected_normal_graphs = [graph for graph, _ in normal_graphs_sorted[-max_normal_graphs:]]
        
        balanced_graphs = selected_attack_graphs + selected_normal_graphs
        random.shuffle(balanced_graphs)
        
        print(f"Created balanced dataset: {len(selected_normal_graphs)} normal, {num_attacks} attack")
        return balanced_graphs

    def predict_student(self, data):
        """Optimized batch prediction using student models."""
        data = data.to(self.device)
        
        with torch.no_grad():
            # Single forward pass for entire batch
            cont_out, _, _, _, _ = self.student_autoencoder(data.x, data.edge_index, data.batch)
            node_errors = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            
            # Vectorized graph-level error computation
            batch_size = data.batch.max().item() + 1
            graph_errors = torch.zeros(batch_size, device=self.device)
            
            # Efficient scatter operation for max errors per graph
            graph_errors.scatter_reduce_(0, data.batch, node_errors, reduce='amax')
            
            # Vectorized threshold check
            anomaly_mask = graph_errors > self.threshold
            
            if anomaly_mask.any():
                # Process anomalous graphs
                anomalous_graphs = []
                anomaly_indices = torch.nonzero(anomaly_mask).flatten()
                
                for idx in anomaly_indices:
                    mask = data.batch == idx
                    # Create subgraph efficiently
                    edge_mask = (data.batch[data.edge_index[0]] == idx) & (data.batch[data.edge_index[1]] == idx)
                    
                    if mask.sum() > 0:  # Ensure graph has nodes
                        graph_data = data.clone()
                        graph_data.x = data.x[mask]
                        
                        if edge_mask.sum() > 0:  # Ensure graph has edges
                            subgraph_edges = data.edge_index[:, edge_mask]
                            # Reindex edges for subgraph
                            node_mapping = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                            node_mapping[mask] = torch.arange(mask.sum(), device=self.device)
                            graph_data.edge_index = node_mapping[subgraph_edges]
                        else:
                            # Handle isolated nodes
                            graph_data.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                        
                        graph_data.batch = torch.zeros(mask.sum(), dtype=torch.long, device=self.device)
                        anomalous_graphs.append(graph_data)
                
                # Batch process anomalous graphs if possible
                preds = torch.zeros(batch_size, device=self.device)
                
                if len(anomalous_graphs) > 1:
                    # Try to batch classifier calls
                    try:
                        batched_anomalous = Batch.from_data_list(anomalous_graphs)
                        classifier_probs = self.student_classifier(batched_anomalous)
                        preds[anomaly_indices[:len(anomalous_graphs)]] = (classifier_probs.flatten() > 0.5).float()
                    except:
                        # Fallback to individual processing
                        for i, graph in enumerate(anomalous_graphs):
                            if i < len(anomaly_indices):
                                prob = self.student_classifier(graph).item()
                                preds[anomaly_indices[i]] = 1 if prob > 0.5 else 0
                elif len(anomalous_graphs) == 1:
                    # Single anomalous graph
                    prob = self.student_classifier(anomalous_graphs[0]).item()
                    preds[anomaly_indices[0]] = 1 if prob > 0.5 else 0
            else:
                preds = torch.zeros(batch_size, device=self.device)
            
            return preds.long()

class GATPipeline:
    """Original GATPipeline for teacher model evaluation comparison."""
    
    def __init__(self, num_ids, embedding_dim=8, device='cpu'):
        """Initialize the pipeline with autoencoder and classifier."""
        self.device = device
        self.autoencoder = GraphAutoencoderNeighborhood(
            num_ids=num_ids, in_channels=11, embedding_dim=embedding_dim
        ).to(device)
        self.classifier = GATWithJK(
            num_ids=num_ids, in_channels=11, hidden_channels=32, 
            out_channels=1, num_layers=3, heads=8, embedding_dim=embedding_dim
        ).to(device)
        self.threshold = 0.0

    def predict(self, data):
        """Two-stage prediction: anomaly detection + classification."""
        data = data.to(self.device)
        
        with torch.no_grad():
            cont_out, _, _, _, _ = self.autoencoder(data.x, data.edge_index, data.batch)
            error = (cont_out - data.x[:, 1:]).pow(2).mean(dim=1)
            
            preds = []
            graphs = Batch.to_data_list(data)
            start = 0
            for graph in graphs:
                num_nodes = graph.x.size(0)
                node_errors = error[start:start+num_nodes]
                
                if node_errors.numel() > 0 and (node_errors > self.threshold).any():
                    graph_batch = graph.to(self.device)
                    prob = self.classifier(graph_batch).item()
                    preds.append(1 if prob > 0.5 else 0)
                else:
                    preds.append(0)
                start += num_nodes
            
            return torch.tensor(preds, device=self.device)

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):
    """Knowledge distillation pipeline for CAN bus anomaly detection."""
    # Setup
    config_dict = OmegaConf.to_container(config, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Dataset paths
    root_folders = {
        'hcrl_ch': r"datasets/can-train-and-test-v1.5/hcrl-ch",
        'hcrl_sa': r"datasets/can-train-and-test-v1.5/hcrl-sa",
        'set_01': r"datasets/can-train-and-test-v1.5/set_01",
        'set_02': r"datasets/can-train-and-test-v1.5/set_02",
        'set_03': r"datasets/can-train-and-test-v1.5/set_03",
        'set_04': r"datasets/can-train-and-test-v1.5/set_04",
    }
    
    # Load data
    KEY = config_dict['root_folder']
    root_folder = root_folders[KEY]
    id_mapping = build_id_mapping_from_normal(root_folder)
    dataset = graph_creation(root_folder, id_mapping=id_mapping, window_size=100)
    # Fix the random seed for consistent batching
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    
    print(f"Dataset: {len(dataset)} graphs, {len(id_mapping)} unique CAN IDs")
    
    # Configuration with optimized batch size
    DATASIZE = config_dict['datasize']
    TRAIN_RATIO = config_dict['train_ratio']
    BATCH_SIZE = min(config_dict['batch_size'] * 2, 64)  # Increase for better GPU utilization
    
    # Train/test split
    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    
    # Create normal-only training subset
    normal_indices = [i for i, data in enumerate(train_dataset) if int(data.y.flatten()[0]) == 0]
    if DATASIZE < 1.0:
        subset_size = int(len(normal_indices) * DATASIZE)
        indices = np.random.choice(normal_indices, subset_size, replace=False)
    else:
        indices = normal_indices
    
    normal_subset = Subset(train_dataset, indices)
    
    # Optimized data loaders with pin_memory and num_workers
    train_loader = DataLoader(
        normal_subset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=4 if torch.cuda.is_available() else 0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=4 if torch.cuda.is_available() else 0
    )
    full_train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=4 if torch.cuda.is_available() else 0
    )
    
    # Create teacher and student models
    teacher_ae, teacher_clf, student_ae, student_clf = create_teacher_student_models(
        num_ids=len(id_mapping), embedding_dim=8, device=device
    )
    
    # Initialize knowledge distillation pipeline
    kd_pipeline = KnowledgeDistillationPipeline(
        teacher_autoencoder=teacher_ae,
        teacher_classifier=teacher_clf,
        student_autoencoder=student_ae,
        student_classifier=student_clf,
        device=device
    )
    
    # Load pre-trained teacher models
    teacher_ae_path = f"output_model_1/autoencoder_best_{KEY}.pth"
    teacher_clf_path = f"output_model_1/classifier_{KEY}.pth"

    try:
        kd_pipeline.load_teacher_models(teacher_ae_path, teacher_clf_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run osc-training-AD.py first to train teacher models!")
        return
    
    # Knowledge distillation training
    print("\n=== Stage 1: Autoencoder Knowledge Distillation ===")
    kd_pipeline.distill_autoencoder(train_loader, epochs=5, alpha=0.5, temperature=5.0)
    
    print("\n=== Stage 2: Classifier Knowledge Distillation ===")
    kd_pipeline.distill_classifier(full_train_loader, epochs=5, alpha=0.7, temperature=5.0)
    
    # Save student models
    save_folder = "saved_models"
    os.makedirs(save_folder, exist_ok=True)
    torch.save(kd_pipeline.student_autoencoder.state_dict(), 
               os.path.join(save_folder, f'student_autoencoder_{KEY}.pth'))
    torch.save(kd_pipeline.student_classifier.state_dict(), 
               os.path.join(save_folder, f'student_classifier_{KEY}.pth'))
    print(f"Student models saved to '{save_folder}'")
    
    # Optimized evaluation
    print("\n=== Evaluation: Student Model Performance ===")
    
    all_student_preds = []
    all_labels = []
    
    # Process in optimized batches
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Batch prediction
            student_preds = kd_pipeline.predict_student(batch)
            labels = batch.y
            
            all_student_preds.append(student_preds.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate results
    student_preds = torch.cat(all_student_preds)
    labels = torch.cat(all_labels)
    
    student_accuracy = (student_preds == labels).float().mean().item()

    print(f"Student Model Accuracy: {student_accuracy:.4f}")
    print("\nStudent Confusion Matrix:")
    print(confusion_matrix(labels.numpy(), student_preds.numpy()))

    # Model size comparison
    teacher_params = sum(p.numel() for p in teacher_ae.parameters()) + sum(p.numel() for p in teacher_clf.parameters())
    student_params = sum(p.numel() for p in student_ae.parameters()) + sum(p.numel() for p in student_clf.parameters())

    print(f"\nModel Size Comparison:")
    print(f"Teacher Parameters: {teacher_params:,}")
    print(f"Student Parameters: {student_params:,}")
    print(f"Compression Ratio: {teacher_params/student_params:.1f}x")

    print("\nKnowledge Distillation Complete!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds")