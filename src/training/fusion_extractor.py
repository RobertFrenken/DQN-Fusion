"""
Fusion Data Extractor for GPU-optimized fusion training.
Extracts anomaly scores and GAT probabilities for fusion learning.
"""
from contextlib import nullcontext
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Tuple, List
from src.config.fusion_config import FUSION_WEIGHTS


class FusionDataExtractor:
    """Extract anomaly scores and GAT probabilities for fusion training - GPU OPTIMIZED."""
    
    def __init__(self, autoencoder: nn.Module, classifier: nn.Module, 
                 device: str, threshold: float = 0.0):
        self.autoencoder = autoencoder.to(device)
        self.classifier = classifier.to(device)
        self.device = torch.device(device)
        self.threshold = threshold
        
        # Set models to evaluation mode
        self.autoencoder.eval()
        self.classifier.eval()
        
        # Pre-compute fusion weights as tensor for GPU operations
        self.fusion_weights = torch.tensor([
            FUSION_WEIGHTS['node_reconstruction'],
            FUSION_WEIGHTS['neighborhood_prediction'],
            FUSION_WEIGHTS['can_id_prediction']
        ], dtype=torch.float32, device=self.device)
        
        print(f"✓ Fusion Data Extractor initialized (GPU-Optimized) with threshold: {threshold:.4f}")

    def compute_anomaly_scores(self, batch) -> torch.Tensor:
        """Memory-efficient computation without massive tensors."""
        with torch.no_grad():
            # Forward pass (normal)
            cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                batch.x, batch.edge_index, batch.batch
            )
            
            # Create neighborhood targets more efficiently
            neighbor_targets = self.autoencoder.create_neighborhood_targets(
                batch.x, batch.edge_index, batch.batch
            )
            
            # Node-level errors (efficient)
            node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
            neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
                neighbor_logits, neighbor_targets
            ).mean(dim=1)
            canid_pred = canid_logits.argmax(dim=1)
            true_canids = batch.x[:, 0].long()
            canid_errors = (canid_pred != true_canids).float()
            
            # MEMORY OPTIMIZATION: Process graphs in smaller chunks
            num_graphs = batch.batch.max().item() + 1
            chunk_size = min(2048, num_graphs)  # Process 2048 graphs at a time
            
            graph_errors_list = []
            for chunk_start in range(0, num_graphs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_graphs)
                chunk_graph_errors = torch.zeros(chunk_end - chunk_start, 3, 
                                            device=self.device, dtype=node_errors.dtype)
                
                for i, graph_idx in enumerate(range(chunk_start, chunk_end)):
                    node_mask = (batch.batch == graph_idx)
                    if node_mask.any():
                        graph_node_errors = torch.stack([
                            node_errors[node_mask], 
                            neighbor_errors[node_mask], 
                            canid_errors[node_mask]
                        ], dim=1)
                        chunk_graph_errors[i] = graph_node_errors.max(dim=0)[0]
                
                graph_errors_list.append(chunk_graph_errors)
            
            # Combine chunks
            graph_errors = torch.cat(graph_errors_list, dim=0)
            
            # Weighted composite score (efficient)
            composite_scores = (graph_errors * self.fusion_weights).sum(dim=1)
            return torch.sigmoid(composite_scores * 3 - 1.5)

    def compute_gat_probabilities(self, batch) -> torch.Tensor:
        """
        GPU-accelerated computation of GAT classification probabilities.
        Already efficient - just needs to stay on GPU.
        """
        with torch.no_grad():
            logits = self.classifier(batch)
            probabilities = torch.sigmoid(logits.squeeze())
            return probabilities  # Keep on GPU!

    def extract_fusion_data(self, data_loader: DataLoader, max_samples: int = None) -> Tuple[List, List, List]:
        """Eliminate CPU serialization bottleneck."""
        print("🚀 GPU-Optimized Fusion Data Extraction...")
        
        # Pre-allocate GPU tensors to avoid repeated allocation
        device_tensors = {
            'anomaly_scores': [],
            'gat_probs': [],  
            'labels': []
        }
        
        samples_processed = 0
        total_batches = len(data_loader)
        
        # Process in larger chunks to reduce Python overhead
        with torch.cuda.stream(torch.cuda.Stream()) if self.device.type == 'cuda' else nullcontext() as stream:
            with tqdm(data_loader, desc="GPU Extraction", total=total_batches, 
                    miniters=max(1, total_batches//20)) as pbar:
                
                for batch_idx, batch in enumerate(pbar):
                    if self.device.type == 'cuda':
                        # Async GPU transfer with stream
                        batch = batch.to(self.device, non_blocking=True)
                        
                    # Vectorized computation without intermediate CPU transfers
                    with torch.no_grad():
                        batch_anomaly_scores = self.compute_anomaly_scores(batch)
                        batch_gat_probs = self.compute_gat_probabilities(batch)
                        
                        # Extract labels efficiently (keep on GPU)
                        if hasattr(batch, 'y') and batch.y.shape[0] == batch.num_graphs:
                            batch_labels = batch.y
                        else:
                            # Handle per-node labels -> per-graph labels
                            num_graphs = batch.batch.max().item() + 1
                            batch_labels = torch.zeros(num_graphs, device=self.device, dtype=batch.y.dtype)
                            
                            for graph_idx in range(num_graphs):
                                node_mask = (batch.batch == graph_idx)
                                if node_mask.any():
                                    batch_labels[graph_idx] = batch.y[node_mask].max()
                    
                    # Accumulate on GPU (no CPU transfer yet)
                    device_tensors['anomaly_scores'].append(batch_anomaly_scores)
                    device_tensors['gat_probs'].append(batch_gat_probs)
                    device_tensors['labels'].append(batch_labels)
                    
                    samples_processed += batch.num_graphs
                    
                    # Update progress less frequently for speed
                    if batch_idx % 50 == 0:
                        pbar.set_postfix({
                            'samples': f"{samples_processed:,}",
                            'gpu_util': f"{torch.cuda.utilization():.0f}%" if self.device.type == 'cuda' else "N/A"
                        })
                    
                    if max_samples and samples_processed >= max_samples:
                        break
                    
                    # Less frequent GPU cache clearing
                    if batch_idx % 100 == 0 and batch_idx > 0:
                        torch.cuda.empty_cache()
        
        # Single GPU→CPU transfer at the end (minimizes transfer overhead)
        print("📥 Transferring results from GPU to CPU...")
        anomaly_scores = torch.cat(device_tensors['anomaly_scores']).cpu().numpy().tolist()
        gat_probabilities = torch.cat(device_tensors['gat_probs']).cpu().numpy().tolist()
        labels = torch.cat(device_tensors['labels']).cpu().numpy().tolist()
        
        # Clean up GPU memory
        del device_tensors
        torch.cuda.empty_cache()
        
        return anomaly_scores, gat_probabilities, labels