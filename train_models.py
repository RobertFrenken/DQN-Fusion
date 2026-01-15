"""
CAN-Graph Model Training - Clean Lightning Implementation

A focused training pipeline that:
1. Loads data with optimal CPU workers
2. Uses Lightning's real Tuner for batch size optimization  
3. Trains models with Lightning's proven infrastructure
4. Handles special cases: autoencoder (normal samples only), knowledge distillation, fusion
5. Configuration managed through Hydra YAML system
"""

import os
import sys
from pathlib import Path
import logging
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf

# Suppress pynvml warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader  # Use PyTorch Geometric's DataLoader
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from src.preprocessing.preprocessing import graph_creation, GraphDataset
from src.models.models import GATWithJK, GraphAutoencoderNeighborhood

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CANGraphLightningModule(pl.LightningModule):
    """
    Lightning Module for CAN intrusion detection models.
    Handles GAT, VGAE, and special training cases (autoencoder, knowledge distillation).
    """
    
    def __init__(self, model_config: DictConfig, training_config: DictConfig, 
                 model_type: str = "gat", training_mode: str = "normal", num_ids: int = 1000):
        super().__init__()
        
        self.save_hyperparameters()
        self.model_config = model_config
        self.training_config = training_config
        self.model_type = model_type
        self.training_mode = training_mode  # normal, autoencoder, knowledge_distillation, fusion
        self.num_ids = num_ids
        
        # Lightning Tuner needs this attribute to modify batch size
        self.batch_size = training_config.batch_size
        
        # Create the model
        self.model = self._create_model()
        
        # For knowledge distillation
        self.teacher_model = None
        if training_mode == "knowledge_distillation":
            self.setup_knowledge_distillation()
    
    def _create_model(self):
        """Create the appropriate model based on configuration."""
        if self.model_type == "gat":
            # Handle both old dict config and new dataclass config
            if hasattr(self.model_config, 'gat'):
                # Old config format
                gat_params = dict(self.model_config.gat)
            else:
                # New dataclass format - convert to dict
                gat_params = {
                    'input_dim': self.model_config.input_dim,
                    'hidden_channels': self.model_config.hidden_channels,
                    'output_dim': self.model_config.output_dim,
                    'num_layers': self.model_config.num_layers,
                    'heads': self.model_config.heads,
                    'dropout': self.model_config.dropout,
                    'num_fc_layers': self.model_config.num_fc_layers,
                    'embedding_dim': self.model_config.embedding_dim,
                }
            
            # Rename parameters to match model signature
            gat_params['in_channels'] = gat_params.pop('input_dim')
            gat_params['out_channels'] = gat_params.pop('output_dim')
            
            # Remove parameters that GATWithJK doesn't use
            for unused_param in ['use_jumping_knowledge', 'jk_mode', 'use_residual', 'use_batch_norm', 'activation']:
                gat_params.pop(unused_param, None)
            
            # Add num_ids from dataset
            gat_params['num_ids'] = self.num_ids
            
            return GATWithJK(**gat_params)
        elif self.model_type == "vgae":
            # Handle both old dict config and new dataclass config
            if hasattr(self.model_config, 'vgae'):
                vgae_params = dict(self.model_config.vgae)
            else:
                # New dataclass format - convert to dict with correct parameter names
                vgae_params = {
                    'in_channels': self.model_config.input_dim,  # Note: in_channels not input_dim
                    'hidden_dim': self.model_config.hidden_channels,  # Note: hidden_dim not hidden_channels
                    'latent_dim': getattr(self.model_config, 'latent_dim', 32),
                    'num_ids': self.num_ids,
                }
            
            # Ensure num_ids is added if not present
            if 'num_ids' not in vgae_params:
                vgae_params['num_ids'] = self.num_ids
                
            return GraphAutoencoderNeighborhood(**vgae_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def setup_knowledge_distillation(self):
        """Setup teacher model for knowledge distillation."""
        if not self.training_config.teacher_model_path:
            raise ValueError("teacher_model_path is required for knowledge distillation mode")
            
        logger.info(f"Loading teacher model from: {self.training_config.teacher_model_path}")
        
        # Create teacher model with same architecture but potentially different size
        teacher_model_config = self.model_config.copy()
        self.teacher_model = self._create_model()
        
        # Load teacher weights
        teacher_path = Path(self.training_config.teacher_model_path)
        if not teacher_path.exists():
            raise FileNotFoundError(f"Teacher model not found: {teacher_path}")
            
        # Load state dict
        teacher_state = torch.load(teacher_path, map_location='cpu')
        
        # Handle different save formats
        if 'state_dict' in teacher_state:
            # Lightning checkpoint format
            state_dict = teacher_state['state_dict']
            # Remove 'model.' prefix if present
            state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v 
                         for k, v in state_dict.items()}
        else:
            # Direct state dict format
            state_dict = teacher_state
            
        self.teacher_model.load_state_dict(state_dict)
        self.teacher_model.eval()  # Set to eval mode
        
        # Freeze teacher parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Setup teacher output caching for memory optimization
        self.teacher_cache = {}
        self.cache_counter = 0
        self.clear_cache_every_n_steps = self.training_config.get('memory_optimization', {}).get('clear_cache_every_n_steps', 100)
        
        # Scale student model if configured
        student_scale = self.training_config.get('student_model_scale', 1.0)
        if student_scale != 1.0:
            self._scale_student_model(student_scale)
            logger.info(f"Student model scaled by factor: {student_scale}")
        
        logger.info("Knowledge distillation setup complete")
    
    def _scale_student_model(self, scale: float):
        """Scale student model size for distillation."""
        if self.model_type == "gat":
            # Scale hidden dimensions
            original_hidden = self.model_config.gat.hidden_channels
            scaled_hidden = max(8, int(original_hidden * scale))  # Minimum 8 channels
            self.model_config.gat.hidden_channels = scaled_hidden
            
            # Recreate model with scaled config
            self.model = self._create_model()
            logger.info(f"Student GAT hidden channels scaled: {original_hidden} -> {scaled_hidden}")
    
    def _get_teacher_output_cached(self, batch, use_cache=True):
        """Get teacher output with optional caching for memory efficiency."""
        if not use_cache or self.training_config.get('memory_optimization', {}).get('use_teacher_cache', True) == False:
            with torch.no_grad():
                if self.model_type == "vgae":
                    return self.teacher_model(batch.x, batch.edge_index, batch.batch)
                else:
                    return self.teacher_model(batch)
        
        # Simple batch-based caching (could be improved with more sophisticated hashing)
        batch_hash = hash(str(batch.x.shape) + str(batch.edge_index.shape) + str(batch.batch.max().item()))
        
        if batch_hash in self.teacher_cache:
            return self.teacher_cache[batch_hash]
        
        with torch.no_grad():
            if self.model_type == "vgae":
                teacher_output = self.teacher_model(batch.x, batch.edge_index, batch.batch)
            else:
                teacher_output = self.teacher_model(batch)
            self.teacher_cache[batch_hash] = teacher_output
            
        # Clear cache periodically to avoid memory buildup
        self.cache_counter += 1
        if self.cache_counter % self.clear_cache_every_n_steps == 0:
            self.teacher_cache.clear()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return teacher_output
    
    def forward(self, x):
        """Forward pass - handles different model calling conventions."""
        if self.model_type == "vgae":
            # VGAE models expect separate arguments
            return self.model(x.x, x.edge_index, x.batch)
        else:
            # GAT models expect the full batch object
            return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step - handles different training modes."""
        
        if self.training_mode == "autoencoder":
            return self._autoencoder_training_step(batch, batch_idx)
        elif self.training_mode == "knowledge_distillation":
            return self._knowledge_distillation_step(batch, batch_idx)
        elif self.training_mode == "fusion":
            return self._fusion_training_step(batch, batch_idx)
        else:
            return self._normal_training_step(batch, batch_idx)
    
    def _normal_training_step(self, batch, batch_idx):
        """Standard training step."""
        output = self.model(batch) if self.model_type == "gat" else self.forward(batch)
        loss = self._compute_loss(output, batch)
        
        self.log('train_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss
    
    def _autoencoder_training_step(self, batch, batch_idx):
        """Autoencoder training - only use normal samples."""
        # Filter to only normal samples (label == 0)
        if hasattr(batch, 'y'):
            normal_mask = batch.y == 0
            if normal_mask.sum() == 0:
                # No normal samples in this batch, skip
                return None
            
            # Create batch with only normal samples
            filtered_batch = self._filter_batch_by_mask(batch, normal_mask)
            output = self.forward(filtered_batch)
            loss = self._compute_autoencoder_loss(output, filtered_batch)
        else:
            # No labels, assume all are normal
            output = self.forward(batch)
            loss = self._compute_autoencoder_loss(output, batch)
        
        self.log('train_autoencoder_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss
    
    def _knowledge_distillation_step(self, batch, batch_idx):
        """Knowledge distillation training step."""
        # Student forward pass
        student_output = self.forward(batch)
        
        # Teacher forward pass with caching
        teacher_output = self._get_teacher_output_cached(batch)
        
        # Distillation loss
        distillation_loss = self._compute_distillation_loss(
            student_output, teacher_output, batch
        )
        
        # Additional logging for knowledge distillation
        if self.training_config.get('log_teacher_student_comparison', True):
            # Log teacher-student output similarity
            with torch.no_grad():
                if hasattr(batch, 'y'):  # For classification
                    # Handle different output formats
                    if isinstance(teacher_output, tuple):
                        teacher_logits = teacher_output[1]  # canid_logits for VGAE
                        student_logits = student_output[1] if isinstance(student_output, tuple) else student_output
                    else:
                        teacher_logits = teacher_output
                        student_logits = student_output[1] if isinstance(student_output, tuple) else student_output
                        
                    teacher_acc = (teacher_logits.argmax(dim=-1) == batch.y).float().mean()
                    student_acc = (student_logits.argmax(dim=-1) == batch.y).float().mean()
                    self.log('teacher_accuracy', teacher_acc, prog_bar=False, batch_size=batch.y.size(0))
                    self.log('student_accuracy', student_acc, prog_bar=False, batch_size=batch.y.size(0))
                    self.log('accuracy_gap', teacher_acc - student_acc, prog_bar=False, batch_size=batch.y.size(0))
                
                # Log output similarity (cosine similarity)
                teacher_flat = teacher_output[0].flatten() if isinstance(teacher_output, tuple) else teacher_output.flatten()
                student_flat = student_output[0].flatten() if isinstance(student_output, tuple) else student_output.flatten()
                similarity = F.cosine_similarity(teacher_flat.unsqueeze(0), student_flat.unsqueeze(0))
                self.log('teacher_student_similarity', similarity, prog_bar=False, batch_size=batch.y.size(0))
        
        self.log('train_distillation_loss', distillation_loss, prog_bar=True, batch_size=batch.y.size(0))
        return distillation_loss
    
    def _fusion_training_step(self, batch, batch_idx):
        """Fusion training step - handles multiple model outputs."""
        # This will be implemented when we get to fusion training
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)
        
        self.log('train_fusion_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)
        
        self.log('val_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)
        
        self.log('test_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss
    
    def _compute_loss(self, output, batch):
        """Compute standard loss."""
        if self.model_type == "vgae":
            # VGAE outputs: (cont_out, canid_logits, neighbor_logits, z, kl_loss)
            cont_out, canid_logits, neighbor_logits, z, kl_loss = output
            
            # For VGAE, we should only use classification loss for supervised tasks
            # In autoencoder mode, this shouldn't be called - use _compute_autoencoder_loss instead
            if hasattr(batch, 'y') and self.training_mode != "autoencoder":
                # Classification loss using graph-level labels (not CAN ID logits)
                # This would need proper graph-level prediction from VGAE
                # For now, return reconstruction loss as fallback
                reconstruction_loss = nn.functional.mse_loss(cont_out, batch.x[:, 1:])
                canid_loss = nn.functional.cross_entropy(canid_logits, batch.x[:, 0].long())
                total_loss = reconstruction_loss + 0.1 * canid_loss + 0.01 * kl_loss
                return total_loss
            else:
                # Reconstruction loss
                reconstruction_loss = nn.functional.mse_loss(cont_out, batch.x[:, 1:])
                canid_loss = nn.functional.cross_entropy(canid_logits, batch.x[:, 0].long())
                total_loss = reconstruction_loss + 0.1 * canid_loss + 0.01 * kl_loss
                return total_loss
        else:
            # GAT standard output
            if hasattr(batch, 'y'):
                return nn.functional.cross_entropy(output, batch.y)
            else:
                # For unsupervised cases
                return nn.functional.mse_loss(output, batch.x)
    
    def _compute_autoencoder_loss(self, output, batch):
        """Compute reconstruction loss for autoencoder."""
        if self.model_type == "vgae":
            # VGAE outputs: (cont_out, canid_logits, neighbor_logits, z, kl_loss)
            cont_out, canid_logits, neighbor_logits, z, kl_loss = output
            
            # For autoencoder training, we reconstruct node features, not classify graphs
            # Reconstruction loss for continuous features (excluding CAN ID)
            continuous_features = batch.x[:, 1:]  # Skip CAN ID column
            reconstruction_loss = nn.functional.mse_loss(cont_out, continuous_features)
            
            # CAN ID reconstruction loss - predict the CAN ID from node features
            canid_targets = batch.x[:, 0].long()  # CAN IDs from input features
            canid_loss = nn.functional.cross_entropy(canid_logits, canid_targets)
            
            # Combined loss: reconstruction + CAN ID prediction + KL regularization
            total_loss = reconstruction_loss + 0.1 * canid_loss + 0.01 * kl_loss
            return total_loss
        else:
            # GAT autoencoder (if implemented)
            return nn.functional.mse_loss(output, batch.x)
    
    def _compute_distillation_loss(self, student_output, teacher_output, batch):
        """Compute knowledge distillation loss."""
        temperature = self.training_config.get('distillation_temperature', 4.0)
        alpha = self.training_config.get('distillation_alpha', 0.7)
        
        # Handle different output types
        if hasattr(batch, 'y'):  # Supervised case
            # Hard targets loss (standard task loss)
            hard_loss = self._compute_loss(student_output, batch)
            
            # Soft targets loss (knowledge distillation)
            if student_output.dim() > 1 and student_output.size(-1) > 1:  # Classification
                soft_targets = torch.softmax(teacher_output / temperature, dim=-1)
                soft_prob = torch.log_softmax(student_output / temperature, dim=-1)
                soft_loss = nn.functional.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
            else:  # Regression-like output
                soft_loss = nn.functional.mse_loss(student_output, teacher_output)
            
            # Combined loss
            total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
            
            # Log components for monitoring
            self.log('hard_loss', hard_loss, prog_bar=False, batch_size=student_logits.size(0))
            self.log('soft_loss', soft_loss, prog_bar=False, batch_size=student_logits.size(0))
            
        else:  # Unsupervised case (e.g., autoencoders)
            # Direct output matching
            total_loss = nn.functional.mse_loss(student_output, teacher_output)
        
        return total_loss
    
    def _filter_batch_by_mask(self, batch, mask):
        """Filter graph batch by mask (for autoencoder normal samples)."""
        # Implementation depends on your batch structure
        # This is a placeholder - adapt to your GraphDataset structure
        return batch
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers using Hydra configuration."""
        
        # Handle both old dict config and new dataclass config
        def get_config_value(key, default=None):
            if hasattr(self.training_config, 'get'):
                return self.training_config.get(key, default)
            else:
                return getattr(self.training_config, key, default)
        
        # Get optimizer configuration
        if hasattr(self.training_config, 'optimizer'):
            # New dataclass format with nested optimizer config
            optimizer_config = self.training_config.optimizer
            optimizer_name = optimizer_config.name.lower()
            learning_rate = optimizer_config.lr
            weight_decay = optimizer_config.weight_decay
        else:
            # Old dict format or simple dataclass
            optimizer_name = get_config_value('optimizer', 'adam').lower()
            learning_rate = get_config_value('learning_rate', 0.001)
            weight_decay = get_config_value('weight_decay', 0.0001)
        
        # Create optimizer based on config
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = get_config_value('momentum', 0.9)
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Setup scheduler if configured
        use_scheduler = get_config_value('use_scheduler', False)
        if hasattr(self.training_config, 'scheduler') and self.training_config.scheduler:
            # New dataclass format with nested scheduler config
            scheduler_config = self.training_config.scheduler
            use_scheduler = scheduler_config.use_scheduler
            
        if use_scheduler:
            if hasattr(self.training_config, 'scheduler'):
                # New dataclass format
                scheduler_config = self.training_config.scheduler
                scheduler_type = scheduler_config.scheduler_type.lower()
                scheduler_params = scheduler_config.params
                
                if scheduler_type == 'cosine':
                    T_max = scheduler_params.get('T_max', self.training_config.max_epochs)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
                elif scheduler_type == 'step':
                    step_size = scheduler_params.get('step_size', 30)
                    gamma = scheduler_params.get('gamma', 0.1)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                elif scheduler_type == 'exponential':
                    gamma = scheduler_params.get('gamma', 0.95)
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
                else:
                    raise ValueError(f"Unsupported scheduler: {scheduler_type}")
            else:
                # Old dict format
                scheduler_type = get_config_value('scheduler_type', 'cosine').lower()
                scheduler_params = get_config_value('scheduler_params', {})
                
                if scheduler_type == 'cosine':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, 
                        T_max=scheduler_params.get('T_max', self.training_config.max_epochs)
                    )
                elif scheduler_type == 'step':
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer,
                        step_size=scheduler_params.get('step_size', 30),
                        gamma=scheduler_params.get('gamma', 0.1)
                    )
                elif scheduler_type == 'exponential':
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer,
                        gamma=scheduler_params.get('gamma', 0.95)
                    )
                else:
                    raise ValueError(f"Unsupported scheduler: {scheduler_type}")
                
            return [optimizer], [scheduler]
        
        return optimizer


def load_dataset(dataset_name: str, config, force_rebuild_cache: bool = False):
    """Load and prepare dataset"""
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Use data_path from config if available, otherwise fallback to dataset name
    if hasattr(config.dataset, 'data_path') and config.dataset.data_path:
        dataset_path = config.dataset.data_path
    else:
        # Try multiple possible dataset paths
        possible_paths = [
            f"datasets/can-train-and-test-v1.5/{dataset_name}",
            f"datasets/{dataset_name}",
            f"../datasets/{dataset_name}",
            f"data/{dataset_name}"
        ]
        
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                # Check if it has CSV files
                import glob
                csv_files = glob.glob(os.path.join(path, '**', '*train_*.csv'), recursive=True)
                if csv_files:
                    dataset_path = path
                    logger.info(f"Found valid dataset path: {dataset_path} with {len(csv_files)} CSV files")
                    break
                else:
                    logger.warning(f"Path exists but no CSV files found: {path}")
        
        if not dataset_path:
            dataset_path = f"datasets/{dataset_name}"  # Fallback
            logger.error(f"No valid dataset path found! Using fallback: {dataset_path}")
        
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Check for cached processed data - handle both dict and dataclass configs
    if hasattr(config.dataset, 'get'):  # Dictionary config
        cache_enabled = config.dataset.get('preprocessing', {}).get('cache_processed_data', True)
        cache_dir = config.dataset.get('cache_dir', f"datasets/cache/{dataset_name}")
    else:  # Dataclass config
        cache_enabled = getattr(config.dataset, 'cache_processed_data', True)
        cache_dir = getattr(config.dataset, 'cache_dir', None)
        if cache_dir is None:
            cache_dir = f"datasets/cache/{dataset_name}"
    
    cache_file = Path(cache_dir) / "processed_graphs.pt"
    id_mapping_file = Path(cache_dir) / "id_mapping.pkl"
    
    if cache_enabled and cache_file.exists() and id_mapping_file.exists() and not force_rebuild_cache:
        try:
            logger.info(f"Loading cached processed data from {cache_file}")
            
            # Load cached graphs and ID mapping
            import pickle
            graphs = torch.load(cache_file)
            with open(id_mapping_file, 'rb') as f:
                id_mapping = pickle.load(f)
                
            logger.info(f"Loaded {len(graphs)} cached graphs with {len(id_mapping)} unique IDs")
            
            # Validate cache size - detect if cache is corrupted or incomplete
            expected_sizes = {
                'set_01': 300000, 'set_02': 400000, 'set_03': 330000, 'set_04': 240000,
                'hcrl_sa': 18000, 'hcrl_ch': 290000
            }
            
            if dataset_name in expected_sizes:
                expected = expected_sizes[dataset_name]
                actual = len(graphs)
                if actual < expected * 0.1:  # Less than 10% of expected
                    logger.warning(f"🚨 CACHE ISSUE DETECTED: Only {actual} graphs found, expected ~{expected}")
                    logger.warning(f"Cache appears corrupted or incomplete. Rebuilding from scratch.")
                    graphs, id_mapping = None, None
                elif actual < expected * 0.5:  # Less than 50% of expected  
                    logger.warning(f"⚠️  Cache has fewer graphs than expected: {actual} vs ~{expected}")
                    logger.warning(f"This might be a debug/test cache. Use --force-rebuild to recreate.")
                else:
                    logger.info(f"✅ Cache size looks good: {actual} graphs (expected ~{expected})")
            
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}. Processing from scratch.")
            graphs, id_mapping = None, None
    else:
        graphs, id_mapping = None, None
    
    # Process data if not cached or cache loading failed
    if graphs is None or id_mapping is None:
        rebuild_reason = "forced rebuild" if force_rebuild_cache else "processing dataset from scratch"
        logger.info(f"Processing dataset: {rebuild_reason}...")
        logger.info(f"Dataset path: {dataset_path}")
        
        # Check if dataset path exists and log file count
        if os.path.exists(dataset_path):
            import glob
            # Search for CSV files in train folders (train_01_attack_free, train_02_with_attacks, etc.)
            csv_files = []
            for train_folder in ['train_01_attack_free', 'train_02_with_attacks', 'train_*']:
                pattern = os.path.join(dataset_path, train_folder, '*.csv')
                csv_files.extend(glob.glob(pattern))
            
            # Also try the generic pattern for other dataset structures
            if not csv_files:
                csv_files = glob.glob(os.path.join(dataset_path, '**', '*train*.csv'), recursive=True)
                
            logger.info(f"Found {len(csv_files)} CSV files in {dataset_path}")
            if len(csv_files) == 0:
                logger.error(f"🚨 NO CSV FILES FOUND in {dataset_path}!")
                logger.error(f"Available files:")
                all_files = glob.glob(os.path.join(dataset_path, '**', '*.csv'), recursive=True)[:20]
                for f in all_files:
                    logger.error(f"  - {f}")
                    
                # Check for training folders
                train_folders = glob.glob(os.path.join(dataset_path, 'train*'))
                if train_folders:
                    logger.error(f"Found training folders: {train_folders}")
                    for folder in train_folders:
                        folder_files = glob.glob(os.path.join(folder, '*.csv'))
                        logger.error(f"  {folder}: {len(folder_files)} CSV files")
                        
                raise FileNotFoundError(f"No train CSV files found in {dataset_path}")
            elif len(csv_files) < 50:  # Log first few files for debugging
                logger.info(f"CSV files found: {csv_files[:10]}")
        else:
            logger.error(f"Dataset path does not exist: {dataset_path}")
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
            
        # Use sequential processing - Lightning DataLoader handles parallelism
        graphs, id_mapping = graph_creation(dataset_path, 'train_', return_id_mapping=True)
        
        # Save to cache if enabled
        if cache_enabled:
            import pickle
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving processed data to cache: {cache_file}")
            torch.save(graphs, cache_file)
            with open(id_mapping_file, 'wb') as f:
                pickle.dump(id_mapping, f)
    
    dataset = GraphDataset(graphs)
    logger.info(f"📊 Created dataset with {len(dataset)} total graphs")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"📊 Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation")
    
    # Return datasets and number of unique IDs
    num_ids = len(id_mapping) if id_mapping else 1000
    return train_dataset, val_dataset, num_ids


def create_dataloaders(train_dataset, val_dataset, batch_size: int):
    """Create optimized dataloaders - standalone function."""
    num_workers = min(os.cpu_count() or 1, 8)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    
    logger.info(f"Created dataloaders with {num_workers} workers")
    return train_loader, val_loader


class CANGraphDataModule(pl.LightningDataModule):
    """Lightning DataModule for efficient batch size tuning."""
    
    def __init__(self, train_dataset, val_dataset, batch_size: int):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = min(os.cpu_count() or 1, 8)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    """
    Main training function - uses Lightning Trainer directly, no custom wrappers.
    """
    logger.info("Starting CAN-Graph training")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    # Load and prepare dataset
    train_dataset, val_dataset, num_ids = load_dataset(config.dataset.name, config)
    
    # Create Lightning module
    model = CANGraphLightningModule(
        model_config=config.model,
        training_config=config.training,
        model_type=config.model.type,
        training_mode=config.training.get('mode', 'normal'),
        num_ids=num_ids
    )
    
    # Find Optimal Batch Size
    if config.training.get('optimize_batch_size', False):
        logger.info("Finding optimal batch size with Lightning's Tuner")
        
        # Create temporary DataModule
        temp_datamodule = CANGraphDataModule(train_dataset, val_dataset, model.batch_size)
        
        trainer = pl.Trainer(
            accelerator='auto',
            devices='auto',
            precision='32-true',
            max_epochs=1,
            enable_checkpointing=False,
            logger=False
        )
        
        tuner = Tuner(trainer)
        try:
            tuner.scale_batch_size(
                model,
                datamodule=temp_datamodule,
                mode='power',
                steps_per_trial=3,
                init_val=4096,
                max_trials=10
            )
            
            optimized_batch_size = model.batch_size
            logger.info(f"Batch size optimized to: {optimized_batch_size}")
            
        except Exception as e:
            logger.warning(f"Batch size optimization failed: {e}. Using original batch size.")
    
    # Conservative adjustment for knowledge distillation
    if config.training.get('mode') == 'knowledge_distillation':
        model.batch_size = max(model.batch_size // 2)  # KD needs more memory
        logger.info(f"Reduced batch size for knowledge distillation: {model.batch_size}")
    
    # Create dataloaders with optimized batch size
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, model.batch_size
    )
    
    # Setup loggers based on configuration
    loggers = []
    csv_logger = CSVLogger(
        save_dir=config.get('log_dir', 'outputs/lightning_logs'),
        name=f"{config.model.type}_{config.training.get('mode', 'normal')}_training"
    )
    loggers.append(csv_logger)
    
    # Add TensorBoard logger if enabled (optional)
    if config.get('logging', {}).get('enable_tensorboard', False):
        tb_logger = TensorBoardLogger(
            save_dir=config.get('log_dir', 'outputs/lightning_logs'),
            name=f"{config.model.type}_{config.training.get('mode', 'normal')}_training"
        )
        loggers.append(tb_logger)
        logger.info("TensorBoard logging enabled. Run 'tensorboard --logdir outputs/lightning_logs' to view.")
    
    # Setup Lignhtning Trainer
    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto', 
        precision='32-true',  # Use valid precision instead of 'auto'
        max_epochs=config.training.max_epochs,
        gradient_clip_val=config.training.get('gradient_clip_val', 1.0),  # Clip gradients for stability
        logger=loggers,
        callbacks=[
            ModelCheckpoint(
                dirpath='saved_models/lightning_checkpoints',
                filename=f'{config.model.type}_{{epoch:02d}}_{{val_loss:.2f}}',
                save_top_k=3,
                monitor='val_loss',
                mode='min',
                save_last=True
            ),
        ],
        enable_checkpointing=True,
        log_every_n_steps=config.training.get('log_every_n_steps', 50),
    )
    
    # Train with Lightning Trainer
    logger.info("Training with Lightning Trainer")
    trainer.fit(model, train_loader, val_loader)
    
    # Test if enabled
    if config.training.get('run_test', True):
        test_results = trainer.test(model, val_loader)
        logger.info(f"Test results: {test_results}")
    
    # Save model
    model_name = f"{config.model.type}_{config.training.get('mode', 'normal')}_{config.dataset.name}.pth"
    save_path = Path("saved_models") / model_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to: {save_path}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
