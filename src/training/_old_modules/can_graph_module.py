"""
CAN-Graph Lightning Module
Implements the core PyTorch Lightning model for CAN intrusion detection.
Handles GAT, VGAE, autoencoder, knowledge distillation, and fusion training modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from src.models.models import GATWithJK
from src.models.vgae import GraphAutoencoderNeighborhood

class CANGraphLightningModule(pl.LightningModule):
    """
    Lightning Module for CAN intrusion detection models.
    Handles GAT, VGAE, and special training cases (autoencoder, knowledge distillation).
    """
    def __init__(self, model_config, training_config, model_type="gat", training_mode="normal", num_ids=1000):
        super().__init__()
        # save_hyperparameters is provided by LightningModule in some versions; guard for compatibility in tests
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters()
        self.model_config = model_config
        self.training_config = training_config
        self.model_type = model_type
        self.training_mode = training_mode
        self.num_ids = num_ids
        self.batch_size = training_config.batch_size
        self.model = self._create_model()
        self.teacher_model = None
        if training_mode == "knowledge_distillation":
            self.setup_knowledge_distillation()

    def _create_model(self):
        if self.model_type in ["gat", "gat_student"]:
            if hasattr(self.model_config, 'gat'):
                gat_params = dict(self.model_config.gat)
            else:
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
            gat_params['in_channels'] = gat_params.pop('input_dim')
            gat_params['out_channels'] = gat_params.pop('output_dim')
            for unused_param in ['use_jumping_knowledge', 'jk_mode', 'use_residual', 'use_batch_norm', 'activation']:
                gat_params.pop(unused_param, None)
            gat_params['num_ids'] = self.num_ids
            return GATWithJK(**gat_params)
        elif self.model_type in ["vgae", "vgae_student"]:
            # Build params from config and prefer explicit progressive `hidden_dims` if present
            if hasattr(self.model_config, 'hidden_dims'):
                hidden_dims = list(self.model_config.hidden_dims)
            elif hasattr(self.model_config, 'encoder_dims'):
                hidden_dims = list(self.model_config.encoder_dims)
            else:
                hidden_dims = None

            vgae_params = {
                'num_ids': self.num_ids,
                'in_channels': self.model_config.input_dim,
                'hidden_dims': hidden_dims,
                'latent_dim': getattr(self.model_config, 'latent_dim', (hidden_dims[-1] if hidden_dims else 32)),
                'encoder_heads': getattr(self.model_config, 'attention_heads', 4),
                'decoder_heads': getattr(self.model_config, 'attention_heads', 4),
                'embedding_dim': getattr(self.model_config, 'embedding_dim', 32),
                'dropout': getattr(self.model_config, 'dropout', 0.15),
                'batch_norm': getattr(self.model_config, 'batch_norm', True)
            }
            return GraphAutoencoderNeighborhood(**vgae_params)
        elif self.model_type in ["dqn", "dqn_student"]:
            # Use the DQN model factory functions to construct real DQN architectures
            from src.models.models import create_dqn_teacher, create_dqn_student
            if self.model_type == "dqn":
                return create_dqn_teacher()
            else:
                return create_dqn_student()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    # Training step dispatcher (required by Lightning) and normal training step
    def training_step(self, batch, batch_idx):
        """Dispatch to the appropriate training step based on `training_mode`."""
        if self.training_mode == "autoencoder":
            return self._autoencoder_training_step(batch, batch_idx)
        elif self.training_mode == "knowledge_distillation":
            return self._knowledge_distillation_step(batch, batch_idx)
        elif self.training_mode == "fusion":
            return self._fusion_training_step(batch, batch_idx)
        else:
            return self._normal_training_step(batch, batch_idx)

    def _normal_training_step(self, batch, batch_idx):
        """Standard training step for supervised GAT or VGAE when applicable."""
        if self.model_type == "gat":
            output = self.model(batch)
        else:
            output = self.forward(batch)

        base_loss = self._compute_loss(output, batch)
        try:
            batch_size = batch.y.size(0)
        except Exception:
            batch_size = None
        self.log('train_loss', base_loss, prog_bar=True, batch_size=batch_size)
        return base_loss

    # Training steps and loss computation
    def _autoencoder_training_step(self, batch, batch_idx):
        if hasattr(batch, 'y'):
            normal_mask = batch.y == 0
            if normal_mask.sum() == 0:
                return None
            filtered_batch = self._filter_batch_by_mask(batch, normal_mask)
            output = self.forward(filtered_batch)
            loss = self._compute_autoencoder_loss(output, filtered_batch)
        else:
            output = self.forward(batch)
            loss = self._compute_autoencoder_loss(output, batch)
        self.log('train_autoencoder_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss

    def _knowledge_distillation_step(self, batch, batch_idx):
        student_output = self.forward(batch)
        teacher_output = self._get_teacher_output_cached(batch)
        distillation_loss = self._compute_distillation_loss(student_output, teacher_output, batch)
        if self.training_config.get('log_teacher_student_comparison', True):
            with torch.no_grad():
                if hasattr(batch, 'y'):
                    if isinstance(teacher_output, tuple):
                        teacher_logits = teacher_output[1]
                        student_logits = student_output[1] if isinstance(student_output, tuple) else student_output
                    else:
                        teacher_logits = teacher_output
                        student_logits = student_output[1] if isinstance(student_output, tuple) else student_output
                    teacher_acc = (teacher_logits.argmax(dim=-1) == batch.y).float().mean()
                    student_acc = (student_logits.argmax(dim=-1) == batch.y).float().mean()
                    self.log('teacher_accuracy', teacher_acc, prog_bar=False, batch_size=batch.y.size(0))
                    self.log('student_accuracy', student_acc, prog_bar=False, batch_size=batch.y.size(0))
                    self.log('accuracy_gap', teacher_acc - student_acc, prog_bar=False, batch_size=batch.y.size(0))
                teacher_flat = teacher_output[0].flatten() if isinstance(teacher_output, tuple) else teacher_output.flatten()
                student_flat = student_output[0].flatten() if isinstance(student_output, tuple) else student_output.flatten()
                similarity = F.cosine_similarity(teacher_flat.unsqueeze(0), student_flat.unsqueeze(0))
                self.log('teacher_student_similarity', similarity, prog_bar=False, batch_size=batch.y.size(0))
        self.log('train_distillation_loss', distillation_loss, prog_bar=True, batch_size=batch.y.size(0))
        return distillation_loss

    def _fusion_training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)
        self.log('train_fusion_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)
        self.log('val_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss

    def forward(self, batch):
        """Forward dispatcher for the LightningModule.

        Accepts a batched Graph data object (with attributes `x`, `edge_index`, `batch`) and
        routes to the underlying `self.model` according to `self.model_type`.
        Returns whatever the underlying model returns so training/validation code can
        compute losses consistently.
        """
        # GAT models: expect the full Data object (and return logits)
        if self.model_type in ["gat", "gat_student"]:
            return self.model(batch)

        # VGAE models: GraphAutoencoderNeighborhood.forward(x, edge_index, batch)
        if self.model_type in ["vgae", "vgae_student"]:
            x = getattr(batch, 'x', None)
            edge_index = getattr(batch, 'edge_index', None)
            b = getattr(batch, 'batch', None)
            if x is None or edge_index is None or b is None:
                raise ValueError("Batch is missing required attributes for VGAE forward: x, edge_index, batch")
            return self.model(x, edge_index, b)

        # DQN or other tabular models: pass through features
        if self.model_type in ["dqn", "dqn_student"]:
            x = getattr(batch, 'x', None)
            if x is None:
                raise ValueError("Batch is missing 'x' required for DQN forward")
            # Flatten or select appropriate features for DQN
            return self.model(x)

        raise ValueError(f"Unsupported model_type in forward: {self.model_type}")

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self._compute_loss(output, batch)
        self.log('test_loss', loss, prog_bar=True, batch_size=batch.y.size(0))
        return loss

    def _compute_loss(self, output, batch):
        if self.model_type in ["vgae", "vgae_student"]:
            cont_out, canid_logits, neighbor_logits, z, kl_loss = output
            if hasattr(batch, 'y') and self.training_mode != "autoencoder":
                reconstruction_loss = nn.functional.mse_loss(cont_out, batch.x[:, 1:])
                canid_loss = nn.functional.cross_entropy(canid_logits, batch.x[:, 0].long())
                total_loss = reconstruction_loss + 0.1 * canid_loss + 0.01 * kl_loss
                return total_loss
            else:
                reconstruction_loss = nn.functional.mse_loss(cont_out, batch.x[:, 1:])
                canid_loss = nn.functional.cross_entropy(canid_logits, batch.x[:, 0].long())
                total_loss = reconstruction_loss + 0.1 * canid_loss + 0.01 * kl_loss
                return total_loss
        else:
            if hasattr(batch, 'y'):
                return nn.functional.cross_entropy(output, batch.y)
            else:
                return nn.functional.mse_loss(output, batch.x)

    def _compute_autoencoder_loss(self, output, batch):
        if self.model_type in ["vgae", "vgae_student"]:
            cont_out, canid_logits, neighbor_logits, z, kl_loss = output
            continuous_features = batch.x[:, 1:]
            reconstruction_loss = nn.functional.mse_loss(cont_out, continuous_features)
            canid_targets = batch.x[:, 0].long()
            canid_loss = nn.functional.cross_entropy(canid_logits, canid_targets)
            total_loss = reconstruction_loss + 0.1 * canid_loss + 0.01 * kl_loss
            return total_loss
        else:
            return nn.functional.mse_loss(output, batch.x)

    def _compute_distillation_loss(self, student_output, teacher_output, batch):
        temperature = self.training_config.get('distillation_temperature', 4.0)
        alpha = self.training_config.get('distillation_alpha', 0.7)
        if hasattr(batch, 'y'):
            hard_loss = self._compute_loss(student_output, batch)
            if student_output.dim() > 1 and student_output.size(-1) > 1:
                soft_targets = torch.softmax(teacher_output / temperature, dim=-1)
                soft_prob = torch.log_softmax(student_output / temperature, dim=-1)
                soft_loss = nn.functional.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
            else:
                soft_loss = nn.functional.mse_loss(student_output, teacher_output)
            total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        self.log('hard_loss', hard_loss, prog_bar=False, batch_size=student_output.size(0))
        self.log('soft_loss', soft_loss, prog_bar=False, batch_size=student_output.size(0))
        return total_loss

    def _filter_batch_by_mask(self, batch, mask):
        return batch

    def configure_optimizers(self):
        def get_config_value(key, default=None):
            if hasattr(self.training_config, 'get'):
                return self.training_config.get(key, default)
            else:
                return getattr(self.training_config, key, default)
        if hasattr(self.training_config, 'optimizer'):
            optimizer_config = self.training_config.optimizer
            optimizer_name = optimizer_config.name.lower()
            learning_rate = optimizer_config.lr
            weight_decay = optimizer_config.weight_decay
        else:
            optimizer_name = get_config_value('optimizer', 'adam').lower()
            learning_rate = get_config_value('learning_rate', 0.001)
            weight_decay = get_config_value('weight_decay', 0.0001)
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = get_config_value('momentum', 0.9)
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        use_scheduler = get_config_value('use_scheduler', False)
        if hasattr(self.training_config, 'scheduler') and self.training_config.scheduler:
            scheduler_config = self.training_config.scheduler
            use_scheduler = scheduler_config.use_scheduler
        if use_scheduler:
            if hasattr(self.training_config, 'scheduler'):
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
                scheduler_type = get_config_value('scheduler_type', 'cosine').lower()
                scheduler_params = get_config_value('scheduler_params', {})
                if scheduler_type == 'cosine':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_params.get('T_max', self.training_config.max_epochs))
                elif scheduler_type == 'step':
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_params.get('step_size', 30), gamma=scheduler_params.get('gamma', 0.1))
                elif scheduler_type == 'exponential':
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_params.get('gamma', 0.95))
                else:
                    raise ValueError(f"Unsupported scheduler: {scheduler_type}")
            return [optimizer], [scheduler]
        return optimizer
