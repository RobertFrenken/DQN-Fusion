"""
Model Management for CAN-Graph Training

Handles model loading, saving, and state dict operations with strict
no-pickle policy for production models.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model checkpoints and state dict operations."""
    
    @staticmethod
    def save_state_dict(
        model_obj: Union[nn.Module, Any],
        save_dir: Path,
        filename: str,
        backup_existing: bool = True
    ) -> Path:
        """
        Save a model's state_dict safely (no pickle).
        
        Supports:
        - Plain nn.Module
        - Lightning modules (extracts .model attribute)
        - Fusion agents (dicts containing state_dicts)
        
        Args:
            model_obj: Model object to save
            save_dir: Directory to save in
            filename: Filename (should end with .pth)
            backup_existing: Whether to backup existing file
            
        Returns:
            Path to saved model file
            
        Raises:
            RuntimeError: If state dict extraction fails
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename

        # Backup existing file
        if backup_existing and save_path.exists():
            backup = save_path.with_suffix(save_path.suffix + '.bak')
            if not backup.exists():
                shutil.copy2(save_path, backup)
                logger.info(f"Backed up existing model: {save_path.name} → {backup.name}")

        # Determine state to save
        state_to_save = None
        
        try:
            # Check for fusion agent with explicit save method
            if hasattr(model_obj, 'fusion_agent') and hasattr(model_obj.fusion_agent, 'q_network'):
                # Extract Q-network state dict for DQN fusion agents
                state_to_save = model_obj.fusion_agent.q_network.state_dict()
                logger.debug("Extracted state dict from fusion_agent.q_network")
                
            # Check for standard state_dict method
            elif hasattr(model_obj, 'state_dict') and callable(model_obj.state_dict):
                # Prefer inner .model if it exists (Lightning modules)
                if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'state_dict'):
                    state_to_save = model_obj.model.state_dict()
                    logger.debug("Extracted state dict from model.model")
                else:
                    state_to_save = model_obj.state_dict()
                    logger.debug("Extracted state dict from model")
                    
            else:
                raise RuntimeError(
                    f"Cannot extract state_dict from {type(model_obj)}. "
                    "Model must have state_dict() method or be a fusion agent."
                )

            # Save state dict
            torch.save(state_to_save, save_path)
            logger.info(f"💾 Model state_dict saved to {save_path}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save state_dict: {e}")
            raise RuntimeError(f"Model save failed: {e}") from e
    
    @staticmethod
    def load_state_dict(
        checkpoint_path: Union[str, Path],
        strict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Load a state dict from checkpoint file.
        
        Handles both pure state_dicts and Lightning checkpoint dicts.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to enforce strict loading
            
        Returns:
            State dict ready for model.load_state_dict()
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=False  # Allow complex objects temporarily
            )
            
            # Extract state dict if wrapped in checkpoint dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    logger.debug(f"Extracted 'state_dict' from checkpoint dict")
                else:
                    # Assume the dict itself is the state dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            logger.info(f"✓ Loaded state dict from {checkpoint_path.name}")
            return state_dict
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint loading failed: {e}") from e
    
    @staticmethod
    def sanitize_for_json(obj: Any) -> Any:
        """
        Convert tensors/numpy arrays to JSON-serializable types.
        
        Used for saving hyperparameters and metrics.
        
        Args:
            obj: Object to sanitize
            
        Returns:
            JSON-serializable version of obj
        """
        if isinstance(obj, dict):
            return {k: ModelManager.sanitize_for_json(v) for k, v in obj.items()}
        
        if isinstance(obj, list):
            return [ModelManager.sanitize_for_json(v) for v in obj]
        
        if isinstance(obj, tuple):
            return tuple(ModelManager.sanitize_for_json(v) for v in obj)
        
        # Handle numpy arrays
        if hasattr(obj, 'tolist') and not isinstance(obj, (str, bytes)):
            try:
                return obj.tolist()
            except (AttributeError, TypeError):
                pass
        
        # Handle torch tensors
        if isinstance(obj, torch.Tensor):
            try:
                return obj.detach().cpu().tolist()
            except Exception:
                return str(obj)
        
        # Handle numpy scalars
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
        except ImportError:
            pass
        
        return obj
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """
        Get summary information about a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dict with model info (params, size, device)
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size in MB
        param_size = sum(p.element_size() * p.numel() for p in model.parameters())
        buffer_size = sum(b.element_size() * b.numel() for b in model.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        # Get device
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = 'cpu'
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'size_mb': round(size_mb, 2),
            'device': str(device),
            'model_type': type(model).__name__
        }
    
    @staticmethod
    def verify_state_dict_compatible(
        state_dict: Dict[str, torch.Tensor],
        model: nn.Module,
        strict: bool = True
    ) -> bool:
        """
        Verify that a state dict is compatible with a model.
        
        Args:
            state_dict: State dict to check
            model: Model to check against
            strict: Whether to require exact match
            
        Returns:
            True if compatible, False otherwise
        """
        model_keys = set(model.state_dict().keys())
        state_keys = set(state_dict.keys())
        
        missing = model_keys - state_keys
        unexpected = state_keys - model_keys
        
        if strict:
            if missing or unexpected:
                if missing:
                    logger.warning(f"Missing keys in state dict: {missing}")
                if unexpected:
                    logger.warning(f"Unexpected keys in state dict: {unexpected}")
                return False
        
        return True
