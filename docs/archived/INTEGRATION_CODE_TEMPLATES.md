# Integration Code Templates

Copy-paste ready code templates for each integration step.

## Template 1: Dataset Wrapper Class

**File:** `src/data/datasets.py`

**What to add at the top:**

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class GraphDataset(Dataset):
    """PyTorch Dataset wrapper for graph data"""
    
    def __init__(self, x, edge_index=None, y=None):
        """
        Args:
            x: Node features (numpy array or tensor)
            edge_index: Edge indices (numpy array or tensor)
            y: Labels (numpy array or tensor, optional)
        """
        self.x = torch.tensor(x, dtype=torch.float32) if isinstance(x, np.ndarray) else x
        self.edge_index = torch.tensor(edge_index, dtype=torch.long) if isinstance(edge_index, np.ndarray) else edge_index
        self.y = torch.tensor(y, dtype=torch.long) if isinstance(y, np.ndarray) else y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        batch = {
            'x': self.x[idx] if len(self.x.shape) > 1 else self.x,
            'edge_index': self.edge_index,
        }
        if self.y is not None:
            batch['y'] = self.y[idx]
        return batch


class HCRLCHDataset:
    """HCRLCH Dataset loader with train/val/test split"""
    
    def __init__(self, data_path: str, split_ratio: tuple = (0.7, 0.15, 0.15), 
                 normalization: str = 'zscore', **kwargs):
        """
        Args:
            data_path: Path to dataset directory
            split_ratio: (train, val, test) split ratios
            normalization: 'zscore', 'minmax', or 'none'
        """
        self.data_path = data_path
        self.split_ratio = split_ratio
        self.normalization = normalization
        
        # YOUR EXISTING DATA LOADING LOGIC HERE
        # This is pseudocode - replace with your actual loading
        
        # Example:
        # x = np.load(f'{data_path}/x.npy')  # Node features
        # edge_index = np.load(f'{data_path}/edge_index.npy')  # Edges
        # y = np.load(f'{data_path}/y.npy')  # Labels (if available)
        
        # Apply normalization
        x = self._normalize(x, normalization)
        
        # Split data
        n_samples = len(x)
        train_size = int(n_samples * split_ratio[0])
        val_size = int(n_samples * split_ratio[1])
        
        train_x, val_x, test_x = (
            x[:train_size],
            x[train_size:train_size + val_size],
            x[train_size + val_size:]
        )
        
        # Create PyTorch Dataset objects
        self.train = GraphDataset(train_x, edge_index, y[:train_size] if y is not None else None)
        self.val = GraphDataset(val_x, edge_index, y[train_size:train_size + val_size] if y is not None else None)
        self.test = GraphDataset(test_x, edge_index, y[train_size + val_size:] if y is not None else None)
    
    def _normalize(self, x: np.ndarray, method: str) -> np.ndarray:
        """Normalize features"""
        if method == 'zscore':
            mean = x.mean(axis=0)
            std = x.std(axis=0) + 1e-8
            return (x - mean) / std
        elif method == 'minmax':
            min_val = x.min(axis=0)
            max_val = x.max(axis=0) + 1e-8
            return (x - min_val) / (max_val - min_val)
        else:
            return x


# Other datasets (can inherit from HCRLCHDataset)
class Set01Dataset(HCRLCHDataset):
    def __init__(self, data_path: str, split_ratio: tuple = (0.7, 0.15, 0.15), 
                 normalization: str = 'zscore', **kwargs):
        super().__init__(data_path, split_ratio, normalization, **kwargs)

class Set02Dataset(HCRLCHDataset):
    def __init__(self, data_path: str, split_ratio: tuple = (0.7, 0.15, 0.15), 
                 normalization: str = 'zscore', **kwargs):
        super().__init__(data_path, split_ratio, normalization, **kwargs)

class Set03Dataset(HCRLCHDataset):
    def __init__(self, data_path: str, split_ratio: tuple = (0.7, 0.15, 0.15), 
                 normalization: str = 'zscore', **kwargs):
        super().__init__(data_path, split_ratio, normalization, **kwargs)

class Set04Dataset(HCRLCHDataset):
    def __init__(self, data_path: str, split_ratio: tuple = (0.7, 0.15, 0.15), 
                 normalization: str = 'zscore', **kwargs):
        super().__init__(data_path, split_ratio, normalization, **kwargs)
```

---

## Template 2: Model Parameter Updates

**File:** `src/models/vgae.py`

**Change your `__init__` method to:**

```python
def __init__(self,
             input_dim: int = 128,
             hidden_dim: int = 64,
             latent_dim: int = 32,
             num_layers: int = 2,
             dropout: float = 0.2,
             **kwargs):  # ← IMPORTANT: catches extra config params
    """
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        num_layers: Number of layers
        dropout: Dropout rate
        **kwargs: Other arguments (ignored)
    """
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.latent_dim = latent_dim
    self.num_layers = num_layers
    
    # YOUR EXISTING MODEL ARCHITECTURE HERE
    # Just use self.hidden_dim instead of hardcoded values
```

**File:** `src/models/gat.py`

**Change your `__init__` method to:**

```python
def __init__(self,
             input_dim: int = 128,
             hidden_dim: int = 64,
             output_dim: int = 128,
             num_heads: int = 4,
             num_layers: int = 2,
             dropout: float = 0.2,
             **kwargs):
    """
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_heads: Number of attention heads
        num_layers: Number of GAT layers
        dropout: Dropout rate
        **kwargs: Other arguments (ignored)
    """
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.num_heads = num_heads
    self.num_layers = num_layers
    
    # YOUR EXISTING MODEL ARCHITECTURE HERE
```

**File:** `src/models/dqn.py`

**Change your `__init__` method to:**

```python
def __init__(self,
             input_dim: int = 128,
             hidden_dim: int = 256,
             action_dim: int = 4,
             num_layers: int = 3,
             dropout: float = 0.1,
             dueling: bool = False,
             **kwargs):
    """
    Args:
        input_dim: Input state dimension
        hidden_dim: Hidden layer dimension
        action_dim: Action space dimension
        num_layers: Number of layers
        dropout: Dropout rate
        dueling: Use dueling DQN architecture
        **kwargs: Other arguments (ignored)
    """
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.action_dim = action_dim
    self.num_layers = num_layers
    self.dueling = dueling
    
    # YOUR EXISTING MODEL ARCHITECTURE HERE
```

---

## Template 3: Data Loader Implementation

**File:** `src/training/train_with_hydra_zen.py`

**Replace the `load_data_loaders()` function:**

```python
from torch.utils.data import DataLoader
from src.data.datasets import (
    HCRLCHDataset, Set01Dataset, Set02Dataset, Set03Dataset, Set04Dataset
)

# Map dataset names to classes
DATASET_MAP = {
    'hcrlch': HCRLCHDataset,
    'set01': Set01Dataset,
    'set02': Set02Dataset,
    'set03': Set03Dataset,
    'set04': Set04Dataset,
}

def load_data_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load data loaders according to config"""
    
    # Get dataset class
    dataset_name = cfg.dataset_config.name
    dataset_class = DATASET_MAP.get(dataset_name)
    
    if dataset_class is None:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in DATASET_MAP. "
            f"Available: {list(DATASET_MAP.keys())}"
        )
    
    # Instantiate dataset
    dataset = dataset_class(
        data_path=cfg.dataset_config.data_path,
        split_ratio=cfg.dataset_config.split_ratio,
        normalization=cfg.dataset_config.normalization,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset.train,
        batch_size=cfg.training_config.batch_size,
        shuffle=True,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=cfg.get('pin_memory', False),
    )
    
    val_loader = DataLoader(
        dataset.val,
        batch_size=cfg.training_config.batch_size,
        shuffle=False,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=cfg.get('pin_memory', False),
    )
    
    test_loader = DataLoader(
        dataset.test,
        batch_size=cfg.training_config.batch_size,
        shuffle=False,
        num_workers=cfg.get('num_workers', 0),
        pin_memory=cfg.get('pin_memory', False),
    )
    
    return train_loader, val_loader, test_loader
```

---

## Template 4: Configuration Updates

**File:** `hydra_configs/config_store.py`

**Update the path settings at the top:**

```python
@dataclass
class ExperimentConfig:
    # ← UPDATE THESE PATHS ←
    project_root: str = "/home/username/KD-GAT"  # Change to your actual path!
    data_root: str = "${project_root}/data"
    experiment_root: str = "${project_root}/experimentruns"
    
    # ... rest of config
```

**Update dataset configs to match your data:**

```python
@dataclass
class HCRLCHDatasetConfig:
    _target_: str = "src.data.datasets.HCRLCHDataset"
    name: str = "hcrlch"
    data_path: str = "${data_root}/automotive/hcrlch"  # ← Verify this directory exists!
    split_ratio: tuple = (0.7, 0.15, 0.15)
    normalization: str = "zscore"
```

**Update model configs to match your models:**

```python
@dataclass
class VGAEModelConfig:
    _target_: str = "src.models.vgae.VGAE"  # ← Ensure class exists at this path
    input_dim: int = 128
    output_dim: int = 128
    latent_dim: int = 32

@dataclass
class GATModelConfig:
    _target_: str = "src.models.gat.GAT"  # ← Ensure class exists at this path
    input_dim: int = 128
    output_dim: int = 128
    num_heads: int = 4
```

---

## Template 5: Lightning Module Model Linking

**File:** `src/training/lightning_modules.py`

**Update the model instantiation methods:**

```python
def _build_vgae(self) -> nn.Module:
    """Build VGAE model from config"""
    from src.models.vgae import VGAE
    
    return VGAE(
        input_dim=self.cfg.model_config.input_dim,
        hidden_dim=self.cfg.model_size_config.hidden_dim,
        latent_dim=self.cfg.model_config.latent_dim,
        num_layers=self.cfg.model_size_config.num_layers,
        dropout=self.cfg.model_size_config.dropout,
    )

def _build_gat(self) -> nn.Module:
    """Build GAT model from config"""
    from src.models.gat import GAT
    
    return GAT(
        input_dim=self.cfg.model_config.input_dim,
        hidden_dim=self.cfg.model_size_config.hidden_dim,
        output_dim=self.cfg.model_config.output_dim,
        num_heads=self.cfg.model_config.num_heads,
        num_layers=self.cfg.model_size_config.num_layers,
        dropout=self.cfg.model_size_config.dropout,
    )

def _build_dqn(self) -> nn.Module:
    """Build DQN model from config"""
    from src.models.dqn import DQN
    
    return DQN(
        input_dim=self.cfg.model_config.input_dim,
        hidden_dim=self.cfg.model_size_config.hidden_dim,
        action_dim=self.cfg.model_config.action_dim,
        num_layers=self.cfg.model_size_config.num_layers,
        dropout=self.cfg.model_size_config.dropout,
        dueling=self.cfg.model_config.dueling,
    )
```

---

## Verification Checklist

After making changes, verify each step:

```bash
# 1. Test imports work
python3 -c "from src.data.datasets import HCRLCHDataset; print('✅ Dataset')"
python3 -c "from src.models.vgae import VGAE; print('✅ VGAE')"

# 2. Test models accept parameters
python3 << 'EOF'
from src.models.vgae import VGAE
model = VGAE(input_dim=128, hidden_dim=64, latent_dim=32, num_layers=2, dropout=0.1)
print(f"✅ VGAE: {sum(p.numel() for p in model.parameters())} params")
EOF

# 3. Test config loads
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --cfg job | head -20

# 4. Test data loader
python3 << 'EOF'
from src.data.datasets import HCRLCHDataset
from torch.utils.data import DataLoader
dataset = HCRLCHDataset('./data/automotive/hcrlch', (0.7, 0.15, 0.15), 'zscore')
loader = DataLoader(dataset.train, batch_size=32)
batch = next(iter(loader))
print(f"✅ DataLoader: batch keys = {list(batch.keys())}")
EOF

# 5. Test single epoch training
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    device=cpu \
    training_config.epochs=1
```

---

**These templates should be 95% of what you need. Customize as needed for your specific models!**
