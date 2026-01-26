# Knowledge Distillation Refactor Plan

## Problem Statement
Currently `knowledge_distillation` is treated as a training **mode**, but it should be a **toggle/modifier** that can be applied to ANY mode (autoencoder, curriculum, fusion, normal).

## Design Principle
**KD is orthogonal to training mode.** You should be able to:
- Train VGAE student with autoencoder mode + KD from VGAE teacher
- Train GAT student with curriculum mode + KD from GAT teacher
- Train either without KD (current teacher training)

---

## Phase 1: Config Layer Changes

### 1.1 Base TrainingConfig Updates
Add KD fields to the base config (not as a separate mode):

```python
@dataclass
class TrainingConfig:
    mode: str  # autoencoder, curriculum, fusion, normal

    # Knowledge Distillation (toggle, not a mode)
    use_knowledge_distillation: bool = False
    teacher_model_path: Optional[str] = None  # Path to teacher .pth file
    distillation_temperature: float = 4.0     # Softmax temperature for soft labels
    distillation_alpha: float = 0.7           # Weight: alpha*KD_loss + (1-alpha)*task_loss

    # ... rest of existing fields
```

### 1.2 Remove KnowledgeDistillationConfig
- Delete or deprecate `KnowledgeDistillationConfig` class
- Remove `"knowledge_distillation"` from mode choices in CLI
- Update mode dispatch logic

### 1.3 Path Generation
Current path structure already supports this:
```
{modality}/{dataset}/{learning_type}/{model}/{model_size}/{distillation}/{mode}
                                                          ↑
                                              "with-kd" or "no-kd"
```

Example paths:
- Teacher: `automotive/hcrl_sa/unsupervised/vgae/teacher/no-kd/autoencoder`
- Student+KD: `automotive/hcrl_sa/unsupervised/vgae/student/with-kd/autoencoder`

---

## Phase 2: Lightning Module Changes

### 2.1 Add KD Mixin or Base Logic
Create a utility for loading teacher and computing KD loss:

```python
# src/training/knowledge_distillation.py

class KDHelper:
    """Helper for knowledge distillation in any lightning module."""

    def __init__(self, cfg, student_model, model_type: str = "vgae"):
        self.enabled = cfg.training.use_knowledge_distillation
        self.temperature = cfg.training.distillation_temperature
        self.alpha = cfg.training.distillation_alpha
        self.model_type = model_type  # "vgae" or "gat"
        self.teacher = None
        self.projection_layer = None
        self.device = next(student_model.parameters()).device

        if self.enabled:
            self.teacher = self._load_teacher(cfg.training.teacher_model_path)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

            # Setup projection layer for dimension mismatch
            self.projection_layer = self._setup_projection_layer(student_model)

    def _load_teacher(self, path):
        # Load teacher model matching student architecture but larger
        checkpoint = torch.load(path, map_location=self.device)
        # ... model construction based on checkpoint config
        return teacher_model

    def _setup_projection_layer(self, student_model) -> Optional[nn.Module]:
        """Setup projection layer if teacher and student have different latent dimensions."""
        teacher_latent_dim = getattr(self.teacher, 'latent_dim', 32)
        student_latent_dim = getattr(student_model, 'latent_dim', 16)

        if teacher_latent_dim != student_latent_dim:
            projection_layer = nn.Linear(student_latent_dim, teacher_latent_dim).to(self.device)
            print(f"✓ Projection layer added: {student_latent_dim} → {teacher_latent_dim}")
            return projection_layer
        return None

    @torch.no_grad()
    def get_teacher_outputs(self, batch):
        """Get teacher outputs for KD. Returns dict with model-specific outputs."""
        if not self.enabled:
            return None

        if self.model_type == "vgae":
            # VGAE returns: cont_out, canid_logits, neighbor_logits, z, kl_loss
            cont_out, canid_logits, neighbor_logits, z, _ = self.teacher(batch)
            return {
                'z': z,
                'cont_out': cont_out,
                'canid_logits': canid_logits,
                'neighbor_logits': neighbor_logits
            }
        else:  # GAT
            logits = self.teacher(batch)
            return {'logits': logits}

    def compute_vgae_kd_loss(self, student_z, student_recon, teacher_outputs):
        """
        VGAE-specific KD loss: distill BOTH latent space AND reconstruction.

        Args:
            student_z: Student latent vectors
            student_recon: Dict with student reconstruction outputs
            teacher_outputs: Dict from get_teacher_outputs()

        Returns:
            Combined KD loss for latent + reconstruction
        """
        if not self.enabled:
            return torch.tensor(0.0, device=self.device)

        # 1. Latent space distillation (with projection if needed)
        if self.projection_layer is not None:
            projected_student_z = self.projection_layer(student_z)
        else:
            projected_student_z = student_z

        latent_loss = F.mse_loss(projected_student_z, teacher_outputs['z'])

        # 2. Reconstruction distillation
        recon_loss = F.mse_loss(student_recon['cont_out'], teacher_outputs['cont_out'])

        # Combine with equal weighting (can be tuned)
        kd_loss = 0.5 * latent_loss + 0.5 * recon_loss

        return kd_loss

    def compute_gat_kd_loss(self, student_logits, teacher_logits):
        """
        GAT-specific KD loss: standard soft label distillation.

        Args:
            student_logits: Student classification logits
            teacher_logits: Teacher classification logits

        Returns:
            KL divergence loss with temperature scaling
        """
        if not self.enabled:
            return torch.tensor(0.0, device=self.device)

        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)

        return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)

    def combine_losses(self, task_loss, kd_loss):
        """Weighted combination of task loss and KD loss."""
        if not self.enabled:
            return task_loss
        return self.alpha * kd_loss + (1 - self.alpha) * task_loss
```

### 2.2 Update VAELightningModule
```python
class VAELightningModule(pl.LightningModule):
    def __init__(self, cfg, num_ids):
        super().__init__()
        self.model = self._build_vgae()

        # Initialize KD helper (no-op if disabled)
        self.kd_helper = KDHelper(cfg, self.model, model_type="vgae")

        # Optimizer for projection layer if KD enabled
        if self.kd_helper.projection_layer is not None:
            self.proj_optimizer = torch.optim.Adam(
                self.kd_helper.projection_layer.parameters(), lr=1e-3
            )

    def training_step(self, batch, batch_idx):
        # Student forward pass
        cont_out, canid_logits, neighbor_logits, z, kl_loss = self.forward(batch)

        # Compute task loss (reconstruction + KL)
        task_loss = self._compute_reconstruction_loss(
            batch, cont_out, canid_logits, neighbor_logits, kl_loss
        )

        # Get teacher outputs if KD enabled (dual-signal: latent + reconstruction)
        if self.kd_helper.enabled:
            teacher_outputs = self.kd_helper.get_teacher_outputs(batch)

            # Package student reconstruction outputs
            student_recon = {
                'cont_out': cont_out,
                'canid_logits': canid_logits,
                'neighbor_logits': neighbor_logits
            }

            # Compute VGAE-specific KD loss (latent + reconstruction)
            kd_loss = self.kd_helper.compute_vgae_kd_loss(
                student_z=z,
                student_recon=student_recon,
                teacher_outputs=teacher_outputs
            )

            loss = self.kd_helper.combine_losses(task_loss, kd_loss)

            self.log('kd_loss', kd_loss)
            self.log('kd_latent_loss', ...)  # Can track components separately
            self.log('kd_recon_loss', ...)
        else:
            loss = task_loss

        self.log('train_loss', loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Memory cleanup every N batches."""
        if batch_idx % 20 == 0:
            cleanup_memory()  # gc.collect() + torch.cuda.empty_cache()
```

### 2.3 Update GATLightningModule
Similar pattern - add KD helper and modify training_step.

---

## Phase 3: Mode-Specific KD Support Matrix

| Mode | KD Support | Notes |
|------|------------|-------|
| autoencoder | ✅ YES | VGAE dual-signal (latent + reconstruction) |
| curriculum | ✅ YES | GAT soft label distillation |
| fusion | ❌ NO | DQN uses already-distilled models |
| normal | ✅ YES | Standard soft label KD |

### 3.1 Autoencoder Mode + KD ✅
- Teacher: VGAE teacher trained on normal samples
- Student learns: reconstruction + soft latent representations from teacher
- **KD signals (BOTH)**:
  - Latent space: MSE between projected student `z` and teacher `z_teacher`
  - Reconstruction: MSE between student and teacher reconstructed features
- Projection layer handles dimension mismatch (student_dim → teacher_dim)

### 3.2 Curriculum Mode + KD ✅
- Teacher: GAT teacher trained with curriculum
- Student learns: classification + soft labels from teacher
- KD signal: KL divergence with temperature-scaled softmax logits

### 3.3 Fusion Mode + KD ❌
- **NOT SUPPORTED** - explicitly excluded
- The DQN agent trains on outputs from already-distilled VGAE and GAT
- No additional KD needed at fusion level
- Validation should reject `--mode fusion --distillation with-kd`

### 3.4 Normal Mode + KD ✅
- Standard soft label KD with temperature scaling
- Most straightforward implementation

---

## Phase 4: CLI and Validation

### 4.1 CLI Changes
```python
# --mode choices: remove 'distillation', keep others
core.add_argument(
    '--mode',
    choices=['normal', 'autoencoder', 'curriculum', 'fusion'],  # No 'distillation'
    help='Training strategy/mode'
)

# --distillation already exists and is correct
core.add_argument(
    '--distillation',
    choices=['with-kd', 'no-kd'],
    default='no-kd',
    help='Knowledge distillation toggle'
)
```

### 4.2 Config Validation
```python
def validate_config(config):
    if config.training.use_knowledge_distillation:
        # REJECT: Fusion mode with KD is not supported
        if config.training.mode == "fusion":
            raise ValueError(
                "Knowledge distillation is not supported for fusion mode. "
                "The DQN agent uses already-distilled VGAE and GAT models."
            )

        # Must have teacher path
        if not config.training.teacher_model_path:
            raise ValueError("KD enabled but no teacher_model_path specified")

        # Teacher must exist
        if not Path(config.training.teacher_model_path).exists():
            raise FileNotFoundError(f"Teacher model not found: {config.training.teacher_model_path}")

        # Model size should be student
        if config.model_size != "student":
            logger.warning("KD typically used with student models, but teacher size specified")
```

### 4.3 Auto-Discovery of Teacher Path
When `--distillation with-kd` is set but no explicit teacher path:
```python
def infer_teacher_path(config):
    """Auto-discover teacher model path based on config."""
    base = Path(config.experiment_root) / config.modality / config.dataset.name

    if config.model.type in ['vgae']:
        return base / "unsupervised/vgae/teacher/no-kd/autoencoder/models/vgae_autoencoder.pth"
    elif config.model.type in ['gat', 'gat_student']:
        return base / "supervised/gat/teacher/no-kd/curriculum/models/gat_curriculum.pth"
    # etc.
```

---

## Implementation Order

1. **Create KDHelper class** (`src/training/knowledge_distillation.py`)
2. **Update base TrainingConfig** with KD fields
3. **Update VAELightningModule** to use KDHelper
4. **Update GATLightningModule** to use KDHelper
5. **Update CLI** to map `--distillation with-kd` to config fields
6. **Update validation** to check KD requirements
7. **Test with hcrl_sa**: VGAE student + autoencoder + KD
8. **Test with hcrl_sa**: GAT student + curriculum + KD

---

## Backwards Compatibility

- Existing teacher training (`--distillation no-kd`) works unchanged
- Old `knowledge_distillation` mode can be deprecated with warning
- Paths remain compatible (already have distillation component)

---

## Resolved Design Decisions

### 1. VGAE KD Signal: Distill BOTH
Based on previous implementation experience, VGAE knowledge distillation should distill **both**:
- **Latent vectors (`z`)**: Student learns teacher's latent space representation
- **Reconstructed features**: Student matches teacher's reconstruction quality

This dual-signal approach provides richer knowledge transfer than either alone.

### 2. Dimension Mismatch: Projection Layer
Teacher (larger) → Student (smaller) dimension mismatch is handled via **projection layer**:
```python
# Student latent dim < Teacher latent dim
projection_layer = nn.Linear(student_latent_dim, teacher_latent_dim).to(device)
# Project student embeddings UP to teacher dimension for comparison
projected_student_z = projection_layer(student_z)
kd_loss = F.mse_loss(projected_student_z, teacher_z)
```

### 3. Fusion KD: NOT NEEDED
**Decision: Skip KD for fusion mode entirely.**
- Fusion mode trains the DQN agent that combines VGAE and GAT outputs
- The underlying VGAE and GAT models will already be distilled separately
- Adding KD at the fusion/DQN level adds complexity without clear benefit
- Focus on autoencoder + KD and curriculum + KD only

---

## OOM Mitigation for KD Training

KD training requires extra memory due to:
- Teacher model loaded in memory (frozen, but still takes space)
- Intermediate tensors from teacher forward pass
- Projection layer gradients (if dimension mismatch)

### Safety Factor Adjustments
Update `config/batch_size_factors.json` with KD-specific factors:
```json
{
  "_comment_kd": "KD training needs ~25% more memory for teacher + projection",

  "hcrl_ch_kd": 0.45,     // 0.6 * 0.75
  "hcrl_sa_kd": 0.41,     // 0.55 * 0.75
  "set_01_kd": 0.41,      // 0.55 * 0.75
  "set_02_kd": 0.26,      // 0.35 * 0.75
  "set_03_kd": 0.26,
  "set_04_kd": 0.26,

  "_default_kd": 0.38     // 0.5 * 0.75
}
```

### Memory Optimization Strategies
1. **AMP (Automatic Mixed Precision)**: Already in VGAE_Distillation.md pattern
   ```python
   scaler = torch.cuda.amp.GradScaler() if is_cuda else None
   with torch.amp.autocast('cuda', dtype=torch.float16):
       total_loss = compute_kd_loss(...)
   ```

2. **Teacher in eval mode with no_grad**: Prevents gradient graph for teacher
   ```python
   self.teacher.eval()
   for p in self.teacher.parameters():
       p.requires_grad = False

   @torch.no_grad()
   def get_teacher_outputs(self, batch):
       return self.teacher(batch)
   ```

3. **Aggressive memory cleanup**: Every N batches
   ```python
   if batch_idx % 20 == 0:
       cleanup_memory()  # gc.collect() + torch.cuda.empty_cache()
   ```

4. **Gradient checkpointing**: Already implemented for VGAE and GAT

---

## Updated Mode-Specific Considerations

### Autoencoder Mode + KD (SUPPORTED)
- Teacher: VGAE teacher trained on normal samples
- Student learns: reconstruction + soft latent representations from teacher
- **KD signals (both)**:
  - Latent space: MSE loss between projected student `z` and teacher `z`
  - Reconstruction: MSE loss between student and teacher reconstructions
- Projection layer handles dimension mismatch

### Curriculum Mode + KD (SUPPORTED)
- Teacher: GAT teacher trained with curriculum
- Student learns: classification + soft labels from teacher
- KD signal: KL divergence between softmax logits (standard KD)
- Temperature scaling for soft labels

### Fusion Mode + KD (NOT SUPPORTED)
- **Explicitly excluded** - no KD at fusion level
- The DQN agent trains on already-distilled VGAE and GAT outputs
- This simplifies implementation and avoids complex multi-model KD

### Normal Mode + KD (SUPPORTED)
- Standard KD with soft labels
- Most straightforward implementation
