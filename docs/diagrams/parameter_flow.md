# Parameter Flow: CLI → SLURM → Training

This diagram shows how configuration parameters flow through the KD-GAT system from CLI submission to training execution.

---

## Complete Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as can-train CLI
    participant Pydantic as Pydantic Validators
    participant Builder as Config Builder
    participant JM as Job Manager
    participant Freeze as Frozen Config
    participant SLURM
    participant Train as train_with_hydra_zen.py
    participant Config as CANGraphConfig

    User->>CLI: ./can-train pipeline --model gat --dataset hcrl_sa --submit
    CLI->>Pydantic: Validate CLI arguments
    Pydantic-->>CLI: Validated input

    CLI->>Builder: build_config_from_buckets(run_type, model_args, slurm_args)
    Builder->>Builder: Parse comma-separated values
    Builder->>Builder: Create config dataclasses
    Builder-->>CLI: CANGraphConfig instances

    CLI->>JM: submit_pipeline(configs, slurm_args)

    loop For each stage
        JM->>Freeze: save_frozen_config(config, timestamp)
        Freeze->>Freeze: Serialize dataclasses to JSON
        Freeze-->>JM: frozen_config_20260126_220158.json

        JM->>JM: Generate SLURM script
        Note over JM: Includes --frozen-config path<br/>and dependency on previous job

        JM->>SLURM: sbatch job.sh
        SLURM-->>JM: job_id (e.g., 12345678)
    end

    SLURM->>Train: Execute on compute node
    Train->>Freeze: load_frozen_config(path)
    Freeze->>Freeze: Deserialize JSON to dataclasses
    Freeze-->>Train: Reconstructed CANGraphConfig

    Train->>Config: Validate config
    Config-->>Train: Validated config

    Train->>Train: HydraZenTrainer.train()
    Train->>Train: Save models, checkpoints, logs
```

---

## Frozen Config Pattern

### Why Frozen Configs?

**Problem**: Traditional config systems are fragile:
- CLI flags can change between submission and execution
- Hydra configs require exact Python environment
- Hard to reproduce old runs after code changes

**Solution**: Freeze entire config to JSON at submission time:

```mermaid
flowchart LR
    subgraph Submission["Submission Time (Login Node)"]
        CLI[CLI Args] --> Build[Build Config]
        Build --> Freeze[Freeze to JSON]
        Freeze --> File[frozen_config_20260126_220158.json]
    end

    subgraph Execution["Execution Time (Compute Node)"]
        File --> Load[Load from JSON]
        Load --> Reconstruct[Reconstruct Dataclasses]
        Reconstruct --> Training[Start Training]
    end

    style Submission fill:#e3f2fd
    style Execution fill:#e8f5e9
```

**Benefits**:
- ✅ Reproducible: Config is self-contained
- ✅ Auditable: Exact params saved per run
- ✅ Robust: Works even if CLI changes

---

## Config Validation Layers

```mermaid
flowchart TD
    Input[User CLI Input] --> Layer1[Layer 1: Pydantic]
    Layer1 --> |validated| Layer2[Layer 2: Config Builder]
    Layer2 --> |constructed| Layer3[Layer 3: HydraZen Dataclasses]
    Layer3 --> |frozen| Layer4[Layer 4: Validator]
    Layer4 --> |approved| SLURM[SLURM Submission]

    Layer1 -.->|reject| Error1[CLI Error:<br/>Invalid argument]
    Layer2 -.->|reject| Error2[Builder Error:<br/>Incompatible combo]
    Layer4 -.->|reject| Error4[Validation Error:<br/>Missing artifact]

    style Layer1 fill:#f3e5f5
    style Layer2 fill:#e1f5fe
    style Layer3 fill:#e8f5e9
    style Layer4 fill:#fff9c4
    style Error1 fill:#ffcdd2
    style Error2 fill:#ffcdd2
    style Error4 fill:#ffcdd2
```

**Layer 1: Pydantic** ([pydantic_validators.py](../../src/cli/pydantic_validators.py))
- Validates CLI input types
- Enforces P→Q rules (e.g., model size must match distillation)
- User-facing error messages

**Layer 2: Config Builder** ([config_builder.py](../../src/cli/config_builder.py))
- Parses comma-separated buckets
- Maps CLI params → config fields
- Creates HydraZen dataclass instances

**Layer 3: HydraZen** ([hydra_zen_configs.py](../../src/config/hydra_zen_configs.py))
- Dataclass schema definitions
- Default values
- Type hints

**Layer 4: Validator** ([validator.py](../../src/cli/validator.py))
- Pre-flight checks (dataset exists, teacher model found, etc.)
- SLURM resource validation
- Mode-specific checks (KD requirements, fusion artifacts)

---

## Example: Pipeline Submission

### User Command
```bash
./can-train pipeline \
  --modality automotive \
  --model vgae,gat,dqn \
  --learning-type unsupervised,supervised,rl_fusion \
  --training-strategy autoencoder,curriculum,fusion \
  --dataset hcrl_sa \
  --model-size teacher \
  --distillation no-kd \
  --epochs 5 \
  --submit
```

### Step 1: CLI Parsing
```python
args.modality = 'automotive'
args.model = 'vgae,gat,dqn'
args.learning_type = 'unsupervised,supervised,rl_fusion'
args.training_strategy = 'autoencoder,curriculum,fusion'
args.dataset = 'hcrl_sa'
args.model_size = 'teacher'
args.distillation = 'no-kd'
args.epochs = 5
args.submit = True
```

### Step 2: Build Configs (3 stages)
```python
# Job 1: VGAE Autoencoder
run_type = {
    'model': 'vgae',
    'dataset': 'hcrl_sa',
    'mode': 'autoencoder',
    'modality': 'automotive',
    'model_size': 'teacher',
    'distillation': 'no-kd',
    'learning_type': 'unsupervised'
}
model_args = {'epochs': 5}
slurm_args = {'gpus': 1, 'walltime': '06:00:00', ...}

# Job 2: GAT Curriculum (similar structure)
# Job 3: DQN Fusion (similar structure)
```

### Step 3: Freeze Configs
```json
// experimentruns/automotive/hcrl_sa/unsupervised/vgae/teacher/no_distillation/autoencoder/configs/frozen_config_20260126_220158.json
{
  "_type": "CANGraphConfig",
  "_frozen_at": "2026-01-26T22:01:58.375388",
  "_version": "1.0.0",
  "modality": "automotive",
  "model_size": "teacher",
  "distillation": "no_distillation",
  "experiment_root": "experimentruns",
  "model": {
    "_type": "VGAEConfig",
    "num_ids": 2049,
    "embedding_dim": 64,
    "hidden_dims": [1024, 512],
    "latent_dim": 96,
    ...
  },
  "training": {
    "_type": "AutoencoderTrainingConfig",
    "mode": "autoencoder",
    "max_epochs": 5,
    ...
  },
  ...
}
```

### Step 4: Generate SLURM Script
```bash
#!/bin/bash
#SBATCH --job-name=vgae_hcrl_sa_autoencoder
#SBATCH --account=PAS2022
#SBATCH --time=06:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

python train_with_hydra_zen.py \
  --frozen-config experimentruns/.../frozen_config_20260126_220158.json

echo "Exit code: $?"
echo "End: $(date)"
```

### Step 5: Submit to SLURM
```bash
$ sbatch experimentruns/.../slurm_logs/vgae_hcrl_sa_autoencoder.sh
Submitted batch job 12345678
```

### Step 6: Execution on Compute Node
```python
# train_with_hydra_zen.py
config = load_frozen_config(args.frozen_config)  # Deserialize JSON → dataclasses
trainer = HydraZenTrainer(config)
model, lightning_trainer = trainer.train()
```

---

## SLURM Job Dependencies

When pipeline stages depend on each other, SLURM ensures sequential execution:

```mermaid
flowchart TD
    Submit[User: ./can-train pipeline --submit]

    Submit --> Job1[Job 1: VGAE<br/>sbatch vgae.sh<br/>Job ID: 12345678]
    Submit --> Job2[Job 2: GAT<br/>sbatch --dependency=afterok:12345678 gat.sh<br/>Job ID: 12345679]
    Submit --> Job3[Job 3: DQN<br/>sbatch --dependency=afterok:12345679 dqn.sh<br/>Job ID: 12345680]

    Job1 --> |completes successfully| Job2_Start[Job 2 starts]
    Job2_Start --> Job2_Run[Job 2 runs]
    Job2_Run --> |completes successfully| Job3_Start[Job 3 starts]

    Job1 -.->|fails| Job2_Cancel[Job 2 cancelled]
    Job2_Run -.->|fails| Job3_Cancel[Job 3 cancelled]

    style Job1 fill:#bbdefb
    style Job2 fill:#c8e6c9
    style Job3 fill:#ffe0b2
    style Job2_Cancel fill:#ffcdd2
    style Job3_Cancel fill:#ffcdd2
```

**Key Point**: If any job fails, downstream jobs are automatically cancelled by SLURM.

---

## Model Args Flow

Model args override default config values:

```mermaid
flowchart LR
    CLI[CLI: --epochs 5] --> Parse[Parse to model_args]
    Parse --> Map[Map 'epochs' → 'max_epochs']
    Map --> Override[Override default in AutoencoderTrainingConfig]
    Override --> Freeze[Freeze to JSON]
    Freeze --> Execute[Execute with max_epochs=5]

    Default[Default: max_epochs=400] -.->|overridden by| Override

    style Default fill:#ffcdd2
    style Override fill:#c8e6c9
```

**Mapping** ([config_builder.py:393](../../src/cli/config_builder.py#L393)):
```python
arg_mapping = {
    'epochs': 'max_epochs',
    'learning_rate': 'learning_rate',
    'batch_size': 'batch_size',
    ...
}
```

---

## References

- **Frozen Config Implementation**: [src/config/frozen_config.py](../../src/config/frozen_config.py)
- **CLI Entry Point**: [src/cli/main.py](../../src/cli/main.py)
- **Job Manager**: [src/cli/job_manager.py](../../src/cli/job_manager.py)
- **Config Builder**: [src/cli/config_builder.py](../../src/cli/config_builder.py)
- **Training Script**: [train_with_hydra_zen.py](../../train_with_hydra_zen.py)
