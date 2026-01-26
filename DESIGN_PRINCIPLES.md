
================================================================================
STRUCTURED IMPLEMENTATION PLAN
================================================================================

## PRINCIPLE 1: Folder Structure → Explicit CLI (P→Q Logic)
## Directory Structure
```
{experiment_root}/{modality}/{dataset}/{model_size}/{learning_type}/{model_arch}/{kd_enabled}/{training_strategy}/
```

## Levels (in order)
**modality** --modality
    - automotive
**dataset** - CLI: --dataset
   - hcrl_ch, hcrl_sa, set_01, set_02, set_03, set_04

**model_size** - CLI: --model-size
   - teacher, student

**learning_type** --learning-type
   - supervised, unsupervised, rl_fusion

**distillation** - --distillation
   - with-kd, no-kd
**model_arch** - CLI: --model
   - vgae, gat, dqn

**training_mode** - CLI: --mode
   - normal, curriculum, autoencoder, fusion, distillation
## PRINCIPLE 2: Parameter Bible (Reference Database)

Create central parameter schema:

parameters/
├── required_cli.yaml       # MUST be explicit in CLI (from folder structure)
├── supplementary_cli.yaml  # CAN override, have defaults
└── schema_validation.py    # Verify consistency

required_cli.yaml:
```yaml
modality:
  type: string
  choices: [automotive, industrial, robotics]
  description: "Application domain"

dataset:
  type: string
  choices: [hcrl_ch, hcrl_sa, set_01, set_02, set_03, set_04]
  description: "Dataset name"
  #TODO: Let's add the new data folder path data/automotive
  # Let's also add a nice note that when we increase our modality, we will need to move the path up to just data and then do a modality to path lookup 

learning_type:
  type: string
  choices: [supervised, unsupervised, semi_supervised, rl_fusion]
  description: "ML learning paradigm"

model:
  type: string
  choices: [vgae, gat, dqn, gcn, gnn]
  description: "Model architecture"

model_size:
  type: string
  choices: [teacher, student]
  description: "Model capacity"

distillation:
  type: string
  choices: [with-kd, no-kd]
  description: "Knowledge distillation enabled"

mode:
  type: string
  choices: [normal, curriculum, autoencoder, fusion, distillation, evaluation]
  description: "Training strategy/mode"
  details:
    normal: "Standard supervised training - ALL samples (normal + attacks)"
    curriculum: "Supervised with class imbalance handling - ALL samples, progressive difficulty (1:1 → 10:1 ratio)"
    autoencoder: "Unsupervised reconstruction - NORMAL SAMPLES ONLY"
    fusion: "RL-based fusion of pretrained models - requires both autoencoder AND classifier"
    distillation: "Knowledge distillation from teacher to student - requires teacher model"
    evaluation: "Comprehensive evaluation pipeline - requires trained models"
```

supplementary_cli.yaml:
```yaml
learning_rate:
  type: float
  default: 0.001
  range: [1e-5, 1e-1]

batch_size:
  type: int
  default: adaptive
  range: [8, 512]

epochs:
  type: int
  default: 100
  range: [1, 10000]

dropout:
  type: float
  default: 0.2
  range: [0.0, 0.9]

num_layers:
  type: int
  default: 3
  range: [1, 10]

heads:  # GAT-specific
  type: int
  default: 4
  range: [1, 16]

distillation_temperature:  # KD-specific
  type: float
  default: 1.0
  range: [0.1, 10.0]

distillation_alpha:  # KD-specific
  type: float
  default: 0.5
  range: [0.0, 1.0]
```

## PRINCIPLE 3: Supplementary Args + Naming Consistency

Consistent naming everywhere
After making a big change or adding a feature, we will just check to make sure the naming convention is the same throughout.

