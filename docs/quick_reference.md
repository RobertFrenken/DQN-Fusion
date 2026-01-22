# KD-GAT Hydra-Zen Quick Reference

## Core Concepts

### 1. Configuration Hierarchy (8 Levels)

```
Modality → Dataset → Learning Type → Model Arch → Model Size → Distillation → Training Mode
   ↓          ↓           ↓            ↓            ↓              ↓            ↓
 automotive  hcrlch    unsupervised   VGAE       student         no       all_samples
              set01     classifier    GAT        teacher       standard   normals_only
              set02     fusion        DQN        intermediate  topology   curriculum_*
              set03                   
              set04
```

### 2. Config Name Format

```
{modality}_{dataset}_{learning_type}_{model_arch}_{model_size}_{distillation}_{training_mode}

automotive_hcrlch_unsupervised_vgae_student_no_all_samples
```

### 3. Path Structure

```
experimentruns/automotive/hcrlch/unsupervised/VGAE/student/no/all_samples/run_000/
│
├── model.pt                         ← Trained model weights
├── config.yaml                      ← Exact config used (reproducibility)
├── checkpoints/                     ← Training checkpoints
│   ├── best_model_epoch_05_0.234.ckpt
│   └── ...
├── logs/                            ← MLflow logs
├── training_metrics.json            ← Loss curves, training metrics
├── validation_metrics.json          ← Validation metrics
└── evaluation/                      ← Test set results
    ├── test_results.json
    ├── test_set/
    ├── known_unknowns/
    └── unknown_unknowns/
```

## Command Reference

### Run Experiments Locally

```bash
# Single experiment
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples

# Hyperparameter sweep (runs 3 experiments)
python src/training/train_with_hydra_zen.py -m \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    model_size_config.hidden_dim=64,128,256

# Multiple hyperparameters (Cartesian product)
python src/training/train_with_hydra_zen.py -m \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    model_size_config.hidden_dim=64,128 \
    training_config.learning_rate=1e-3,1e-4
# This runs 2 × 2 = 4 experiments

# Show loaded config without running
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --cfg job

# Run with custom output directory
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    hydra.run.dir=/custom/output/path
```

### Submit to OSC Slurm

```bash
# Single experiment
python oscjobmanager.py submit automotive_hcrlch_unsupervised_vgae_student_no_all_samples

# Dry run (preview script)
python oscjobmanager.py submit automotive_hcrlch_unsupervised_vgae_student_no_all_samples --dry-run

# With custom resources
python oscjobmanager.py submit \
    automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --walltime 04:00:00 \
    --memory 64G

# Sweep experiments
python oscjobmanager.py sweep \
    --modality automotive \
    --dataset hcrlch \
    --learning-type unsupervised \
    --model-architecture VGAE \
    --model-sizes student,teacher \
    --distillations no,standard \
    --training-modes all_samples,normals_only

# Sweep with dry-run
python oscjobmanager.py sweep --dry-run --model-sizes student,teacher
```

## Model Architecture + Learning Type Valid Combinations

```python
# Unsupervised (AutoEncoder-based)
VGAE  ← Only option

# Classification (Supervised)
GAT, DQN

# Fusion (Multi-modal)
GAT, DQN
```

## Model Size Options

```python
"teacher"        # Larger capacity (hidden_dim=256, layers=3)
"student"        # Compressed (hidden_dim=64, layers=2)
"intermediate"   # Middle (hidden_dim=128, layers=2)
"huge"           # Maximum (hidden_dim=512, layers=4)
"tiny"           # Minimal (hidden_dim=32, layers=1)
```

## Distillation Options

```python
"no"                       # Standard training
"standard"                 # KD with temperature scaling
"topology_preserving"      # KD with graph topology loss
```

## Training Mode Options

```python
# Works with all learning types:
"all_samples"              # Train on all data

# Additional options per learning type:
"normals_only"             # Train only on benign/normal samples
"curriculum_classifier"    # Curriculum for classification
"curriculum_fusion"        # Curriculum for fusion models
```

## Key File Locations

```
KD-GAT/
├── hydra_configs/
│   └── config_store.py              ← All configuration definitions
├── src/
│   ├── training/
│   │   ├── train_with_hydra_zen.py  ← Main training entry point
│   │   └── lightning_modules.py     ← Lightning modules
│   ├── utils/
│   │   └── experiment_paths.py      ← Path management (NO fallbacks)
│   ├── models/
│   │   ├── vgae.py                  ← VGAE implementation
│   │   ├── gat.py                   ← GAT implementation
│   │   └── dqn.py                   ← DQN implementation
│   └── data/
│       └── datasets.py              ← Dataset classes
├── oscjobmanager.py                 ← Slurm submission manager
├── IMPLEMENTATION_GUIDE.md          ← Detailed guide
└── QUICK_REFERENCE.md               ← This file
```

## Common Patterns

### Running All VGAE Variants on Dataset
```bash
python src/training/train_with_hydra_zen.py -m \
    config_store=automotive_hcrlch_unsupervised_vgae_*_*_* \
    --config-search-path .
```

### Comparing Model Sizes
```bash
python src/training/train_with_hydra_zen.py -m \
    learning_type=unsupervised \
    model_architecture=VGAE \
    model_size=student,intermediate,teacher \
    distillation=no
```

### Testing Distillation Impact
```bash
python oscjobmanager.py sweep \
    --learning-type classifier \
    --model-architecture GAT \
    --model-sizes student \
    --distillations no,standard,topology_preserving \
    --training-modes all_samples
```

## Important Notes

### ✅ DO
- Use pre-generated config names: `automotive_hcrlch_unsupervised_vgae_student_no_all_samples`
- Check experiment structure before training
- Save models to indicated paths (code does this automatically)
- Review config before training: `--cfg job`
- Use `--dry-run` before actual submission

### ❌ DON'T
- Hardcode paths - always use config
- Use `pickle` for models - use `torch.save()`
- Ignore path creation errors - they're informative
- Modify running configs - regenerate configs in config_store.py
- Use fallback paths - paths must be exact or fail

## Debugging

### Check if config exists
```bash
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    --cfg job | head -30
```

### List what got saved
```bash
ls -la experimentruns/automotive/hcrlch/unsupervised/VGAE/student/no/all_samples/run_000/
```

### View training metrics
```bash
cat experimentruns/automotive/hcrlch/unsupervised/VGAE/student/no/all_samples/run_000/training_metrics.json
```

### View exact config used
```bash
cat experimentruns/automotive/hcrlch/unsupervised/VGAE/student/no/all_samples/run_000/config.yaml
```

### Check MLflow runs
```bash
mlflow ui --backend-store-uri experimentruns/.mlruns
```

## Performance Tips

### Memory issues?
```bash
python oscjobmanager.py submit config_name \
    --memory 64G \
    --walltime 04:00:00
```

### Slow training?
- Increase `batch_size` in config
- Increase `cpus_per_task` for data loading
- Use `num_workers > 0` for parallel loading

### Need more runs?
```bash
python oscjobmanager.py sweep --model-sizes student,teacher,intermediate,huge --training-modes all_samples,normals_only --dry-run > jobs.txt
# Review jobs.txt before submitting
```

## File Organization After Training

```
experimentruns/automotive/hcrlch/unsupervised/VGAE/student/no/all_samples/
├── run_000/
│   ├── model.pt                      ← Load with torch.load()
│   ├── config.yaml                   ← Exact reproducible config
│   ├── checkpoints/                  ← Backup checkpoints
│   ├── logs/                         ← MLflow logs
│   ├── training_metrics.json         ← Loss over epochs
│   ├── validation_metrics.json       ← Val metrics
│   └── evaluation/
│       ├── test_results.json
│       ├── test_set/                 ← Results on test set
│       ├── known_unknowns/           ← Known attacks
│       └── unknown_unknowns/         ← Unknown attacks
├── run_001/
│   ├── model.pt
│   └── ...
└── slurm_runs/
    ├── config_name.sh                ← Slurm submission script
    ├── config_name.log               ← Job stdout
    └── config_name.err               ← Job stderr
```

## Config Override Examples

```bash
# Change learning rate only
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    training_config.learning_rate=5e-4

# Change multiple settings
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    training_config.learning_rate=5e-4 \
    training_config.batch_size=32 \
    training_config.epochs=200

# Override to use CPU (for testing)
python src/training/train_with_hydra_zen.py \
    config_store=automotive_hcrlch_unsupervised_vgae_student_no_all_samples \
    device=cpu
```

## Expected Training Output

```
========================================
🚀 Starting KD-GAT Training Pipeline
========================================
✅ Saved config to experimentruns/.../run_000/config.yaml

📊 Loading dataset...
✅ Loaded hcrlch dataset

🏗️  Building model...
✅ Created VGAE Lightning module
   Model size: student
   Distillation: no

⚙️  Configuring trainer...
✅ Trainer configured

🎯 Starting training...
   Epochs: 100
   Learning rate: 0.001
   Batch size: 64

Epoch 1/100: [████████░░] 80% | loss: 0.234

...training proceeds...

✅ Training completed successfully

📈 Running test evaluation...
✅ Test evaluation complete
   Results saved to experimentruns/.../evaluation/test_results.json

💾 Saving final model...
✅ Model saved to experimentruns/.../model.pt

======================================
✅ EXPERIMENT COMPLETED SUCCESSFULLY
======================================
Model path: experimentruns/.../model.pt
Results saved to: experimentruns/.../run_000
```

## When Something Goes Wrong

### Error: "Missing required configuration fields"
**Solution**: Use full config name from `config_store.py`, don't create custom configs

### Error: "Path not properly configured"
**Solution**: Check `project_root` and `experiment_root` in config_store.py

### Error: "Failed to create experiment directory"
**Solution**: Check that parent directories exist and have write permissions

### GPU out of memory
**Solution**: Reduce `batch_size` or use smaller model size

### Training stuck
**Solution**: Check logs with `tail -f experimentruns/.../logs/lightning_logs.csv`

## Summary

1. **Configs are pre-generated** - Use names from `config_store.py`
2. **Paths are deterministic** - No fallbacks, strict error handling  
3. **Sweeps are easy** - Use `-m` with comma-separated values
4. **Slurm integration is clean** - `oscjobmanager.py` handles script generation
5. **Reproducibility is built-in** - Every run saves its config
6. **Error messages are informative** - They tell you exactly what's wrong

Good luck with your research! 🚀
