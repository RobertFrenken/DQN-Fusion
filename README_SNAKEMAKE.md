# Snakemake Pipeline for KD-GAT Training

This directory contains a Snakemake-based pipeline system that replaces the custom `can-train` CLI for managing GNN training experiments.

## 🎯 Quick Start

### 1. Install Snakemake
```bash
conda install -c conda-forge snakemake
```

### 2. Setup
```bash
chmod +x profiles/slurm/slurm-status.py
```

### 3. Test
```bash
# Dry run (see what will execute)
snakemake --profile profiles/slurm -n

# Visualize pipeline
snakemake --dag | dot -Tpdf > dag.pdf
```

### 4. Run
```bash
# Train all teacher models
snakemake --profile profiles/slurm all_teachers --jobs 20

# Train everything (teachers + students)
snakemake --profile profiles/slurm --jobs 20
```

## 📚 Documentation

| File | Purpose |
|------|---------|
| [SNAKEMAKE_SUMMARY.md](SNAKEMAKE_SUMMARY.md) | **START HERE** - Overview and key benefits |
| [SNAKEMAKE_QUICKSTART.md](SNAKEMAKE_QUICKSTART.md) | Common commands and examples |
| [SNAKEMAKE_MIGRATION_PLAN.md](SNAKEMAKE_MIGRATION_PLAN.md) | Detailed migration guide |
| [SNAKEMAKE_MIGRATION_TODO.md](SNAKEMAKE_MIGRATION_TODO.md) | Step-by-step checklist |

## 🗂️ Files Created

```
.
├── Snakefile                               # Main pipeline definition
├── config/
│   └── snakemake_config.yaml              # Configuration (datasets, SLURM settings)
├── profiles/
│   └── slurm/
│       ├── config.yaml                     # SLURM executor settings
│       └── slurm-status.py                 # Job status checker
├── envs/
│   └── gnn-experiments.yaml               # Conda environment
└── examples/
    └── train_with_hydra_zen_snakemake_adapter.py  # Example adapter code
```

## 🎓 How It Works

### Pipeline Structure

```
Teacher Pipeline (per dataset):
  VGAE (autoencoder) → GAT (curriculum) → DQN (fusion) → Evaluation

Student Pipeline (per dataset):
  (waits for teachers) → VGAE-KD → GAT-KD → DQN-KD → Evaluation
```

### Automatic Features

✅ **Dependency Resolution**: GAT waits for VGAE, DQN waits for both
✅ **Parallel Execution**: Different datasets run in parallel
✅ **Failure Recovery**: Resume from any point automatically
✅ **Provenance Tracking**: Complete DAG of execution

## ⚙️ Configuration

### Add/Remove Datasets

Edit `config/snakemake_config.yaml`:
```yaml
datasets:
  - hcrl_sa
  - hcrl_ch
  - set_01
  - my_new_dataset  # ← Add here
```

### Customize SLURM Resources

Edit `profiles/slurm/config.yaml`:
```yaml
default-resources:
  - slurm_account=PAS3209
  - slurm_partition=gpu
  - runtime=360  # minutes
  - mem_mb=64000
```

Or override per run:
```bash
snakemake --profile profiles/slurm \
    --set-resources train_dqn:mem_mb=128000 \
    --jobs 20
```

## 🔧 Critical Code Change Required

⚠️ **You must modify `train_with_hydra_zen.py`** to support simple CLI arguments from Snakemake.

See `examples/train_with_hydra_zen_snakemake_adapter.py` for example code.

### Why?

Snakemake calls your training script like this:
```bash
python train_with_hydra_zen.py \
    --model vgae \
    --model-size teacher \
    --dataset hcrl_sa \
    --output-dir /path/to/output
```

Your current script expects frozen configs from `can-train`. The adapter code lets it support both.

## 📋 Common Commands

```bash
# Dry run (see what will execute)
snakemake --profile profiles/slurm -n

# Run specific dataset
snakemake --profile profiles/slurm \
    results/automotive/hcrl_sa/teacher/no_distillation/evaluation/dqn_eval.json

# Resume from failures
snakemake --profile profiles/slurm --rerun-incomplete --jobs 20

# Force rerun specific stage
snakemake --profile profiles/slurm --forcerun train_gat --jobs 20

# Monitor jobs
watch -n 10 'squeue -u $USER'

# Check logs
tail -f experimentruns/automotive/hcrl_sa/vgae/teacher/no_distillation/autoencoder/logs/training.log
```

## 🐛 Troubleshooting

### Jobs Not Submitting
```bash
# Test SLURM access
sacctmgr show assoc where user=$USER
sinfo -o "%20P %5a %10l %6D"

# Test status script
python profiles/slurm/slurm-status.py <job_id>
```

### Missing Input Files
```bash
# Check what Snakemake expects
snakemake --profile profiles/slurm --summary

# See why a file will be created
snakemake --profile profiles/slurm -n -r <target_file>
```

### Locked Directory
```bash
# If Snakemake was interrupted
snakemake --unlock
```

See [SNAKEMAKE_MIGRATION_PLAN.md](SNAKEMAKE_MIGRATION_PLAN.md#common-issues--solutions) for more.

## 🚀 Benefits Over Old System

| Feature | Old (can-train) | New (Snakemake) |
|---------|----------------|-----------------|
| **Lines of code** | 2,000+ | ~400 |
| **Dependency tracking** | Manual job IDs | Automatic (file-based) |
| **Resume from failures** | Manual resubmit | Automatic |
| **Pipeline visualization** | None | Built-in DAG |
| **Parallel execution** | Manual coordination | Automatic |
| **Reproducibility** | Good (frozen configs) | Excellent (DAG + configs) |
| **Community support** | None (custom) | Large Snakemake community |

## 📊 Example: Run Full Pipeline

```bash
# 1. Dry run to see plan
snakemake --profile profiles/slurm -n

# 2. Visualize dependencies
snakemake --dag | dot -Tpdf > pipeline_dag.pdf

# 3. Run all teachers (6 datasets × 3 models = 18 training jobs + evaluations)
snakemake --profile profiles/slurm all_teachers --jobs 20

# 4. Monitor progress
watch -n 10 'squeue -u $USER'

# 5. Once teachers complete, run students (waits for teachers automatically)
snakemake --profile profiles/slurm all_students --jobs 20

# 6. Generate summary report
snakemake --profile profiles/slurm generate_report
```

## 🗺️ Migration Roadmap

1. **Read** [SNAKEMAKE_SUMMARY.md](SNAKEMAKE_SUMMARY.md) ← Start here
2. **Install** Snakemake and setup profile
3. **Modify** `train_with_hydra_zen.py` (critical!)
4. **Test** with single dataset
5. **Validate** results match old pipeline
6. **Migrate** fully

Estimated time: 1-2 weeks

See [SNAKEMAKE_MIGRATION_TODO.md](SNAKEMAKE_MIGRATION_TODO.md) for detailed checklist.

## 🤝 Getting Help

- **Quick reference**: [SNAKEMAKE_QUICKSTART.md](SNAKEMAKE_QUICKSTART.md)
- **Detailed guide**: [SNAKEMAKE_MIGRATION_PLAN.md](SNAKEMAKE_MIGRATION_PLAN.md)
- **Official docs**: https://snakemake.readthedocs.io/
- **Tutorial**: https://snakemake.readthedocs.io/en/stable/tutorial/tutorial.html

## 🎯 Success Criteria

Migration is complete when:
- ✅ All datasets train via Snakemake
- ✅ Results match old pipeline
- ✅ Dependencies work automatically
- ✅ Failures recover gracefully
- ✅ Team is comfortable with new system

## 🔄 Backward Compatibility

During migration:
- `can-train` CLI still works
- Old shell scripts still functional
- Gradual transition supported

After migration:
- Old scripts moved to `legacy/` directory
- Can keep `can-train` for quick ad-hoc experiments (optional)

## 📝 Next Steps

1. Read [SNAKEMAKE_SUMMARY.md](SNAKEMAKE_SUMMARY.md)
2. Follow [SNAKEMAKE_MIGRATION_TODO.md](SNAKEMAKE_MIGRATION_TODO.md)
3. Test with single dataset
4. Validate and migrate

Good luck! 🚀
