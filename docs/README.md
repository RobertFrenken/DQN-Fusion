# KD-GAT Documentation

**Complete guide to training CAN intrusion detection models with knowledge distillation**

---

## 🚀 Start Here

**New to KD-GAT?** → [GETTING_STARTED.md](GETTING_STARTED.md)

Follow the 5-minute setup guide to train your first model.

---

## 📚 Documentation Index

### Essential Guides

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [**GETTING_STARTED**](GETTING_STARTED.md) | Quick setup & first training | New users, initial setup |
| [**CODE_TEMPLATES**](CODE_TEMPLATES.md) | Copy-paste code snippets | Need working examples |
| [**WORKFLOW_GUIDE**](WORKFLOW_GUIDE.md) | Job submission & pipelines | Submitting to SLURM |
| [**TROUBLESHOOTING**](TROUBLESHOOTING.md) | Common errors & solutions | Hitting errors |

### Reference Documentation

| Document | Purpose |
|----------|---------|
| [**QUICK_REFERENCES**](QUICK_REFERENCES.md) | Fast command lookup |
| [**ARCHITECTURE_SUMMARY**](ARCHITECTURE_SUMMARY.md) | System architecture |
| [**JOB_TEMPLATES**](JOB_TEMPLATES.md) | Complete job configurations |
| [**SUBMITTING_JOBS**](SUBMITTING_JOBS.md) | Detailed job submission |

### Advanced Topics

| Document | Purpose |
|----------|---------|
| [**EXPERIMENTAL_DESIGN**](EXPERIMENTAL_DESIGN.md) | Research methodology |
| [**MODEL_SIZE_CALCULATIONS**](MODEL_SIZE_CALCULATIONS.md) | Parameter budgets (LaTeX) |
| [**DEPENDENCY_MANIFEST**](DEPENDENCY_MANIFEST.md) | Manifest format spec |
| [**MLflow_SETUP**](MLflow_SETUP.md) | Experiment tracking |

---

## 🎯 Quick Navigation

### By Task

**I want to...**

- **Train my first model** → [GETTING_STARTED.md](GETTING_STARTED.md#2-first-training-run-normal-gat)
- **Copy working code** → [CODE_TEMPLATES.md](CODE_TEMPLATES.md)
- **Submit a job to OSC** → [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#1-job-submission-workflow)
- **Fix an error** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Use knowledge distillation** → [GETTING_STARTED.md](GETTING_STARTED.md#b-knowledge-distillation)
- **Train with curriculum learning** → [GETTING_STARTED.md](GETTING_STARTED.md#c-curriculum-learning)
- **Run fusion training** → [GETTING_STARTED.md](GETTING_STARTED.md#d-multi-model-fusion)
- **Configure my model** → [CODE_TEMPLATES.md](CODE_TEMPLATES.md#configuration-templates)
- **Chain multiple jobs** → [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md#4-job-chaining-pipeline)
- **Understand the architecture** → [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md)

### By Role

**Research/Student**:
1. Start: [GETTING_STARTED.md](GETTING_STARTED.md)
2. Examples: [CODE_TEMPLATES.md](CODE_TEMPLATES.md)
3. Experiments: [EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md)

**Developer**:
1. Architecture: [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md)
2. Templates: [CODE_TEMPLATES.md](CODE_TEMPLATES.md)
3. Reference: [QUICK_REFERENCES.md](QUICK_REFERENCES.md)

**Cluster User**:
1. Workflow: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)
2. Jobs: [JOB_TEMPLATES.md](JOB_TEMPLATES.md)
3. Submission: [SUBMITTING_JOBS.md](SUBMITTING_JOBS.md)

---

## 🎓 Learning Path

### Beginner (Day 1)

1. **Setup** (30 min)
   - Install environment: [GETTING_STARTED.md](GETTING_STARTED.md#1-quick-setup-5-minutes)
   - Download datasets
   - Verify installation

2. **First Model** (1 hour)
   - Train GAT: [GETTING_STARTED.md](GETTING_STARTED.md#2-first-training-run-normal-gat)
   - Understand configs: [GETTING_STARTED.md](GETTING_STARTED.md#3-configuration-basics)
   - Check results

### Intermediate (Week 1)

3. **All Model Types** (1 day)
   - Train VGAE autoencoder
   - Train DQN fusion agent
   - Compare performance

4. **Advanced Training** (2 days)
   - Knowledge distillation: [GETTING_STARTED.md](GETTING_STARTED.md#b-knowledge-distillation)
   - Curriculum learning: [EXPERIMENTAL_DESIGN.md](EXPERIMENTAL_DESIGN.md)
   - Multi-model fusion

5. **Cluster Usage** (1 day)
   - Submit jobs: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)
   - Monitor runs
   - Chain pipelines

### Advanced (Ongoing)

6. **Experimentation**
   - Hyperparameter sweeps
   - Custom training modes
   - Research experiments

7. **Production**
   - Model deployment
   - Performance optimization
   - MLflow tracking

---

## 🔧 Configuration System

**Single source of truth**: `src/config/hydra_zen_configs.py`

All configuration in one place:
- Model configs (GAT, VGAE, DQN)
- Dataset configs
- Training modes
- Config store & validation

See [GETTING_STARTED.md](GETTING_STARTED.md#7-key-configuration-files) for details.

---

## 📊 Model Types

| Model | Parameters | Purpose | Training Time |
|-------|------------|---------|---------------|
| **GAT Teacher** | ~1.1M | Supervised classification | 4h |
| **GAT Student** | ~55K | Onboard deployment | 2h |
| **VGAE Teacher** | ~1.74M | Unsupervised reconstruction | 6h |
| **VGAE Student** | ~87K | Onboard autoencoder | 3h |
| **DQN Teacher** | ~687K | Fusion agent | 6h |
| **DQN Student** | ~32K | Onboard fusion | 3h |

See [MODEL_SIZE_CALCULATIONS.md](MODEL_SIZE_CALCULATIONS.md) for parameter budgets.

---

## 🎯 Training Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **normal** | Standard supervised | Classification |
| **autoencoder** | Unsupervised VGAE | Anomaly detection |
| **curriculum** | Hard sample mining | Improved accuracy |
| **knowledge_distillation** | Teacher→Student | Model compression |
| **fusion** | Multi-model DQN | Ensemble learning |

See [GETTING_STARTED.md](GETTING_STARTED.md#4-common-workflows) for workflows.

---

## 📁 Project Structure

```
KD-GAT/
├── data/                         # CAN datasets
├── experiment_runs/              # Training outputs (canonical)
├── src/
│   ├── config/
│   │   └── hydra_zen_configs.py  # ⭐ All configs
│   ├── models/                   # Model architectures
│   ├── training/                 # Training logic
│   └── paths.py                  # Path management
├── train_with_hydra_zen.py       # ⭐ Main training script
├── oscjobmanager.py              # ⭐ Job submission
└── docs/                         # ⭐ This documentation
```

---

## 💡 Tips

### Getting Help

1. **Check error message** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. **Search docs** → Use browser find (Ctrl+F)
3. **Look at examples** → [CODE_TEMPLATES.md](CODE_TEMPLATES.md)
4. **Test locally first** → Catch errors before cluster submission

### Best Practices

- ✅ Read [GETTING_STARTED.md](GETTING_STARTED.md) first
- ✅ Test with `--fast-dev-run` locally
- ✅ Use `--dry-run` before submitting jobs
- ✅ Validate configs before training
- ✅ Monitor GPU usage
- ✅ Keep experiment notes

### Common Pitfalls

- ❌ Wrong dataset path → [TROUBLESHOOTING.md](TROUBLESHOOTING.md#dataset-not-found)
- ❌ CUDA OOM → [TROUBLESHOOTING.md](TROUBLESHOOTING.md#out-of-memory-oom)
- ❌ Missing teacher model → [TROUBLESHOOTING.md](TROUBLESHOOTING.md#missing-teacher-model)
- ❌ Config errors → [TROUBLESHOOTING.md](TROUBLESHOOTING.md#config-validation-failed)

---

## 🔄 Updates

**Latest Changes** (2026-01-24):
- ✅ Configuration system consolidated (single source of truth)
- ✅ Documentation reduced from 30 to 12 files
- ✅ New comprehensive guides created
- ✅ All obsolete docs archived

See [../CLEANUP_COMPLETE.md](../CLEANUP_COMPLETE.md) for details.

---

## 📞 Support

**Have questions?**

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review [QUICK_REFERENCES.md](QUICK_REFERENCES.md)
3. Search existing issues
4. Consult [ARCHITECTURE_SUMMARY.md](ARCHITECTURE_SUMMARY.md)

---

**Ready to start?** → [GETTING_STARTED.md](GETTING_STARTED.md)
