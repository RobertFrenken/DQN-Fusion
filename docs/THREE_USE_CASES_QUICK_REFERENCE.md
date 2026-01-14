# Quick Reference: Three Use Cases

## At a Glance

### 🎯 **Individual Training** - Single Model Baseline
Train one model independently (GAT classifier or VGAE autoencoder).
```bash
python train_with_hydra_zen.py --model gat --training normal      # GAT
python train_with_hydra_zen.py --training autoencoder             # VGAE
```
**Goal:** Establish baseline performance  
**Output:** One trained model (92-96% accuracy)  
**Use When:** Need single-model simplicity

---

### 📦 **Knowledge Distillation** - Model Compression
Train small student model that mimics large teacher model.
```bash
python train_with_hydra_zen.py \
    --training knowledge_distillation \
    --teacher_path saved_models/best_teacher_model_hcrl_sa.pth \
    --student_scale 0.5
```
**Goal:** Trade accuracy for speed (2-4x faster, 75% fewer params)  
**Output:** Compressed student model (95% of teacher accuracy)  
**Use When:** Need to deploy on edge devices or reduce inference latency

---

### 🧠 **Fusion Training** - Optimal Combination
Learn DQN that dynamically weights VGAE + GAT on per-sample basis.
```bash
python train_fusion_lightning.py --dataset hcrl_sa --max-epochs 50
# OR
python train_with_hydra_zen.py --preset fusion_hcrl_sa
```
**Goal:** Maximize accuracy by combining strengths of both models  
**Output:** DQN fusion agent (98% accuracy)  
**Use When:** Want best possible accuracy and can afford multiple model inference

---

## Decision Tree

```
                    Want to train a model?
                            |
                    ┌───────┴───────┐
                    ↓               ↓
            Need fast inference?  Need best accuracy?
                    |                  |
                  YES                 YES
                    |                  |
                    ↓                  ↓
         Knowledge Distillation    Fusion Training
         (2-4x faster inference)   (2-5% accuracy gain)
                    
            If unsure: Individual Training first
            (establish baselines, understand data)
```

---

## Loss Functions

| Use Case | Loss Function | Why |
|----------|---------------|-----|
| **Individual (GAT)** | CrossEntropy(predictions, labels) | Classification: predict correct class |
| **Individual (VGAE)** | MSE(reconstructed, original) + KL(latent) | Reconstruction: learn normal patterns |
| **Distillation** | α·KL(soft_targets) + (1-α)·CrossEntropy(hard) | Combine knowledge transfer + task loss |
| **Fusion** | MSE(Q_pred, Q_target) where target = r + γ·max(Q(s')) | Temporal difference (Bellman equation) |

---

## Output Models

| Use Case | Files | What They Do |
|----------|-------|--------------|
| **Individual (GAT)** | `gat_*.pth` | Classify CAN IDs (supervised) |
| **Individual (VGAE)** | `autoencoder_*.pth` | Detect anomalies by reconstruction error (unsupervised) |
| **Distillation** | `student_model_*.pth` | Predict like teacher, but 3-4x faster |
| **Fusion** | `fusion_agent_*.pth` + `fusion_policy_*.npy` | Learn optimal α for each (anomaly_score, gat_prob) pair |

---

## Key Differences

### Individual Training
**Input:** Raw CAN data  
**Process:** Train model from scratch  
**Output:** Single model predictions  
**Speed:** Baseline (reference)  
**Accuracy:** Single-model baseline

### Knowledge Distillation
**Input:** Raw CAN data + Teacher model  
**Process:** Student learns from teacher's logits  
**Output:** Smaller model with similar performance  
**Speed:** 2-4x faster than teacher  
**Accuracy:** 95-99% of teacher

### Fusion Training
**Input:** Cached VGAE + GAT predictions  
**Process:** DQN learns optimal weight α for each sample  
**Output:** Learned fusion weights  
**Speed:** Multiple forward passes (slower inference)  
**Accuracy:** 2-5% better than best single model

---

## Configuration Highlights

### Individual Training
```python
# Most important parameters
model.type = "gat" or "vgae"
training.mode = "normal" or "autoencoder"
training.learning_rate = 0.001
training.batch_size = 32
training.max_epochs = 50
```

### Knowledge Distillation
```python
# Special parameters for distillation
training.mode = "knowledge_distillation"
training.teacher_model_path = "path/to/teacher.pth"
training.student_model_scale = 0.5  # 50% of teacher size
training.distillation_temperature = 4.0  # Softens probability
training.distillation_alpha = 0.7  # Weight soft vs hard loss
```

### Fusion Training
```python
# DQN parameters
training.alpha_steps = 21              # 21 discretized weights
fusion.fusion_lr = 0.001               # Q-network learning rate
fusion.gamma = 0.9                     # Temporal discount
fusion.fusion_epsilon = 0.9            # Exploration rate
fusion.fusion_epsilon_decay = 0.995    # Decay per step
fusion.fusion_buffer_size = 100000     # Experience replay buffer
```

---

## When to Use What

| Scenario | Use This | Why |
|----------|----------|-----|
| First time exploring data | Individual (GAT) | Supervised learning is easiest |
| Want unsupervised baseline | Individual (VGAE) | Learn normal patterns without labels |
| Need deployment on edge | Distillation | Reduce model size 3-4x |
| Have tight latency SLA | Distillation | Fastest inference |
| Want maximum accuracy | Fusion | Combines both model strengths |
| Research / publication | Fusion | Novel approach, best results |
| Need interpretability | Individual | Single model easier to explain |
| Can afford inference cost | Fusion | Multiple forward passes OK |

---

## Command Cheat Sheet

```bash
# ===== INDIVIDUAL TRAINING =====
# GAT classifier
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal

# VGAE autoencoder
python train_with_hydra_zen.py --training autoencoder --dataset hcrl_sa

# ===== KNOWLEDGE DISTILLATION =====
# Compress GAT with distillation
python train_with_hydra_zen.py \
    --training knowledge_distillation \
    --dataset hcrl_sa \
    --teacher_path saved_models/best_teacher_model_hcrl_sa.pth \
    --student_scale 0.5

# ===== FUSION TRAINING =====
# Full fusion training
python train_fusion_lightning.py --dataset hcrl_sa --max-epochs 50

# Quick fusion demo
python fusion_training_demo.py --dataset hcrl_sa --quick

# Using main script
python train_with_hydra_zen.py --preset fusion_hcrl_sa

# ===== UTILITIES =====
# Verify setup
python verify_fusion_setup.py

# List available presets
python train_with_hydra_zen.py --list-presets
```

---

## Files to Read for Each Use Case

| Use Case | Main File | Key Methods |
|----------|-----------|-------------|
| Individual | `train_models.py` | `_normal_training_step()`, `_autoencoder_training_step()` |
| Distillation | `train_models.py` | `setup_knowledge_distillation()`, `_knowledge_distillation_step()` |
| Fusion | `train_with_hydra_zen.py` | `_train_fusion()` |
| Fusion | `src/training/fusion_lightning.py` | `FusionLightningModule.training_step()` |
| Fusion Cache | `src/training/prediction_cache.py` | `PredictionCacheBuilder` |

---

## Expected Results

| Metric | Individual GAT | Individual VGAE | Distillation | Fusion |
|--------|---|---|---|---|
| Accuracy | 96% | 92% | 95% | 98% |
| Model Size | 100% | 100% | 25% | 300% (3 models) |
| Inference Time | 1.0x | 1.0x | 0.25x | 2.0x |
| Training Time | 20 min | 20 min | 30 min | 6 min (cached) |
| Best For | Baseline | Unsupervised | Deployment | Maximum accuracy |

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Individual training slow | Large dataset | Reduce batch_size or use GPU |
| Distillation: student acc much lower | Temperature too high | Reduce distillation_temperature to 2.0 |
| Fusion training slow | No cached predictions | Run prediction cache first |
| Models not loading | Wrong checkpoint format | Check if .pth has 'state_dict' key |

---

For more details:
- **Complete Explanation:** [THREE_USE_CASES_EXPLAINED.md](THREE_USE_CASES_EXPLAINED.md)
- **Visual Summary:** [THREE_USE_CASES_VISUAL_SUMMARY.sh](THREE_USE_CASES_VISUAL_SUMMARY.sh)
- **Fusion Details:** [FUSION_TRAINING_GUIDE.md](FUSION_TRAINING_GUIDE.md)
