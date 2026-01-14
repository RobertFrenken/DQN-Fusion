# How Your Project Works: Three Training Use Cases

Your CAN-Graph project implements **three distinct training approaches** for anomaly detection on CAN bus networks. Each approach serves a different purpose and has its own training pipeline. Here's how they work:

---

## 🎯 Overview: The Three Use Cases

| Use Case | Purpose | Input | Output | Best For |
|----------|---------|-------|--------|----------|
| **Individual Training** | Train single model (GAT or VGAE) independently | Raw CAN data | Classification/Anomaly model | Baseline performance, single-model inference |
| **Knowledge Distillation** | Train smaller student from larger teacher | Teacher checkpoint + data | Smaller, faster student model | Deployment with limited compute |
| **Fusion Training** | Learn optimal weighting of multiple models | Pre-trained VGAE + GAT | Dynamic fusion weights via DQN | Best accuracy with multiple models |

---

## 1️⃣ INDIVIDUAL TRAINING

### Purpose
Train a single model (GAT classifier or VGAE autoencoder) on its own without any knowledge transfer or fusion. This is the baseline approach.

### Entry Point
```bash
# Train GAT classifier
python train_with_hydra_zen.py --model gat --dataset hcrl_sa --training normal

# Train VGAE autoencoder
python train_with_hydra_zen.py --training autoencoder --dataset hcrl_sa
```

### Architecture Flow

```
Raw CAN Data (train/val splits)
        ↓
[load_dataset] → GraphDataset (creates pyG graphs)
        ↓
[create_dataloaders] → DataLoader with variable-size batching
        ↓
[CANGraphLightningModule] with two variants:
        ├─ GAT Model (for supervised classification)
        │   └─ GATWithJK → [node embeddings] → [classification logits]
        │
        └─ VGAE Model (for unsupervised anomaly detection)
            └─ GraphAutoencoderNeighborhood → [reconstruction + KL loss]
        ↓
[training_step] → Compute loss → Backprop
        ↓
[validation_step] → Compute metrics
        ↓
Trained model saved → `saved_models/gat_hcrl_sa.pth` or similar
```

### Training Loop Details

#### For GAT (Supervised Classification)

```python
# In CANGraphLightningModule._normal_training_step()

1. Forward pass:
   output = self.model(batch)
   # output shape: (batch_size, num_classes)
   # Contains logits for CAN ID classification

2. Loss computation:
   loss = nn.functional.cross_entropy(output, batch.y)
   # Compares predicted logits with ground truth labels

3. Backward pass:
   optimizer.step()  # Lightning handles this automatically

4. Validation:
   Same forward pass but in eval mode (no dropout)
   Compute accuracy, precision, recall
```

**Key Config Parameters:**
```python
model_config:
  input_dim: 4             # CAN message features
  hidden_channels: 64      # Hidden dimension
  output_dim: 2            # Number of classes (normal/attack)
  num_layers: 3            # Graph attention layers
  heads: 8                 # Attention heads
  dropout: 0.3

training_config:
  learning_rate: 0.001
  batch_size: 32
  max_epochs: 50
  early_stopping_patience: 10
```

#### For VGAE (Unsupervised Anomaly Detection)

```python
# In CANGraphLightningModule._autoencoder_training_step()

1. Data filtering:
   # Only use normal samples (label == 0) for training
   normal_mask = batch.y == 0
   filtered_batch = batch[normal_mask]

2. Forward pass:
   output = self.forward(filtered_batch)
   # output: (reconstructed_features, canid_logits, neighbor_logits, latent_z, kl_loss)

3. Loss computation:
   reconstruction_loss = MSE(reconstructed, original_features)
   canid_loss = CrossEntropy(canid_logits, canid_labels)
   total_loss = reconstruction_loss + canid_loss + 0.01 * kl_loss

4. Backward & optimize:
   optimizer.step()

5. Inference (anomaly detection):
   reconstruction_error = ||original - reconstructed||²
   if error > threshold → ANOMALY
```

**Why Filter to Normal Samples?**
- The autoencoder learns the distribution of normal CAN bus traffic
- By only seeing normal examples during training, it learns what "normal" looks like
- During inference, anomalies will have high reconstruction error because they deviate from learned patterns
- This is unsupervised: no attack labels needed during training

### Output Files
```
saved_models/
├── gat_hcrl_sa.pth                    (supervised classifier)
├── autoencoder_hcrl_sa.pth            (unsupervised anomaly detector)
└── lightning_checkpoints/
    └── gat_hcrl_sa_epoch10_val_loss0.45.ckpt
```

### Performance Metrics
- **GAT:** Accuracy, Precision, Recall, F1-score (supervised)
- **VGAE:** Reconstruction error distribution, TPR@FPR (unsupervised)

---

## 2️⃣ KNOWLEDGE DISTILLATION

### Purpose
Train a smaller, faster student model that mimics a larger teacher model. Useful for:
- **Model compression:** Reduce VRAM and inference time
- **Deployment:** Run on edge devices with limited compute
- **Knowledge transfer:** Student learns from teacher's learned patterns

### Entry Point
```bash
# Train small student from large teacher
python train_with_hydra_zen.py \
    --training knowledge_distillation \
    --dataset hcrl_sa \
    --teacher_path saved_models/best_teacher_model_hcrl_sa.pth \
    --student_scale 0.5
```

Or use preset:
```bash
python train_with_hydra_zen.py --preset distillation_hcrl_sa_scale_0.5
```

### Architecture Flow

```
Pre-trained Teacher Model
        ↓
[Load from checkpoint] → Freeze weights, set to eval mode
        ↓
Raw CAN Data
        ↓
[create_dataloaders] → Same as individual training
        ↓
Split into two streams:

    Stream 1: Student                    Stream 2: Teacher (frozen)
    ↓                                    ↓
[Student Model]                         [Teacher Model]
[GAT/VGAE, smaller]                     [GAT/VGAE, larger]
    ↓                                    ↓
[Student logits]                        [Teacher logits] (no grad)
    ↓                                    ↓
    └────────────→ [KD Loss] ←─────────┘
                        ↓
              ┌─────────┼─────────┐
              ↓                   ↓
         Hard Loss           Soft Loss
         (task loss)      (distillation)
              ↓                   ↓
              └─────────┬─────────┘
                        ↓
        total_loss = α·soft_loss + (1-α)·hard_loss
                        ↓
              Backprop through student only
```

### Training Loop Details

```python
# In CANGraphLightningModule._knowledge_distillation_step()

1. Student forward pass:
   student_output = self.forward(batch)

2. Teacher forward pass (no gradients):
   with torch.no_grad():
       teacher_output = self.teacher_model(batch)

3. Compute two losses:
   
   # Hard loss: standard task loss
   hard_loss = CrossEntropy(student_output, batch.y)
   
   # Soft loss: KL divergence at temperature T
   temperature = 4.0  # config parameter
   soft_targets = softmax(teacher_output / T)
   soft_prob = log_softmax(student_output / T)
   soft_loss = KL(soft_prob, soft_targets) * (T²)

4. Combine losses:
   alpha = 0.7  # weight for soft vs hard
   total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

5. Backprop:
   total_loss.backward()  # Only updates student weights
   optimizer.step()
```

### Temperature Parameter Explanation

Temperature softens the probability distributions:
- **T = 1.0:** Standard softmax (hard probability peaks)
- **T = 4.0:** Softer probabilities (student learns nuanced patterns)
- **T → ∞:** Uniform distribution (all classes equally likely)

**Why use temperature?**
- Without it, teacher's confident predictions (e.g., [0.99, 0.01]) don't provide learning signal
- With temperature, they become [0.73, 0.27], allowing gradient flow for wrong classes
- Student learns not just what teacher predicts, but HOW teacher reasons

### Model Scaling

Your config allows scaling student size:

```python
student_scale: 0.5  # Half-size student

Before scaling:
  hidden_channels: 64

After scaling:
  hidden_channels: 32  # = 64 * 0.5
```

**Benefits:**
- Reduced parameters: $\approx O(n^2)$ for graph networks, so 0.5 scale ≈ 0.25 params
- Faster inference: ~2-4x speedup
- Still high accuracy: knowledge transfer makes up for smaller capacity

### Output Files
```
saved_models/
├── student_model_hcrl_sa_scale_0.5.pth    (compressed model)
├── distillation_hcrl_sa_epoch20_val_loss0.32.ckpt
└── logs/
    └── distillation_hcrl_sa/metrics.csv
```

### Performance Metrics Logged
```
train_distillation_loss    # Combined loss
hard_loss                   # Task loss component
soft_loss                   # Distillation loss component
teacher_accuracy            # How well teacher performs (reference)
student_accuracy            # How well student performs
accuracy_gap                # teacher_acc - student_acc (should be small)
teacher_student_similarity  # Cosine similarity of outputs
```

**Expected Results:**
- Student accuracy: 95-99% of teacher accuracy
- Inference time: 2-4x faster
- Model size: 0.25-0.5x of teacher

---

## 3️⃣ FUSION TRAINING

### Purpose
Learn **dynamic per-sample fusion weights** that optimally combine VGAE (unsupervised anomaly scores) and GAT (supervised attack probability) predictions.

Instead of always weighting them the same way (e.g., 50-50), the DQN learns: *"For this specific sample, weight GAT 70% and VGAE 30%"*

### Entry Point
```bash
# Using main script
python train_with_hydra_zen.py --preset fusion_hcrl_sa

# Or standalone
python train_fusion_lightning.py --dataset hcrl_sa --max-epochs 50

# Quick demo
python fusion_training_demo.py --dataset hcrl_sa --quick
```

### Architecture Flow

```
Pre-trained Models
  ├─ VGAE Autoencoder
  └─ GAT Classifier
           ↓
[PredictionCacheBuilder]
  • Run inference on all training data
  • Extract VGAE anomaly scores: shape (N,)
  • Extract GAT probabilities: shape (N,)
  • Cache to disk (pickle)
           ↓
[FusionPredictionCache Dataset]
  • Loads cached predictions (not raw data)
  • No model forward passes during training
  • Enables 4-5x speedup
           ↓
[FusionLightningModule - DQN Agent]
  • Q-network learns optimal fusion weights
  • Input: (anomaly_score, gat_prob) → discrete state
  • Output: learned weight alpha ∈ [0, 1]
  • fused_score = α·gat_prob + (1-α)·anomaly_score
           ↓
[Experience Replay Buffer]
  • Store transitions: (state, action, reward, next_state)
  • Sample minibatches for Q-learning updates
  • Stabilizes training with decorrelated samples
           ↓
[Target Network]
  • Separate Q-network copy
  • Updated every 100 steps
  • Reduces overestimation bias
           ↓
[Epsilon-Greedy Exploration]
  • ε=0.9 initially (explore 90% of the time)
  • Decay to ε=0.2 over training
  • Balance exploration vs exploitation
           ↓
Learned Fusion Policy
  ├─ fusion_agent_hcrl_sa.pth      (DQN weights)
  └─ fusion_policy_hcrl_sa.npy     (heatmap visualization)
```

### Why Prediction Caching?

**Without caching (old approach):**
```
For each training batch:
  1. Run VGAE forward pass → wait for GPU
  2. Run GAT forward pass → wait for GPU
  3. Compute DQN loss
  4. Backprop
  = 3+ model forward passes per batch
  = Very inefficient
```

**With caching (new approach):**
```
One-time setup (2-5 minutes):
  1. Run all data through VGAE → save outputs
  2. Run all data through GAT → save outputs
  3. Cache to disk

During training (3-10 minutes for 50 epochs):
  1. Load cached predictions (CPU memory)
  2. Compute DQN loss (GPU)
  3. Backprop (GPU)
  = Only DQN computations on GPU
  = 4-5x faster, 70-85% GPU utilization
```

### DQN Training Loop

```python
# In FusionLightningModule.training_step()

1. Get batch of cached predictions:
   anomaly_scores.shape = (batch_size,)
   gat_probs.shape = (batch_size,)
   labels.shape = (batch_size,)

2. Discretize to state:
   state = discretize_state(anomaly_scores, gat_probs, bins=10)
   # state.shape = (batch_size, 2) with values 0-9 each

3. Epsilon-greedy action selection:
   if random() < epsilon:
       action = random_action()  # Explore
   else:
       q_values = q_network(state)
       action = argmax(q_values)  # Exploit

4. Compute fusion weight:
   alpha = action / (alpha_steps - 1)  # Map [0,20] → [0, 1]
   # e.g., if action=10 and alpha_steps=21: alpha = 10/20 = 0.5

5. Compute fused score:
   fused_score = alpha * gat_probs + (1 - alpha) * anomaly_scores

6. Compute reward (immediate feedback):
   reward = 1.0 if detection is correct else 0.0
   # Or more sophisticated: TPR, F1-score, AUC

7. Store in experience replay buffer:
   replay_buffer.push((state, action, reward, next_state, done))

8. Sample minibatch from buffer:
   batch_size = 256
   for (s, a, r, s', done) in sample_minibatch(256):

9. Q-learning update (Bellman equation):
   q_target = r + gamma * max(q_network(s')) if not done
   q_pred = q_network(s)[a]
   loss = MSE(q_pred, q_target)

10. Backprop:
    loss.backward()
    optimizer.step()

11. Decay epsilon:
    epsilon *= epsilon_decay  # e.g., 0.995
    epsilon = max(epsilon, min_epsilon)  # Clamp to [0.2, 1.0]

12. Update target network (every 100 steps):
    if step % target_update_freq == 0:
        target_q_network.load_state_dict(q_network.state_dict())
```

### Configuration Parameters

```python
FusionAgentConfig:
  alpha_steps: 21                  # Discretization: [0/20, 1/20, ..., 20/20]
  fusion_lr: 0.001                 # Q-network learning rate
  gamma: 0.9                       # Temporal discount factor
  fusion_epsilon: 0.9              # Initial exploration rate
  fusion_epsilon_decay: 0.995      # Decay per training step
  fusion_min_epsilon: 0.2          # Minimum exploration rate
  fusion_buffer_size: 100000       # Experience replay memory
  fusion_batch_size: 256           # Training batch size
  target_update_freq: 100          # Update target net every N steps
```

### Output Files
```
saved_models/
├── fusion_agent_hcrl_sa.pth              (DQN Q-network weights)
├── fusion_policy_hcrl_sa.npy             (learned fusion weights heatmap)
├── fusion_checkpoints/hcrl_sa/
│   ├── fusion_epoch00_val_accuracy0.942.ckpt
│   └── fusion_epoch10_val_accuracy0.958.ckpt
└── logs/fusion_hcrl_sa/
    └── metrics.csv                       (training curves)
```

### What the Policy Heatmap Shows

```
         VGAE Anomaly Score →
         0.0      0.5      1.0
    ┌─────────────────────────┐
1.0 │  blue    purple   red   │  ← GAT Attack Prob
    │ α=0.1   α=0.5    α=0.9  │
0.5 │  blue    purple   red   │
    │ α=0.2   α=0.5    α=0.8  │
0.0 │  blue    purple   red   │
    │ α=0.3   α=0.5    α=0.7  │
    └─────────────────────────┘

Red zones: High α (trust GAT more)
Blue zones: Low α (trust VGAE more)
Purple zones: Medium α (balanced)

This heatmap shows what the DQN learned!
```

### Expected Results

**Before Fusion Training:**
- VGAE accuracy: 92%
- GAT accuracy: 96%
- Best single model: 96%

**After Fusion Training:**
- Fused accuracy: 98%
- Improvement: +2% over best single model
- Some samples trust VGAE more, others trust GAT more

---

## 📊 Comparison Table

| Aspect | Individual | Distillation | Fusion |
|--------|-----------|--------------|--------|
| **Training Time** | 10-30 min | 15-40 min | 6-15 min (with caching) |
| **Compute Needed** | 1 GPU | 1 GPU | 1 GPU + cached predictions |
| **Model Complexity** | Single model | 2 models (frozen teacher) | 3 models (2 pre-trained, 1 DQN) |
| **Inference Models** | 1 | 1 (student) | 2-3 (all models for fusion) |
| **Inference Speed** | Fast | Fastest | Medium (multiple forward passes) |
| **Accuracy** | Baseline | Near-teacher | Best |
| **Memory (inference)** | Baseline | 25-50% of baseline | Baseline + VGAE/GAT |
| **Research Value** | High (baseline) | Medium (compression) | Highest (novel fusion) |

---

## 🔄 Typical Project Workflow

### Step 1: Establish Baselines (Individual Training)
```bash
# Train autoencoder for anomaly detection
python train_with_hydra_zen.py --training autoencoder --dataset hcrl_sa

# Train classifier for attack detection
python train_with_hydra_zen.py --model gat --dataset hcrl_sa

# Compare: which is better?
```

### Step 2: Compress Models (Knowledge Distillation)
```bash
# If inference latency is critical
python train_with_hydra_zen.py \
    --training knowledge_distillation \
    --dataset hcrl_sa \
    --teacher_path saved_models/best_teacher_model_hcrl_sa.pth \
    --student_scale 0.5
```

### Step 3: Optimize Accuracy (Fusion Training)
```bash
# If you have both pre-trained models
python train_fusion_lightning.py --dataset hcrl_sa --max-epochs 50

# Check improvement:
# "Before fusion: 96% accuracy"
# "After fusion: 98% accuracy"
```

---

## 🎓 Key Concepts Summary

### Individual Training
- **What:** Single model trained on labeled or unlabeled data
- **How:** Standard supervised (GAT) or unsupervised (VGAE) learning
- **Loss:** CrossEntropy (GAT) or Reconstruction+KL (VGAE)

### Knowledge Distillation
- **What:** Small student learns from large teacher
- **How:** Soft target matching with temperature scaling
- **Loss:** α·soft_loss + (1-α)·hard_loss
- **Why:** Trade accuracy for speed (2-4x faster)

### Fusion Training
- **What:** DQN learns optimal weight between VGAE and GAT
- **How:** Q-learning with experience replay and target network
- **Loss:** Bellman equation: r + γ·max(Q(s'))
- **Why:** Leverage strengths of both models (2-5% accuracy gain)

---

**Need more details on any specific use case? Check:**
- Individual: [train_models.py](train_models.py#L240) - `_normal_training_step()` and `_autoencoder_training_step()`
- Distillation: [train_models.py](train_models.py#L271) - `_knowledge_distillation_step()`
- Fusion: [train_with_hydra_zen.py](train_with_hydra_zen.py#L197) - `_train_fusion()` method
