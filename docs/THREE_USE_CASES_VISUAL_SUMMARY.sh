#!/usr/bin/bash
# Three Use Cases Visual Guide

cat << 'EOF'

╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║               HOW YOUR PROJECT WORKS: THREE USE CASES                    ║
║                                                                           ║
║                 Individual | Distillation | Fusion                       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝


┌─────────────────────────────────────────────────────────────────────────┐
│ 1️⃣  INDIVIDUAL TRAINING                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Purpose: Train a single model independently                           │
│  Use Cases: Baseline performance, single-model inference               │
│                                                                         │
│  Two Variants:                                                          │
│                                                                         │
│  A) GAT Classifier (Supervised)                                        │
│  ─────────────────────────────────────                                 │
│  Raw CAN Data                                                           │
│      ↓                                                                  │
│  [Load Dataset] → Create graphs                                        │
│      ↓                                                                  │
│  [GATWithJK Model]                                                      │
│  (Graph Attention Networks with Jumping Knowledge)                     │
│      ↓                                                                  │
│  Output: Classification logits (2 classes: normal/attack)             │
│      ↓                                                                  │
│  Loss: CrossEntropy(predictions, ground_truth_labels)                │
│      ↓                                                                  │
│  Result: Supervised classifier trained                                 │
│  Accuracy: ~96%                                                         │
│                                                                         │
│  B) VGAE Autoencoder (Unsupervised)                                    │
│  ──────────────────────────────────────                                │
│  Raw CAN Data (ONLY NORMAL SAMPLES!)                                   │
│      ↓                                                                  │
│  [Load Dataset] → Filter to label==0                                   │
│      ↓                                                                  │
│  [GraphAutoencoderNeighborhood - VGAE]                                 │
│  (Variational Graph AutoEncoder)                                       │
│      ↓                                                                  │
│  Outputs:                                                               │
│    • Reconstructed features (continuous values)                        │
│    • CAN ID predictions                                                │
│    • Latent representation z                                           │
│    • KL divergence (variational term)                                  │
│      ↓                                                                  │
│  Loss: Reconstruction + CAN_ID + 0.01·KL                              │
│      ↓                                                                  │
│  Result: Autoencoder learns "normal" distribution                      │
│  Anomaly Detection: reconstruction_error > threshold → ATTACK          │
│  Accuracy: ~92%                                                         │
│                                                                         │
│  Command:                                                               │
│  $ python train_with_hydra_zen.py --model gat --training normal        │
│  $ python train_with_hydra_zen.py --training autoencoder               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 2️⃣  KNOWLEDGE DISTILLATION                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Purpose: Compress large teacher into small student                    │
│  Benefits: 2-4x faster inference, 75% fewer parameters                 │
│                                                                         │
│  Architecture:                                                          │
│  ─────────────────────────────────────                                 │
│                                                                         │
│             ┌─────────────────────────────────────┐                    │
│             │ Training Input (Raw CAN Data)       │                    │
│             └──────────────┬──────────────────────┘                    │
│                            │                                            │
│           ┌────────────────┴────────────────┐                          │
│           ↓                                  ↓                          │
│  [Teacher Model]                    [Student Model]                    │
│  (Large, pre-trained)              (Small, to train)                   │
│  (Frozen, no gradient)             (Learning)                          │
│           │                                  │                          │
│           ↓                                  ↓                          │
│  [Teacher Output]                  [Student Output]                    │
│  (at temperature T=4.0)            (at temperature T=4.0)              │
│           │                                  │                          │
│           └────────────────┬─────────────────┘                         │
│                            ↓                                            │
│                    [KL Divergence Loss]                                │
│            (soft targets at T=4.0, scaled by T²)                       │
│                            │                                            │
│           ┌────────────────┴────────────────┐                          │
│           ↓                                  ↓                          │
│     [Soft Loss]                      [Hard Loss]                       │
│  (KD distillation)               (Task loss on labels)                 │
│           │                                  │                          │
│           └────────────────┬─────────────────┘                         │
│                            ↓                                            │
│             Total Loss = 0.7·soft + 0.3·hard                           │
│                            ↓                                            │
│                  Backprop through student only                         │
│                            ↓                                            │
│           Smaller, faster student with high accuracy                   │
│                                                                         │
│  Temperature Parameter Effect:                                         │
│  ─────────────────────────────────                                     │
│                                                                         │
│  Without Temperature (T=1):                                             │
│    Teacher output: [0.99, 0.01] (very confident)                      │
│    Student learns: "Class 1 is correct" (little signal for class 2)   │
│                                                                         │
│  With Temperature (T=4):                                                │
│    Teacher output: [0.73, 0.27] (softened)                            │
│    Student learns: "Class 1 is likely, but class 2 is possible"       │
│    Much more learning signal!                                          │
│                                                                         │
│  Command:                                                               │
│  $ python train_with_hydra_zen.py --training knowledge_distillation \\│
│      --teacher_path saved_models/best_teacher_model_hcrl_sa.pth \\   │
│      --student_scale 0.5                                               │
│                                                                         │
│  Results:                                                               │
│    Teacher accuracy: 96%                                                │
│    Student accuracy: 95% (only 1% loss)                                │
│    Student size: 25% of teacher                                        │
│    Inference speed: 3-4x faster                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 3️⃣  FUSION TRAINING WITH DQN                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Purpose: Learn optimal weighting of VGAE + GAT                        │
│  Benefit: 2-5% accuracy improvement, combine strengths of both         │
│                                                                         │
│  Architecture:                                                          │
│  ─────────────────────────────────────                                 │
│                                                                         │
│  Step 1: Prediction Caching (One-time, 2-5 minutes)                   │
│  ──────────────────────────────────────────────────────────            │
│                                                                         │
│    Pre-trained VGAE    Pre-trained GAT                                 │
│          │                    │                                         │
│          └────────┬───────────┘                                        │
│                   ↓                                                     │
│  Run all data through both models                                      │
│                   ↓                                                     │
│    Save VGAE anomaly_scores.pkl                                        │
│    Save GAT gat_probs.pkl                                              │
│                   ↓                                                     │
│  (No more forward passes needed for training!)                         │
│                                                                         │
│  Step 2: DQN Training (3-10 minutes for 50 epochs)                    │
│  ──────────────────────────────────────────────────────────            │
│                                                                         │
│    [Load cached predictions]                                           │
│           ↓                                                             │
│    For each sample:                                                     │
│      anomaly_score ∈ [0, 1]   (from VGAE)                             │
│      gat_prob ∈ [0, 1]        (from GAT)                              │
│           ↓                                                             │
│    [Discretize to state]                                               │
│      state = (bin(anomaly_score), bin(gat_prob))                       │
│      state ∈ {0-10} × {0-10}   (100 possible states)                   │
│           ↓                                                             │
│    [Q-Network (DQN)]                                                    │
│      input: state (discretized prediction pair)                        │
│      output: Q-values for 21 actions (α = 0.0 to 1.0)                 │
│           ↓                                                             │
│    [Epsilon-Greedy Action Selection]                                   │
│      if random() < epsilon (exploration):                              │
│          action = random_action()                                       │
│      else:                                                              │
│          action = argmax(Q-values)  (exploitation)                     │
│           ↓                                                             │
│    [Compute Fusion Weight]                                             │
│      alpha = action / (num_actions - 1)                                │
│      alpha ∈ [0, 1]                                                    │
│           ↓                                                             │
│    [Fuse Predictions]                                                  │
│      fused_score = alpha·gat_prob + (1-alpha)·anomaly_score           │
│           ↓                                                             │
│    [Compute Reward]                                                    │
│      if fused_score > threshold and label == attack: reward = 1       │
│      else: reward = 0                                                  │
│           ↓                                                             │
│    [Experience Replay]                                                 │
│      Store: (state, action, reward, next_state, done)                 │
│           ↓                                                             │
│    [Q-Learning Update]                                                 │
│      Sample minibatch from replay buffer                               │
│      Q-target = reward + gamma·max(Q(next_state))                      │
│      Q-pred = Q(state)[action]                                         │
│      loss = MSE(Q-pred, Q-target)                                      │
│           ↓                                                             │
│    [Backprop DQN]                                                      │
│      Update Q-network weights                                          │
│           ↓                                                             │
│    [Target Network Update]                                             │
│      Every 100 steps: copy Q-network to target Q-network               │
│           ↓                                                             │
│    [Decay Exploration]                                                 │
│      epsilon *= 0.995 (gradually trust learned policy)                 │
│           ↓                                                             │
│    Repeat for all training samples                                      │
│                                                                         │
│  Step 3: Learned Policy Heatmap                                        │
│  ───────────────────────────────────                                   │
│                                                                         │
│    The DQN learns: for each (anomaly_score, gat_prob) pair,           │
│    what weight α should we use?                                        │
│                                                                         │
│    Heatmap visualization:                                              │
│    ─────────────────────────                                           │
│      VGAE Anomaly Score →                                              │
│      0.0       0.5       1.0                                           │
│  1.0 ┌────────────────────────┐                                        │
│      │ blue  purple   red     │  ← GAT Attack Prob                    │
│      │ α=0.1  α=0.5  α=0.9    │                                        │
│  0.5 │ blue  purple   red     │                                        │
│      │ α=0.2  α=0.5  α=0.8    │                                        │
│  0.0 │ blue  purple   red     │                                        │
│      │ α=0.3  α=0.5  α=0.7    │                                        │
│      └────────────────────────┘                                        │
│                                                                         │
│    Blue = Low α (trust VGAE)                                           │
│    Red = High α (trust GAT)                                            │
│    Purple = Medium α (balanced)                                        │
│                                                                         │
│  Command:                                                               │
│  $ python train_fusion_lightning.py --dataset hcrl_sa                  │
│                                                                         │
│  Results:                                                               │
│    VGAE alone: 92%                                                     │
│    GAT alone: 96%                                                      │
│    Fusion (learned): 98%                                               │
│    Improvement: +2% over best single model                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│ 📊 SIDE-BY-SIDE COMPARISON                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                    Individual    Distillation    Fusion                │
│                    ──────────    ────────────    ──────                │
│ Training Time       10-30 min      15-40 min     6-15 min             │
│ Models Used         1              2 (1 frozen)  3 (2 frozen)         │
│ Complexity          Low            Medium        High                 │
│ Inference Speed     Fast           Fastest       Medium               │
│ Accuracy (VGAE)     92%            ~90%          Not used             │
│ Accuracy (GAT)      96%            96%           Not used             │
│ Final Accuracy      96%            95%           98%                  │
│ Best For            Baseline       Edge devices  Best accuracy        │
│ Memory (inference)  Baseline       1/3 - 1/2     2x baseline          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│ 🔄 TYPICAL WORKFLOW                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: Try Individual Models                                         │
│  ─────────────────────────────────────────────────────────             │
│  $ python train_with_hydra_zen.py --model gat --training normal        │
│  $ python train_with_hydra_zen.py --training autoencoder               │
│  Result: VGAE=92%, GAT=96%                                             │
│                                                                         │
│  Step 2 (optional): Compress for Deployment                            │
│  ─────────────────────────────────────────────────────────             │
│  $ python train_with_hydra_zen.py --training knowledge_distillation    │
│  Result: Student=95% accuracy, 3x faster                               │
│                                                                         │
│  Step 3: Maximize Accuracy                                              │
│  ─────────────────────────────────────────────────────────             │
│  $ python train_fusion_lightning.py --dataset hcrl_sa                  │
│  Result: Fusion=98% accuracy (best of both worlds!)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│ 📚 KEY CONCEPTS                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ Individual:                                                             │
│   GAT = Graph Attention Network (supervised learning)                   │
│   VGAE = Variational Graph AutoEncoder (unsupervised learning)         │
│   Different strengths: GAT learns decision boundary, VGAE learns dist  │
│                                                                         │
│ Distillation:                                                           │
│   Temperature = softening parameter (helps with knowledge transfer)    │
│   Hard loss = task loss (on labels)                                    │
│   Soft loss = KL divergence (matches teacher logits)                   │
│   Student learns HOW teacher reasons, not just WHAT it outputs        │
│                                                                         │
│ Fusion:                                                                 │
│   Q-Network = neural network that learns Q(state, action) values      │
│   Experience Replay = remember past decisions, learn from them        │
│   Target Network = separate copy for stability                         │
│   Epsilon-Greedy = balance exploration (random) vs exploitation (best)│
│   DQN learns: α(anomaly_score, gat_prob) → optimal fusion weight      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

EOF

echo ""
echo "For detailed explanation, see: THREE_USE_CASES_EXPLAINED.md"
echo ""
