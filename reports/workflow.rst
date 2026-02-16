KD-GAT: CAN Bus Intrusion Detection Pipeline
=============================================

Three-stage knowledge distillation pipeline for CAN bus intrusion detection:

1. **VGAE** (autoencoder): Unsupervised graph reconstruction learns latent
   representations of CAN message windows.

2. **GAT** (curriculum): Supervised classification with curriculum learning
   detects attack types from graph-structured CAN frames.

3. **DQN** (fusion): Reinforcement learning agent fuses VGAE and GAT
   predictions into a final intrusion detection decision.

Each stage trains large and small model variants. Small models are optionally
compressed from large models via knowledge distillation (KD) auxiliaries
for edge deployment.

Datasets: hcrl_ch, hcrl_sa, set_01, set_02, set_03, set_04.
