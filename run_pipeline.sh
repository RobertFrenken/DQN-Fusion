#!/bin/bash
./can-train pipeline \
  --modality automotive \
  --model vgae,gat,dqn \
  --learning-type unsupervised,supervised,rl_fusion \
  --training-strategy autoencoder,curriculum,fusion \
  --dataset hcrl_sa \
  --model-size student \
  --distillation no-kd \
  --epochs 100 \
  --submit