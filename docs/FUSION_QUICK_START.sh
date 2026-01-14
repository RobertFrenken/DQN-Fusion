#!/bin/bash
# Quick reference for fusion training commands

echo "========== FUSION TRAINING QUICK REFERENCE =========="
echo ""

echo "1️⃣  Train fusion using main script:"
echo "   python train_with_hydra_zen.py --preset fusion_hcrl_sa"
echo "   python train_with_hydra_zen.py --training fusion --dataset set_04"
echo ""

echo "2️⃣  Train fusion using dedicated script:"
echo "   python train_fusion_lightning.py --dataset hcrl_sa"
echo "   python train_fusion_lightning.py --dataset set_04 --max-epochs 100"
echo ""

echo "3️⃣  List available datasets:"
echo "   ls datasets/can-train-and-test-v1.5/"
echo ""

echo "4️⃣  Check available models:"
echo "   ls saved_models/*.pth | grep -E '(autoencoder|best_teacher)'
echo ""

echo "5️⃣  View training curves:"
echo "   tensorboard --logdir logs/fusion_*"
echo ""

echo "6️⃣  View cached predictions:"
echo "   ls cache/fusion/"
echo ""

echo "========== EXAMPLE WORKFLOWS =========="
echo ""

echo "WORKFLOW 1: Full pipeline from scratch"
echo "  1. Train VGAE autoencoder:    python train_with_hydra_zen.py --training autoencoder --dataset hcrl_sa"
echo "  2. Train GAT classifier:      python train_with_hydra_zen.py --model gat --dataset hcrl_sa"
echo "  3. Train fusion agent:        python train_fusion_lightning.py --dataset hcrl_sa"
echo ""

echo "WORKFLOW 2: Quick test with existing models"
echo "  1. Check saved models exist:  ls saved_models/ | grep hcrl_sa"
echo "  2. Train fusion:              python train_fusion_lightning.py --dataset hcrl_sa --max-epochs 20"
echo ""

echo "WORKFLOW 3: Evaluate learned fusion policy"
echo "  1. Load fusion model:         python -c \"import torch; m = torch.load('saved_models/fusion_agent_hcrl_sa.pth'); print(m.keys())\""
echo "  2. Plot policy:               python scripts/visualize_fusion_policy.py --agent saved_models/fusion_agent_hcrl_sa.pth"
echo ""

echo "========== ARGUMENTS =========="
echo ""

echo "Main training script (train_with_hydra_zen.py):"
echo "  --preset PRESET              Use predefined config (e.g., fusion_hcrl_sa)"
echo "  --list-presets               List all available presets"
echo "  --model {gat,vgae}           Model type (default: gat)"
echo "  --dataset DATASET            Dataset (hcrl_sa, hcrl_ch, set_01-04, car_hacking)"
echo "  --training {normal,autoencoder,knowledge_distillation,fusion}"
echo "  --epochs N                   Number of epochs"
echo "  --batch_size N               Batch size"
echo "  --learning_rate LR           Learning rate"
echo ""

echo "Fusion training script (train_fusion_lightning.py):"
echo "  --dataset DATASET            Dataset name (required)"
echo "  --autoencoder PATH           Path to VGAE checkpoint"
echo "  --classifier PATH            Path to GAT checkpoint"
echo "  --max-epochs N               Max training epochs (default: 50)"
echo "  --batch-size N               Fusion batch size (default: 256)"
echo "  --lr LR                      Q-network learning rate (default: 0.001)"
echo "  --device {cuda,cpu}          Device (default: cuda)"
echo ""

echo "========== FILE LOCATIONS =========="
echo ""

echo "Fusion source code:"
echo "  src/training/fusion_lightning.py      Main Lightning module"
echo "  src/training/prediction_cache.py      Cache builder"
echo "  src/config/fusion_config.py           Hydra-Zen configs"
echo ""

echo "Training scripts:"
echo "  train_with_hydra_zen.py               Main training entry point"
echo "  train_fusion_lightning.py             Dedicated fusion training"
echo ""

echo "Documentation:"
echo "  FUSION_TRAINING_GUIDE.md              Complete guide (you are here)"
echo "  KNOWLEDGE_DISTILLATION_README.md      Teacher-student framework"
echo "  PROJECT_STRUCTURE.md                  Codebase layout"
echo ""

echo "Output locations:"
echo "  saved_models/fusion_*/                Checkpoints and agents"
echo "  logs/fusion_*/                        Training metrics (CSV)"
echo "  cache/fusion/                         Cached predictions"
echo ""

echo "========== COMMON TASKS =========="
echo ""

echo "Task: Train multiple datasets"
echo "  for ds in hcrl_sa hcrl_ch set_04; do"
echo "    python train_fusion_lightning.py --dataset \$ds --max-epochs 50"
echo "  done"
echo ""

echo "Task: Compare fusion vs individual models"
echo "  python -c \"from train_models import *; \\"
echo "  print('VGAE baseline'); print('GAT baseline'); print('Fusion result')\""
echo ""

echo "Task: Check if cache exists for a dataset"
echo "  ls cache/fusion/ | grep hcrl_sa && echo 'Cache found' || echo 'Cache missing - will rebuild'"
echo ""

echo "Task: Clear old caches to save disk space"
echo "  rm -rf cache/fusion/old_*"
echo ""

echo "========== TROUBLESHOOTING =========="
echo ""

echo "❌ Problem: 'Autoencoder not found'"
echo "✅ Solution: Train it first or check saved_models/ path"
echo "   python train_with_hydra_zen.py --training autoencoder --dataset hcrl_sa"
echo ""

echo "❌ Problem: 'CUDA out of memory during cache build'"
echo "✅ Solution: Process on CPU instead"
echo "   python train_fusion_lightning.py --dataset hcrl_sa --device cpu"
echo ""

echo "❌ Problem: 'Policy stuck at α=0.5'"
echo "✅ Solution: Increase learning rate or adjust epsilon decay"
echo ""

echo "❌ Problem: Cache mismatch errors"
echo "✅ Solution: Rebuild cache from scratch"
echo "   rm -rf cache/fusion/ && python train_fusion_lightning.py --dataset hcrl_sa"
echo ""

echo "========== HELP =========="
echo ""

echo "For detailed documentation:"
echo "  cat FUSION_TRAINING_GUIDE.md"
echo ""

echo "For script help:"
echo "  python train_with_hydra_zen.py -h"
echo "  python train_fusion_lightning.py -h"
echo ""

echo "========== END REFERENCE =========="
