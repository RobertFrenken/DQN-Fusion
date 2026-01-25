# KD-GAT + Hydra-Zen Integration Guide

**Welcome!** You've received a complete Hydra-Zen training system for your KD-GAT project. This document explains what you have and how to use it.

## Quick Start (2 minutes)

1. **Read this file** (you're doing it!)
2. **Open `INTEGRATION_TODO.md`** to track progress
3. **Follow the 5-step integration** in `INTEGRATION_SUMMARY.md`
4. **Use code templates** from `INTEGRATION_CODE_TEMPLATES.md`
5. **Debug as needed** with `INTEGRATION_DEBUGGING.md`

## What You've Received

### Core System Files (5)
Complete, production-ready training framework:

1. **`hydra_configs/config_store.py`** (650+ lines)
   - 100+ pre-generated experiment configurations
   - Type-safe configuration using Hydra-Zen
   - All combinations: modality × dataset × learning_type × architecture × size × distillation × training_mode

2. **`src/utils/experiment_paths.py`** (350+ lines)
   - Deterministic path generation from config hierarchy
   - Strict validation (no fallbacks, informative errors)
   - Auto-creates experiment directory structure

3. **`src/training/train_with_hydra_zen.py`** (450+ lines)
   - Main training entry point
   - Integrates Hydra-Zen + PyTorch Lightning
   - MLflow experiment tracking
   - Includes data loader placeholder ⚠️ (you'll implement)

4. **`src/training/lightning_modules.py`** (550+ lines)
   - Base Lightning module with optimizer/scheduler config
   - VGAE module (unsupervised)
   - GAT module (supervised/fusion)
   - DQN module (RL/fusion)
   - Includes model instantiation placeholders ⚠️ (you'll link)

5. **`oscjobmanager.py`** (400+ lines)
   - Slurm job submission for Ohio Supercomputer Center
   - Auto-generates submission scripts
   - Dry-run preview before actual submission

### Documentation (6)
Comprehensive guides for every aspect:

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **`INTEGRATION_SUMMARY.md`** | Overview of integration process | 10 min |
| **`INTEGRATION_TODO.md`** | Step-by-step checklist to track progress | 5 min |
| **`INTEGRATION_CODE_TEMPLATES.md`** | Exact code patterns to copy | 20 min |
| **`INTEGRATION_DEBUGGING.md`** | Common errors & solutions | 30 min |
| **`KD-GAT_INTEGRATION_GUIDE.md`** | Detailed integration instructions | 15 min |
| **`README_INTEGRATION.md`** | This file - overview | 10 min |

## Your Integration Task

You need to do **5 things** (total ~2 hours):

### 1. Update Dataset Classes (30 min)
**File:** `src/data/datasets.py`

Wrap your datasets in PyTorch `Dataset` class. Make them return dict batches.

**Template in:** `INTEGRATION_CODE_TEMPLATES.md` → Template 1

### 2. Update Model Classes (30 min)
**Files:** `src/models/vgae.py`, `src/models/gat.py`, `src/models/dqn.py`

Make models accept configurable parameters instead of hardcoding values.

**Template in:** `INTEGRATION_CODE_TEMPLATES.md` → Template 2

### 3. Implement Data Loaders (20 min)
**File:** `src/training/train_with_hydra_zen.py`

Replace the `load_data_loaders()` placeholder with actual implementation.

**Template in:** `INTEGRATION_CODE_TEMPLATES.md` → Template 3

### 4. Update Configuration (20 min)
**File:** `hydra_configs/config_store.py`

Set correct paths and parameters for your project.

### 5. Link Models to Lightning (20 min)
**File:** `src/training/lightning_modules.py`

Update model instantiation in Lightning modules.

## Getting Started

**Open these files in order:**

1. ⭐ **README_INTEGRATION.md** (this file) - Overview
2. **INTEGRATION_SUMMARY.md** - 5-step process
3. **INTEGRATION_TODO.md** - Track your progress here ✅
4. **INTEGRATION_CODE_TEMPLATES.md** - Copy exact code while working
5. **INTEGRATION_DEBUGGING.md** - Reference when things break

---

**Next:** Open `INTEGRATION_TODO.md` and start Phase 1! 🚀
