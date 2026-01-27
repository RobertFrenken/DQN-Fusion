# Frozen Config Pattern - Implementation Roadmap

**Status**: Infrastructure complete, integration in progress

---

## What's Been Completed ✅

### 1. Core Serialization (src/config/frozen_config.py)
- [x] `save_frozen_config()` - serialize CANGraphConfig to JSON
- [x] `load_frozen_config()` - deserialize JSON to CANGraphConfig
- [x] Handles nested dataclasses with type metadata
- [x] Version tracking for compatibility
- [x] Linter issues fixed (unused imports, parameters)

### 2. SLURM Integration (src/cli/job_manager.py)
- [x] `SLURM_FROZEN_CONFIG_TEMPLATE` - new template using frozen configs
- [x] `save_frozen_config()` method - persists config to disk
- [x] `generate_script()` - defaults to frozen config mode (`use_frozen_config=True`)
- [x] Fallback to legacy CLI mode if frozen config save fails
- [x] Dependency handling preserved

### 3. Training Script Support (train_with_hydra_zen.py)
- [x] `--frozen-config` argument added
- [x] Frozen config mode loads JSON and bypasses all other args
- [x] Configuration summary printed on load
- [x] Error handling for missing/invalid frozen configs
- [x] Examples added to help text

### 4. Documentation
- [x] [MAINTENANCE.md](.claude/MAINTENANCE.md) - Update strategy created
- [x] [INDEX.md](.claude/INDEX.md) - Updated with frozen config info
- [x] Frozen Config Pattern documented in INDEX.md Recent Changes

---

## What Still Needs to be Done 🚧

### Phase 1: Verification & Testing (SHORT TERM)
Priority: **HIGH** - Verify end-to-end flow works

**1a. Test Frozen Config Flow**
- [ ] Manually test single job submission with frozen configs
- [ ] Verify config loads correctly from JSON
- [ ] Check SLURM script contains correct frozen config path
- [ ] Verify training completes successfully
- [ ] Check that reproducibility works (same config = same results)

**Command to test**:
```bash
cd /users/PAS2022/rf15/CAN-Graph-Test/KD-GAT
./can-train single --model gat --dataset hcrl_sa --training normal --modality automotive --dry-run
# Should show SLURM script with --frozen-config path
```

**1b. Test Pipeline Frozen Configs**
- [ ] Test multi-stage pipeline with frozen configs
- [ ] Verify dependency handling still works
- [ ] Check all configs are saved with timestamps
- [ ] Verify each stage loads its own frozen config

**1c. Error Handling**
- [ ] Test corrupt frozen config file
- [ ] Test missing frozen config file
- [ ] Test version mismatch handling
- [ ] Verify fallback to legacy mode works

---

### Phase 2: Documentation & Examples (MEDIUM TERM)
Priority: **MEDIUM** - Make it discoverable and understandable

**2a. Update CLI_BEST_PRACTICES.md**
Add sections:
- [ ] Frozen config workflow (when it's used automatically)
- [ ] Manual frozen config creation (if needed)
- [ ] Where frozen configs are saved (`experimentruns/{dataset}/*/configs/`)
- [ ] How to manually load a frozen config for reproduction
- [ ] Debugging frozen config issues

**2b. Create Frozen Config Examples**
- [ ] Add to [PENDING_WORK.md](Tasks/PENDING_WORK.md):
  - "Test frozen config single job"
  - "Test frozen config pipeline"
  - "Verify reproducibility with frozen configs"

**2c. Update README/Main Docs**
- [ ] Add reference to Frozen Config Pattern in README (if present)
- [ ] Link to MAINTENANCE.md for doc updates

---

### Phase 3: Monitoring & Observability (MEDIUM TERM)
Priority: **MEDIUM** - Make it debuggable

**3a. Logging & Debugging**
- [ ] Ensure frozen config path logged in SLURM script output
- [ ] Add debug mode: `--debug-frozen-config` to show serialized/deserialized config
- [ ] Log timing of config serialization/deserialization
- [ ] Add warnings if config is large (>1MB)

**3b. Config Manifest** (Future Enhancement)
- [ ] Create `manifest.json` alongside frozen config:
  ```json
  {
    "frozen_config_version": "1.0",
    "frozen_at": "2026-01-26T19:30:00",
    "git_sha": "abc123...",
    "config_hash": "sha256:...",
    "file_path": "configs/frozen_config_20260126_193000.json"
  }
  ```

---

### Phase 4: Optimization & Polish (LONG TERM)
Priority: **LOW** - Nice-to-haves

**4a. Performance**
- [ ] Consider compressing frozen configs if >5MB
- [ ] Cache deserialization for repeated loads
- [ ] Benchmark serialization speed

**4b. Compatibility**
- [ ] Add migration script for old configs
- [ ] Document breaking changes if any
- [ ] Version bump plan

**4c. Automation**
- [ ] Pre-commit hook to validate frozen config format
- [ ] CI check: frozen configs can be loaded
- [ ] Auto-clean old frozen configs (>30 days)

---

## Current Flow (How It Works Now)

```
User: ./can-train single ...
        ↓
CLI builds config via config_builder.create_can_graph_config()
        ↓
submit_single(config=None, run_type={...}, ...)
        ↓
generate_script(config, use_frozen_config=True)  ← uses frozen by default!
        ↓
save_frozen_config(config, experiment_dir/configs/)
        ↓
SLURM script generated with: python train_with_hydra_zen.py --frozen-config /path
        ↓
SLURM job submits script
        ↓
Training node runs script, loads frozen config
        ↓
HydraZenTrainer.train(config)
        ↓
Results saved to experiment_dir
```

---

## Fallback Flow (If Frozen Config Fails)

```
save_frozen_config() raises exception
        ↓
logs warning, falls back to legacy mode
        ↓
SLURM script generated with: python train_with_hydra_zen.py --model gat --dataset ...
        ↓
Training runs in legacy mode
```

This ensures robustness - if frozen config fails, jobs still submit.

---

## Key Implementation Details

### Where Frozen Configs Are Saved
```
experimentruns/
└── automotive/
    └── hcrl_sa/
        └── [run_directory]/
            └── configs/
                ├── frozen_config_20260126_193000.json
                └── frozen_config_20260126_194500.json
```

### Frozen Config Structure
```json
{
  "_frozen_config_version": "1.0",
  "_frozen_at": "2026-01-26T19:30:00.123456",
  "_type": "CANGraphConfig",
  "model": {
    "_type": "GATConfig",
    "hidden_channels": 64,
    ...
  },
  "dataset": {...},
  "training": {...},
  ...
}
```

### Loading Flow in train_with_hydra_zen.py
```python
if args.frozen_config:
    from src.config.frozen_config import load_frozen_config
    config = load_frozen_config(args.frozen_config)
    # ALL other CLI args ignored - config is frozen
    trainer = HydraZenTrainer(config)
    trainer.train()
```

---

## Testing Checklist

Before marking Phase 1 complete:

- [ ] **Single Job Test**
  ```bash
  ./can-train single --model gat --dataset hcrl_sa --training normal --modality automotive --dry-run
  # Verify: frozen_config path in script
  ```

- [ ] **Pipeline Test** (if available)
  ```bash
  ./can-train pipeline --model vgae,gat --learning-type unsupervised,supervised --training-strategy autoencoder,curriculum --dataset hcrl_sa --modality automotive --dry-run
  # Verify: multiple frozen configs with timestamps
  ```

- [ ] **Load Frozen Config Test**
  ```bash
  python train_with_hydra_zen.py --frozen-config /path/to/frozen_config.json --dry-run
  # Verify: config loads, summary prints, no errors
  ```

- [ ] **Reproducibility Test**
  ```bash
  # Run same config twice, compare frozen files
  # They should be identical (except timestamps)
  ```

- [ ] **Error Handling Test**
  ```bash
  python train_with_hydra_zen.py --frozen-config /nonexistent/path.json
  # Verify: Clear error message, proper exit code
  ```

---

## Dependencies

- [x] `src/config/frozen_config.py` - CREATED
- [x] `src/config/hydra_zen_configs.py` - Already has dataclasses
- [x] `src/cli/job_manager.py` - Updated
- [x] `train_with_hydra_zen.py` - Updated
- [ ] Tests (not yet created)

---

## Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| Frozen config serialization fails silently | Added logging, fallback to legacy mode |
| Large configs (>10MB) slow down I/O | Monitor config sizes, compress if needed |
| Version incompatibility | Version tracking in frozen config JSON |
| Deserialization of nested dataclasses breaks | Type metadata stored, comprehensive class_map |
| Old jobs can't find frozen config after moves | Absolute paths in SLURM script, document best practices |

---

## Success Criteria

✅ Phase 1 (Verification):
- All jobs submitted use frozen configs by default
- Frozen configs successfully load in training script
- Reproducibility verified

✅ Phase 2 (Documentation):
- Users understand how/why frozen configs work
- Examples show common workflows
- MAINTENANCE.md kept up-to-date

✅ Phase 3 (Monitoring):
- Issues with frozen configs are debuggable
- Frozen config paths clearly logged
- Serialization/deserialization tracked

---

## Next Session

Start with **Phase 1a: Test Frozen Config Flow**

1. Run `./can-train single --help` to see options
2. Generate a dry-run SLURM script
3. Verify frozen config path is in the script
4. Manually run `python train_with_hydra_zen.py --frozen-config <path>`
5. Document findings in PENDING_WORK.md

If tests pass → Mark Phase 1 ✅ → Move to Phase 2 documentation
