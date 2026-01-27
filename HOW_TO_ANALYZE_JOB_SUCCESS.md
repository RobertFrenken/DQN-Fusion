# How to Analyze Job Success Rates

## The Discrepancy: 95% vs 73%

### Initial Count (Early in Session)
**Claimed**: 37/38 succeeded (97.4%)
**Method**: Quick grep of error logs, visual inspection
**Problem**: Counted jobs that started but didn't check completion status

### Corrected Count (After Parsing)
**Actual**: 28/38 succeeded (73.7%)
**Method**: Parsed SLURM output files systematically
**Source**: [scripts/parse_job_results.py](scripts/parse_job_results.py)

## The Proper Method

### Use the Parsing Scripts

**Step 1: Parse all SLURM output files**
```bash
python scripts/parse_job_results.py
```

This generates:
- Terminal output with summary table
- [job_results.json](job_results.json) with all parsed data

**Step 2: Generate pipeline-level summary**
```bash
python scripts/pipeline_summary.py
```

This shows:
- Pipeline-level aggregation (3-stage VGAE→GAT→DQN)
- Per-job details with timing and batch sizes
- Failure breakdown by error type

### What to Look For

**Success Indicators**:
```
✅ JOB COMPLETED SUCCESSFULLY
Exit code: 0
```

**Failure Indicators**:
```
❌ Training failed: <error message>
❌ Failed to load frozen config: <error>
CUDA out of memory
Exit code: <non-zero>
```

### Common Mistakes

❌ **Don't count jobs that:**
- Started but crashed
- Have .out files but no completion message
- Were cancelled by SLURM dependencies

✅ **Do verify:**
- Presence of final model files (*.pth)
- Completion timestamps in logs
- Exit codes in output files

## Current Status Breakdown (Jan 27, 2026)

### By Stage
| Stage | Success | Failed | Rate |
|-------|---------|--------|------|
| VGAE  | 12/12   | 0/12   | 100% |
| GAT   | 11/13   | 2/13   | 84.6% |
| DQN   | 5/13    | 8/13   | 38.5% |

### By Model Size
| Size    | Success | Failed | Rate |
|---------|---------|--------|------|
| Teacher | 20/21   | 1/21   | 95.2% |
| Student | 8/17    | 9/17   | 47.1% |

### By Dataset
| Dataset | Teacher | Student | Total Success |
|---------|---------|---------|---------------|
| hcrl_sa | 6/6     | 2/3     | 8/9 (88.9%)   |
| hcrl_ch | 3/3     | 2/3     | 5/6 (83.3%)   |
| set_01  | 3/3     | 2/3     | 5/6 (83.3%)   |
| set_02  | 3/3     | 1/3     | 4/6 (66.7%)   |
| set_03  | 1/2     | 1/3     | 2/5 (40.0%)   |
| set_04  | 3/3     | 1/3     | 4/6 (66.7%)   |

## Key Insights

1. **Teacher models are reliable** (95.2% success) - only set_03 GAT failed
2. **Student models struggle** (47.1% success) - especially DQN fusion
3. **VGAE is rock solid** (100% success) - all autoencoder jobs completed
4. **DQN fusion is the weakest link** (38.5% success) - needs investigation

## Root Cause of Initial Error

I counted **submitted jobs** rather than **completed jobs**. The proper method is:
1. Parse SLURM .out files for completion markers
2. Check exit codes
3. Verify model artifacts exist
4. Cross-reference with error logs

## Files to Reference

- [job_results.json](job_results.json) - Raw parsed data
- [EXPERIMENT_RESULTS_ANALYSIS.md](EXPERIMENT_RESULTS_ANALYSIS.md) - Full analysis
- [scripts/parse_job_results.py](scripts/parse_job_results.py) - Parser implementation
- [scripts/pipeline_summary.py](scripts/pipeline_summary.py) - Pipeline aggregation
