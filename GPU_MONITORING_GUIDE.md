# GPU Monitoring Guide for KD-GAT Training

**Purpose**: Validate batch size tuning, detect memory leaks, and understand GPU utilization patterns

---

## Why This Matters for Your Implementation

You just implemented batch size tuning with `safety_factor`. Now you need to **verify it's working**:

### Key Questions nvidia-smi Answers:

1. **Is batch size tuning actually reducing memory?**
   - Does tuned_batch_size=105 use less VRAM than 192?
   - Is safety_factor 0.55 appropriate for hcrl_sa?

2. **Are there memory leaks during training?**
   - Does GPU memory grow each epoch (leak indicator)?
   - Or plateau after model loads (healthy)?

3. **Is your code compute-bound or memory-bound?**
   - GPU Util 80%+ with Memory 40%? → Compute-bound, increase batch size
   - GPU Util 30%+ with Memory 80%? → Memory-bound, good tuning

4. **Why did gat_set_03_curriculum fail?**
   - Check memory profile during VGAE+GAT dual training
   - Verify curriculum learning uses more memory than expected

---

## Integration into Your SLURM Scripts

### Option 1: Simple CSV Logging (Recommended for Your Use Case)

Embed in `test_run_counter_batch_size.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=test_run_counter_batch
#SBATCH --time=00:25:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:1
#SBATCH --chdir=/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT

# ... existing setup ...

echo "Python: $(which python)"
echo "=================================================================="

# START GPU MONITORING IN BACKGROUND
# Logs every 2 seconds to CSV: timestamp, GPU%, Memory%, Used(MiB), Total(MiB)
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv -l 2 > gpu_monitor_${SLURM_JOB_ID}.csv &

MONITOR_PID=$!
echo "🔍 GPU monitoring started (PID: $MONITOR_PID, logging to gpu_monitor_${SLURM_JOB_ID}.csv)"
echo ""

FROZEN_CONFIG="/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns/automotive/hcrl_sa_test_10epochs/unsupervised/vgae/student/no_distillation/autoencoder/configs/frozen_config_test.json"

echo "TEST RUN 1: Testing run counter, batch size optimization, and logging"
echo "Running: python train_with_hydra_zen.py --frozen-config $FROZEN_CONFIG"
echo ""

# RUN TRAINING
python train_with_hydra_zen.py --frozen-config "$FROZEN_CONFIG"

EXIT_CODE=$?

# STOP MONITORING
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo ""
echo "=================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TEST RUN 1 COMPLETED SUCCESSFULLY"
    echo "GPU monitoring saved to: gpu_monitor_${SLURM_JOB_ID}.csv"
else
    echo "❌ TEST RUN 1 FAILED (exit code: $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "=================================================================="

exit $EXIT_CODE
```

**Output**: `gpu_monitor_43977157.csv` with columns:
```
timestamp,utilization.gpu [%],utilization.memory [%],memory.used [MiB],memory.total [MiB]
2026-01-27 15:10:00.000, 0, 5, 512, 16384
2026-01-27 15:10:02.000, 5, 15, 2048, 16384
2026-01-27 15:10:04.000, 85, 65, 10650, 16384
...
```

---

## Analyzing GPU Monitoring CSV

### Python Script to Parse and Visualize

**File**: `analyze_gpu_monitor.py`

```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def analyze_gpu_monitoring(csv_file):
    """Analyze GPU monitoring CSV from SLURM job."""

    # Read CSV
    df = pd.read_csv(csv_file, skipinitialspace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Clean column names (nvidia-smi adds extra spaces)
    df.columns = [col.strip() for col in df.columns]

    print(f"\n{'='*70}")
    print(f"GPU Monitoring Analysis: {csv_file}")
    print(f"{'='*70}\n")

    # Summary statistics
    print("📊 MEMORY STATISTICS:")
    print(f"  Peak Memory Used:    {df['memory.used [MiB]'].max():,.0f} MiB "
          f"({df['memory.used [MiB]'].max()/1024:.2f} GB)")
    print(f"  Average Memory Used: {df['memory.used [MiB]'].mean():,.0f} MiB "
          f"({df['memory.used [MiB]'].mean()/1024:.2f} GB)")
    print(f"  GPU Total:           {df['memory.total [MiB]'].iloc[0]:,.0f} MiB "
          f"({df['memory.total [MiB]'].iloc[0]/1024:.2f} GB)")
    print(f"  Peak Utilization:    {df['memory.used [MiB]'].max()/df['memory.total [MiB]'].iloc[0]*100:.1f}%")

    print("\n⚡ COMPUTE STATISTICS:")
    print(f"  Peak GPU Util:       {df['utilization.gpu [%]'].max():.1f}%")
    print(f"  Average GPU Util:    {df['utilization.gpu [%]'].mean():.1f}%")
    print(f"  Peak Memory Util:    {df['utilization.memory [%]'].max():.1f}%")
    print(f"  Average Memory Util: {df['utilization.memory [%]'].mean():.1f}%")

    # Detect memory leaks
    print("\n🔍 MEMORY LEAK DETECTION:")
    df['mem_diff'] = df['memory.used [MiB]'].diff()
    mem_growth_rate = df['mem_diff'].rolling(10).mean()
    peak_growth = mem_growth_rate.max()

    if peak_growth > 50:  # >50 MiB/step growth
        print(f"  ⚠️  Potential memory leak detected!")
        print(f"     Peak growth rate: {peak_growth:.1f} MiB/step")
        print(f"     Check logs for validation loop or checkpoint saving issues")
    else:
        print(f"  ✅ No memory leak detected (growth rate: {peak_growth:.1f} MiB/step)")

    # Detect utilization bottleneck
    print("\n📈 BOTTLENECK ANALYSIS:")
    low_gpu_util = (df['utilization.gpu [%]'] < 50).sum() / len(df) * 100
    high_mem_util = (df['utilization.memory [%]'] > 80).sum() / len(df) * 100

    if low_gpu_util > 30:
        print(f"  💾 DATA LOADING BOUND ({low_gpu_util:.0f}% of time GPU < 50%)")
        print(f"     Recommendation: Increase num_workers, use pin_memory=True")
    elif high_mem_util > 50:
        print(f"  🧠 MEMORY BOUND ({high_mem_util:.0f}% of time Memory > 80%)")
        print(f"     Recommendation: Reduce batch_size or enable gradient checkpointing")
    else:
        print(f"  ⚙️  COMPUTE BOUND (balanced utilization)")
        print(f"     Recommendation: Good tuning! Can potentially increase batch_size")

    # Duration
    duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    print(f"\n⏱️  TRAINING DURATION: {duration/60:.1f} minutes")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Memory usage
    axes[0].plot(df['timestamp'], df['memory.used [MiB]']/1024, linewidth=2, label='GPU Memory Used', color='blue')
    axes[0].axhline(df['memory.total [MiB]'].iloc[0]/1024, color='red', linestyle='--', label='GPU Total', linewidth=2)
    axes[0].fill_between(df['timestamp'], 0, df['memory.used [MiB]']/1024, alpha=0.2, color='blue')
    axes[0].set_ylabel('GPU Memory (GB)', fontsize=11, fontweight='bold')
    axes[0].set_title('GPU Memory Usage Over Time', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # GPU and Memory utilization
    axes[1].plot(df['timestamp'], df['utilization.gpu [%]'], linewidth=2, label='GPU Util', color='green')
    axes[1].plot(df['timestamp'], df['utilization.memory [%]'], linewidth=2, label='Memory Util', color='orange')
    axes[1].set_ylabel('Utilization (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('GPU and Memory Utilization', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)

    # Memory growth rate (derivative)
    axes[2].plot(df['timestamp'][1:], df['mem_diff'][1:].rolling(5).mean(), linewidth=2, color='red', label='Memory Growth Rate')
    axes[2].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_xlabel('Time', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Memory Change (MiB/step)', fontsize=11, fontweight='bold')
    axes[2].set_title('Memory Growth Rate (Leak Detection)', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = csv_file.replace('.csv', '_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_gpu_monitor.py <csv_file>")
        print("Example: python analyze_gpu_monitor.py gpu_monitor_43977157.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not Path(csv_file).exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    analyze_gpu_monitoring(csv_file)
```

**Usage**:
```bash
python analyze_gpu_monitor.py gpu_monitor_43977157.csv
```

**Output Example**:
```
======================================================================
GPU Monitoring Analysis: gpu_monitor_43977157.csv
======================================================================

📊 MEMORY STATISTICS:
  Peak Memory Used:    10,650 MiB (10.40 GB)
  Average Memory Used: 8,200 MiB (8.01 GB)
  GPU Total:           16,384 MiB (16.00 GB)
  Peak Utilization:    65.0%

⚡ COMPUTE STATISTICS:
  Peak GPU Util:       87.5%
  Average GPU Util:    72.3%
  Peak Memory Util:    65.0%
  Average Memory Util: 50.1%

🔍 MEMORY LEAK DETECTION:
  ✅ No memory leak detected (growth rate: 1.2 MiB/step)

📈 BOTTLENECK ANALYSIS:
  ⚙️  COMPUTE BOUND (balanced utilization)
     Recommendation: Good tuning! Can potentially increase batch_size
```

---

## What to Look For in Your Test Job

### Expected Results for hcrl_sa_test (10 epochs, batch_size=105):

| Metric | Expected | Status |
|--------|----------|--------|
| Peak Memory | ~10-12 GB | Within 15.77 GB limit ✅ |
| Memory Growth Rate | <2 MiB/step | No leak ✅ |
| GPU Util | 70-85% | Good compute utilization ✅ |
| Memory Util | 60-75% | Balanced ✅ |

### If Memory Grows Each Epoch:
```
Epoch 0: Memory = 8,500 MiB
Epoch 1: Memory = 8,650 MiB  ← Growing
Epoch 2: Memory = 8,800 MiB  ← Leak!
```

**Fix**: Check for:
1. Gradient accumulation without `.zero_grad()`
2. Validation loop without `@torch.no_grad()`
3. Checkpoint saving during training

---

## Batch Size Validation Workflow

After test job completes, you'll have:

1. **Frozen config says**: `tuned_batch_size=105, safety_factor=0.55`
2. **GPU monitor shows**: Peak memory = 10,500 MiB
3. **You verify**:
   - ✅ Memory peak is ~64% of GPU (good safety margin)
   - ✅ No memory leaks (steady or slight decrease after warmup)
   - ✅ GPU utilization 70%+ (not data-starved)

If next run uses same `tuned_batch_size=105` but VRAM climbs to 15,000 MiB → safety_factor too aggressive, reduce to 0.45.

---

## Extended Monitoring for Production Runs

For long-running jobs (400 epochs), add process-level tracking:

```bash
# Monitor GPU processes in background
(
  echo "timestamp,pid,memory_used_mb" > gpu_procs_${SLURM_JOB_ID}.csv
  while true; do
    nvidia-smi --query-compute-apps=timestamp,pid,used_memory \
      --format=csv >> gpu_procs_${SLURM_JOB_ID}.csv 2>/dev/null || true
    sleep 10
  done
) &

PROC_MONITOR_PID=$!

# ... training ...

kill $PROC_MONITOR_PID
```

This helps identify if memory is held by multiple processes (multi-GPU issues).

---

## Integration with Your Batch Size Feedback Loop

```
RUN 1: Tune → Memory = 10,500 MiB ✅ Save tuned_batch_size=105
         ↓ GPU monitor shows healthy profile

RUN 2: Use tuned_batch_size=105 → Memory = 10,480 MiB ✅ Consistent!
         ↓ Different random seed, same batch size

RUN 3: Use tuned_batch_size=105 → Memory = 10,510 MiB ✅ Still good!
```

This validates your safety_factor is appropriate and reproducible.

---

**Next Step**: After test job 43977157 completes, run:
```bash
python analyze_gpu_monitor.py gpu_monitor_43977157.csv
```

Then check if batch size tuning is working as expected!
