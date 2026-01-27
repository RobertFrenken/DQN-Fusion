#!/usr/bin/env python3
"""
GPU Monitoring CSV Analyzer

Parses nvidia-smi output from SLURM jobs and produces:
1. Memory/utilization statistics
2. Leak detection (memory growth analysis)
3. Bottleneck classification (compute vs memory vs I/O bound)
4. Visualization plots

Usage:
    python analyze_gpu_monitor.py gpu_monitor_43977157.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def analyze_gpu_monitoring(csv_file):
    """Analyze GPU monitoring CSV from SLURM job."""

    # Read CSV (nvidia-smi output has inconsistent spacing)
    df = pd.read_csv(csv_file, skipinitialspace=True)

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"\n{'='*70}")
    print(f"GPU Monitoring Analysis: {csv_file}")
    print(f"{'='*70}\n")

    # Summary statistics
    print("📊 MEMORY STATISTICS:")
    peak_mem = df['memory.used [MiB]'].max()
    avg_mem = df['memory.used [MiB]'].mean()
    total_mem = df['memory.total [MiB]'].iloc[0]

    print(f"  Peak Memory Used:    {peak_mem:,.0f} MiB ({peak_mem/1024:.2f} GB)")
    print(f"  Average Memory Used: {avg_mem:,.0f} MiB ({avg_mem/1024:.2f} GB)")
    print(f"  GPU Total:           {total_mem:,.0f} MiB ({total_mem/1024:.2f} GB)")
    print(f"  Peak Utilization:    {peak_mem/total_mem*100:.1f}%")

    print("\n⚡ COMPUTE STATISTICS:")
    peak_gpu_util = df['utilization.gpu [%]'].max()
    avg_gpu_util = df['utilization.gpu [%]'].mean()
    peak_mem_util = df['utilization.memory [%]'].max()
    avg_mem_util = df['utilization.memory [%]'].mean()

    print(f"  Peak GPU Util:       {peak_gpu_util:.1f}%")
    print(f"  Average GPU Util:    {avg_gpu_util:.1f}%")
    print(f"  Peak Memory Util:    {peak_mem_util:.1f}%")
    print(f"  Average Memory Util: {avg_mem_util:.1f}%")

    # Detect memory leaks
    print("\n🔍 MEMORY LEAK DETECTION:")
    df['mem_diff'] = df['memory.used [MiB]'].diff()
    mem_growth_rate = df['mem_diff'].rolling(10).mean()
    peak_growth = mem_growth_rate.max()

    if peak_growth > 50:  # >50 MiB/step growth
        print(f"  ⚠️  Potential memory leak detected!")
        print(f"     Peak growth rate: {peak_growth:.1f} MiB/step")
        print(f"     Check logs for:")
        print(f"       - Validation loop without @torch.no_grad()")
        print(f"       - Gradient accumulation without .zero_grad()")
        print(f"       - Checkpoint saving during training")
    else:
        print(f"  ✅ No memory leak detected")
        print(f"     Memory growth rate: {peak_growth:.1f} MiB/step (healthy)")

    # Detect utilization bottleneck
    print("\n📈 BOTTLENECK ANALYSIS:")
    low_gpu_util_pct = (df['utilization.gpu [%]'] < 50).sum() / len(df) * 100
    high_mem_util_pct = (df['utilization.memory [%]'] > 80).sum() / len(df) * 100
    mid_gpu_util_pct = ((df['utilization.gpu [%]'] >= 50) & (df['utilization.gpu [%]'] < 80)).sum() / len(df) * 100

    if low_gpu_util_pct > 30:
        print(f"  💾 DATA LOADING BOUND")
        print(f"     {low_gpu_util_pct:.0f}% of time GPU < 50% utilization")
        print(f"     Recommendation:")
        print(f"       - Increase num_workers in DataLoader")
        print(f"       - Use pin_memory=True")
        print(f"       - Use non_blocking=True in .to(device)")
    elif high_mem_util_pct > 50:
        print(f"  🧠 MEMORY BOUND")
        print(f"     {high_mem_util_pct:.0f}% of time Memory > 80% utilization")
        print(f"     Recommendation:")
        print(f"       - Reduce batch_size")
        print(f"       - Enable gradient_checkpointing")
        print(f"       - Reduce safety_factor in batch_size_config")
    else:
        print(f"  ⚙️  COMPUTE BOUND (balanced utilization)")
        print(f"     GPU: {avg_gpu_util:.0f}% avg, Memory: {avg_mem_util:.0f}% avg")
        print(f"     Recommendation: Good tuning! Consider increasing batch_size")

    # Duration
    duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    print(f"\n⏱️  TRAINING DURATION: {duration/60:.1f} minutes ({duration:.0f} seconds)")

    print("\n" + "="*70 + "\n")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(f'GPU Monitoring Analysis: {Path(csv_file).name}', fontsize=14, fontweight='bold')

    # Memory usage
    axes[0].plot(df['timestamp'], df['memory.used [MiB]']/1024, linewidth=2.5, label='GPU Memory Used', color='#1f77b4')
    axes[0].axhline(total_mem/1024, color='#d62728', linestyle='--', label='GPU Total Capacity', linewidth=2)
    axes[0].fill_between(df['timestamp'], 0, df['memory.used [MiB]']/1024, alpha=0.15, color='#1f77b4')
    axes[0].set_ylabel('GPU Memory (GB)', fontsize=12, fontweight='bold')
    axes[0].set_title('GPU Memory Usage Over Time', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, total_mem/1024 * 1.05)

    # GPU and Memory utilization
    axes[1].plot(df['timestamp'], df['utilization.gpu [%]'], linewidth=2.5, label='GPU Utilization', color='#2ca02c')
    axes[1].plot(df['timestamp'], df['utilization.memory [%]'], linewidth=2.5, label='Memory Utilization', color='#ff7f0e')
    axes[1].set_ylabel('Utilization (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('GPU and Memory Utilization', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(70, color='green', linestyle=':', alpha=0.5, label='Ideal Range (70-85%)')
    axes[1].axhline(85, color='green', linestyle=':', alpha=0.5)

    # Memory growth rate (derivative) - leak detection
    mem_growth_smoothed = df['mem_diff'].rolling(5).mean()
    axes[2].plot(df['timestamp'][1:], mem_growth_smoothed[1:], linewidth=2.5, color='#9467bd', label='Memory Growth Rate (5-step MA)')
    axes[2].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[2].axhline(50, color='#d62728', linestyle='--', alpha=0.5, label='Leak Threshold (50 MiB/step)')
    axes[2].set_xlabel('Time', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Memory Change (MiB/step)', fontsize=12, fontweight='bold')
    axes[2].set_title('Memory Growth Rate (Leak Detection)', fontsize=13, fontweight='bold')
    axes[2].legend(loc='upper left', fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = csv_file.replace('.csv', '_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Analysis plot saved to: {output_file}\n")

    return {
        'peak_memory_gb': peak_mem / 1024,
        'peak_gpu_util': peak_gpu_util,
        'peak_mem_util': peak_mem_util,
        'memory_leak': peak_growth > 50,
        'bottleneck_type': _classify_bottleneck(low_gpu_util_pct, high_mem_util_pct),
        'duration_minutes': duration / 60
    }


def _classify_bottleneck(low_gpu, high_mem):
    """Classify bottleneck type based on utilization patterns."""
    if low_gpu > 30:
        return 'data_loading'
    elif high_mem > 50:
        return 'memory'
    else:
        return 'compute'


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("GPU Monitoring CSV Analyzer")
        print("=" * 70)
        print("Usage: python analyze_gpu_monitor.py <csv_file>")
        print("Example: python analyze_gpu_monitor.py gpu_monitor_43977157.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not Path(csv_file).exists():
        print(f"❌ Error: File not found: {csv_file}")
        sys.exit(1)

    analyze_gpu_monitoring(csv_file)
