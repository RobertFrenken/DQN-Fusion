#!/usr/bin/env python3
"""
Resource estimation utility for CAN-Graph training jobs.
Helps determine appropriate wall time and memory for different datasets.
"""

import argparse
from pathlib import Path
import json
from typing import Dict, Tuple


# Dataset characteristics (empirically measured)
DATASET_STATS = {
    'hcrl_sa': {
        'approx_graphs': 50000,
        'complexity': 'standard',
        'base_walltime_hours': 4,
        'base_memory_gb': 32,
    },
    'hcrl_ch': {
        'approx_graphs': 45000,
        'complexity': 'standard',
        'base_walltime_hours': 4,
        'base_memory_gb': 32,
    },
    'set_01': {
        'approx_graphs': 200000,
        'complexity': 'complex',
        'base_walltime_hours': 12,
        'base_memory_gb': 96,
    },
    'set_02': {
        'approx_graphs': 203000,
        'complexity': 'complex',
        'base_walltime_hours': 12,
        'base_memory_gb': 96,
    },
    'set_03': {
        'approx_graphs': 195000,
        'complexity': 'complex',
        'base_walltime_hours': 12,
        'base_memory_gb': 96,
    },
    'set_04': {
        'approx_graphs': 190000,
        'complexity': 'complex',
        'base_walltime_hours': 12,
        'base_memory_gb': 96,
    },
}

# Training mode multipliers
TRAINING_MULTIPLIERS = {
    'vgae_autoencoder': {
        'walltime_mult': 1.0,
        'memory_mult': 1.0,
        'description': 'VGAE autoencoder (unsupervised)',
    },
    'gat_normal': {
        'walltime_mult': 0.8,
        'memory_mult': 0.9,
        'description': 'GAT supervised training',
    },
    'gat_curriculum': {
        'walltime_mult': 1.2,
        'memory_mult': 1.1,
        'description': 'GAT with curriculum learning',
    },
    'dqn_fusion': {
        'walltime_mult': 0.6,
        'memory_mult': 1.2,
        'description': 'DQN fusion agent',
    },
    'distillation': {
        'walltime_mult': 1.0,
        'memory_mult': 1.0,
        'description': 'Knowledge distillation',
    },
}


def estimate_resources(dataset: str, training_mode: str) -> Dict[str, any]:
    """
    Estimate resource requirements for a dataset and training mode.
    
    Args:
        dataset: Dataset name (e.g., 'set_02', 'hcrl_sa')
        training_mode: Training mode (e.g., 'vgae_autoencoder', 'gat_curriculum')
    
    Returns:
        Dictionary with resource estimates
    """
    if dataset not in DATASET_STATS:
        raise ValueError(f"Unknown dataset: {dataset}. Known: {list(DATASET_STATS.keys())}")
    
    if training_mode not in TRAINING_MULTIPLIERS:
        raise ValueError(f"Unknown training mode: {training_mode}. Known: {list(TRAINING_MULTIPLIERS.keys())}")
    
    ds_stats = DATASET_STATS[dataset]
    train_mult = TRAINING_MULTIPLIERS[training_mode]
    
    # Calculate estimates
    walltime_hours = ds_stats['base_walltime_hours'] * train_mult['walltime_mult']
    memory_gb = ds_stats['base_memory_gb'] * train_mult['memory_mult']
    
    # Format as SLURM time string
    walltime_str = f"{int(walltime_hours):02d}:00:00"
    memory_str = f"{int(memory_gb)}G"
    
    return {
        'dataset': dataset,
        'training_mode': training_mode,
        'training_description': train_mult['description'],
        'approx_graphs': ds_stats['approx_graphs'],
        'complexity': ds_stats['complexity'],
        'estimated_walltime': walltime_str,
        'estimated_walltime_hours': walltime_hours,
        'estimated_memory': memory_str,
        'estimated_memory_gb': int(memory_gb),
        'recommended_command': (
            f"python oscjobmanager.py submit autoencoder_{dataset} "
            f"--walltime {walltime_str} --memory {memory_str}"
        ),
    }


def print_estimate(estimate: Dict):
    """Pretty print resource estimate."""
    print(f"\n{'='*70}")
    print(f"📊 Resource Estimate for {estimate['dataset']} - {estimate['training_mode']}")
    print(f"{'='*70}")
    print(f"Training Type: {estimate['training_description']}")
    print(f"Dataset Size:  ~{estimate['approx_graphs']:,} graphs ({estimate['complexity']})")
    print(f"\n⏰ Recommended Wall Time: {estimate['estimated_walltime']} ({estimate['estimated_walltime_hours']:.1f} hours)")
    print(f"💾 Recommended Memory:    {estimate['estimated_memory']} ({estimate['estimated_memory_gb']} GB)")
    print(f"\n🚀 Suggested Command:")
    print(f"   {estimate['recommended_command']}")
    print(f"{'='*70}\n")


def compare_datasets(training_mode: str):
    """Compare resource requirements across all datasets for a training mode."""
    print(f"\n{'='*70}")
    print(f"📊 Resource Comparison for {training_mode}")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Graphs':<12} {'Wall Time':<12} {'Memory':<10} {'Complexity':<10}")
    print(f"{'-'*70}")
    
    for dataset in sorted(DATASET_STATS.keys()):
        est = estimate_resources(dataset, training_mode)
        print(
            f"{dataset:<12} "
            f"{est['approx_graphs']:<12,} "
            f"{est['estimated_walltime']:<12} "
            f"{est['estimated_memory']:<10} "
            f"{est['complexity']:<10}"
        )
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Estimate resource requirements for CAN-Graph training jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate resources for set_02 VGAE training
  python scripts/estimate_resources.py --dataset set_02 --training vgae_autoencoder
  
  # Compare all datasets for curriculum training
  python scripts/estimate_resources.py --compare --training gat_curriculum
  
  # Get JSON output for automation
  python scripts/estimate_resources.py --dataset hcrl_sa --training gat_normal --json
        """
    )
    
    parser.add_argument(
        '--dataset',
        choices=list(DATASET_STATS.keys()),
        help='Dataset to estimate resources for'
    )
    parser.add_argument(
        '--training',
        choices=list(TRAINING_MULTIPLIERS.keys()),
        help='Training mode'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all datasets for a training mode'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.training:
            parser.error("--compare requires --training")
        compare_datasets(args.training)
    elif args.dataset and args.training:
        estimate = estimate_resources(args.dataset, args.training)
        if args.json:
            print(json.dumps(estimate, indent=2))
        else:
            print_estimate(estimate)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
