#!/usr/bin/env python3
"""
Model Parameter Counter for CAN-Graph Training

This script counts and reports the number of parameters for each model type
across different configurations. Perfect for paper documentation.

Usage:
    python count_model_parameters.py
    python count_model_parameters.py --dataset hcrl_sa --detailed
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.gat_model import CAN_GAT
from src.models.vgae_model import VGAE
from src.training.fusion_lightning import FusionLightningModule
from train_models import CANGraphLightningModule

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }

def get_model_size_mb(param_count: int) -> float:
    """Estimate model size in MB (assuming 32-bit floats)."""
    return (param_count * 4) / (1024 * 1024)

def create_gat_model(num_ids: int = 2048, hidden_dim: int = 256, 
                    num_heads: int = 8, num_layers: int = 3) -> CAN_GAT:
    """Create a GAT model with specified configuration."""
    return CAN_GAT(
        num_node_features=8,  # Standard CAN features
        num_ids=num_ids,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1
    )

def create_vgae_model(num_ids: int = 2048, hidden_dim: int = 256) -> VGAE:
    """Create a VGAE model with specified configuration."""
    return VGAE(
        num_node_features=8,  # Standard CAN features
        num_ids=num_ids,
        hidden_dim=hidden_dim,
        latent_dim=128  # Standard latent dimension
    )

def create_lightning_wrapper(model_type: str, num_ids: int = 2048) -> CANGraphLightningModule:
    """Create a Lightning wrapper for the model."""
    from src.config.hydra_zen_configs import CANGraphConfig
    from hydra_zen import builds
    
    # Create a basic config
    if model_type == "gat":
        config = builds(CANGraphConfig,
            model=builds(dict, type="gat", hidden_dim=256, num_heads=8, num_layers=3),
            training=builds(dict, mode="normal", learning_rate=1e-3, max_epochs=200),
            dataset=builds(dict, name="test")
        )()
    else:  # vgae
        config = builds(CANGraphConfig,
            model=builds(dict, type="vgae", hidden_dim=256, latent_dim=128),
            training=builds(dict, mode="autoencoder", learning_rate=1e-3, max_epochs=100),
            dataset=builds(dict, name="test")
        )()
    
    return CANGraphLightningModule(
        model_config=config.model,
        training_config=config.training,
        model_type=model_type,
        training_mode=config.training.mode,
        num_ids=num_ids
    )

def analyze_model_parameters():
    """Analyze parameters for all model configurations."""
    
    # Different dataset sizes (approximate number of unique CAN IDs)
    dataset_configs = {
        "hcrl_sa": 1500,
        "hcrl_ch": 1200, 
        "set_01": 2500,
        "set_02": 3000,
        "set_03": 2800,
        "set_04": 3200
    }
    
    # Model configurations to test
    model_configs = {
        "GAT (Standard)": {"type": "gat", "hidden_dim": 256, "num_heads": 8, "num_layers": 3},
        "GAT (Large)": {"type": "gat", "hidden_dim": 512, "num_heads": 8, "num_layers": 4},
        "GAT (Small)": {"type": "gat", "hidden_dim": 128, "num_heads": 4, "num_layers": 2},
        "VGAE (Standard)": {"type": "vgae", "hidden_dim": 256, "latent_dim": 128},
        "VGAE (Large)": {"type": "vgae", "hidden_dim": 512, "latent_dim": 256},
        "VGAE (Small)": {"type": "vgae", "hidden_dim": 128, "latent_dim": 64}
    }
    
    results = []
    
    print("🔢 CAN-Graph Model Parameter Analysis")
    print("=" * 60)
    
    for model_name, model_config in model_configs.items():
        print(f"\\n📊 Analyzing: {model_name}")
        print("-" * 40)
        
        for dataset_name, num_ids in dataset_configs.items():
            try:
                # Create model based on type
                if model_config["type"] == "gat":
                    model = create_gat_model(
                        num_ids=num_ids,
                        hidden_dim=model_config["hidden_dim"],
                        num_heads=model_config["num_heads"],
                        num_layers=model_config["num_layers"]
                    )
                else:  # vgae
                    model = create_vgae_model(
                        num_ids=num_ids,
                        hidden_dim=model_config["hidden_dim"]
                    )
                
                # Count parameters
                param_counts = count_parameters(model)
                size_mb = get_model_size_mb(param_counts['total'])
                
                results.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Num_IDs': num_ids,
                    'Total_Parameters': param_counts['total'],
                    'Trainable_Parameters': param_counts['trainable'],
                    'Frozen_Parameters': param_counts['frozen'],
                    'Size_MB': round(size_mb, 2),
                    'Configuration': str(model_config)
                })
                
                print(f"  {dataset_name:8s}: {param_counts['total']:,} params ({size_mb:.1f} MB)")
                
            except Exception as e:
                print(f"  {dataset_name:8s}: Error - {e}")
    
    return results

def create_parameter_tables(results: List[Dict]) -> pd.DataFrame:
    """Create formatted tables for paper documentation."""
    
    df = pd.DataFrame(results)
    
    print("\\n\\n📋 PARAMETER COUNT SUMMARY TABLE")
    print("=" * 80)
    
    # Summary table by model type
    summary = df.groupby(['Model']).agg({
        'Total_Parameters': ['min', 'max', 'mean'],
        'Size_MB': ['min', 'max', 'mean']
    }).round(0)
    
    print(summary)
    
    # Detailed table for largest dataset (set_04)
    print("\\n\\n📋 DETAILED PARAMETERS (Complex Dataset - set_04)")
    print("=" * 80)
    
    detailed = df[df['Dataset'] == 'set_04'][['Model', 'Total_Parameters', 'Trainable_Parameters', 'Size_MB']]
    detailed = detailed.copy()
    detailed['Total_Parameters'] = detailed['Total_Parameters'].apply(lambda x: f"{x:,}")
    detailed['Trainable_Parameters'] = detailed['Trainable_Parameters'].apply(lambda x: f"{x:,}")
    
    print(detailed.to_string(index=False))
    
    return df

def export_to_latex(df: pd.DataFrame, filename: str = "model_parameters.tex"):
    """Export parameter table to LaTeX format for papers."""
    
    # Create LaTeX table for paper
    latex_df = df[df['Dataset'] == 'set_04'].copy()  # Use largest dataset
    latex_df = latex_df[['Model', 'Total_Parameters', 'Size_MB']]
    
    # Format numbers
    latex_df['Total_Parameters'] = latex_df['Total_Parameters'].apply(lambda x: f"{x:,}")
    latex_df['Size_MB'] = latex_df['Size_MB'].apply(lambda x: f"{x:.1f}")
    
    # Rename columns for paper
    latex_df.columns = ['Model Architecture', 'Parameters', 'Size (MB)']
    
    # Export to LaTeX
    latex_table = latex_df.to_latex(index=False, escape=False)
    
    with open(filename, 'w') as f:
        f.write("% CAN-Graph Model Parameter Counts\\n")
        f.write("% Generated automatically\\n\\n")
        f.write(latex_table)
    
    print(f"\\n💾 LaTeX table exported to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Count CAN-Graph model parameters')
    parser.add_argument('--export-latex', action='store_true', help='Export LaTeX table')
    parser.add_argument('--detailed', action='store_true', help='Show detailed breakdown')
    
    args = parser.parse_args()
    
    try:
        # Analyze all models
        results = analyze_model_parameters()
        
        # Create summary tables
        df = create_parameter_tables(results)
        
        if args.export_latex:
            export_to_latex(df)
        
        if args.detailed:
            print("\\n\\n🔍 DETAILED BREAKDOWN BY DATASET")
            print("=" * 80)
            for _, row in df.iterrows():
                print(f"{row['Model']} on {row['Dataset']}:")
                print(f"  Total Parameters: {row['Total_Parameters']:,}")
                print(f"  Trainable: {row['Trainable_Parameters']:,}")
                print(f"  Model Size: {row['Size_MB']:.2f} MB")
                print()
        
        print("\\n✅ Parameter analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()