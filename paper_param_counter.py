#!/usr/bin/env python3
"""
Clean Model Parameter Counter for Paper Documentation

Analyzes CAN-Graph model parameter counts for academic paper tables.
Properly categorizes models by type and training mode.

Usage: python paper_param_counter.py
"""

import torch
import os
from pathlib import Path

def analyze_model_name(filename: str) -> dict:
    """Extract model information from filename."""
    name_lower = filename.lower()
    
    # Determine model type
    if "vgae" in name_lower:
        model_type = "VGAE"
        mode = "Autoencoder"
    elif "gat" in name_lower:
        model_type = "GAT"
        if "normal" in name_lower:
            mode = "Normal"
        elif "curriculum" in name_lower:
            mode = "Curriculum"
        elif "fusion" in name_lower:
            mode = "Fusion"
        else:
            mode = "Normal"  # Default
    elif "teacher" in name_lower:
        model_type = "GAT (Teacher)"
        mode = "Curriculum"
    elif "student" in name_lower:
        if "autoencoder" in name_lower:
            model_type = "VGAE (Student)"
            mode = "Autoencoder"
        else:
            model_type = "GAT (Student)"
            mode = "Curriculum"
    elif "autoencoder" in name_lower:
        model_type = "VGAE"
        mode = "Autoencoder"
    elif "classifier" in name_lower:
        model_type = "GAT"
        mode = "Classification"
    else:
        model_type = "Unknown"
        mode = "Unknown"
    
    # Extract dataset
    dataset = "Unknown"
    for ds in ["hcrl_sa", "hcrl_ch", "set_01", "set_02", "set_03", "set_04"]:
        if ds in name_lower:
            dataset = ds.upper().replace("_", "-")
            break
    
    return {
        "model_type": model_type,
        "mode": mode,
        "dataset": dataset,
        "filename": filename
    }

def count_parameters(model_path: str) -> dict:
    """Count parameters in a saved model file."""
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Handle different save formats
        if 'model_state_dict' in state_dict:
            model_params = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            model_params = state_dict['state_dict']
        else:
            model_params = state_dict
        
        # Count parameters
        total_params = sum(param.numel() for param in model_params.values() 
                          if isinstance(param, torch.Tensor))
        
        # Size in MB
        size_mb = (total_params * 4) / (1024 * 1024)
        
        return {
            "total_parameters": total_params,
            "size_mb": round(size_mb, 2)
        }
        
    except Exception as e:
        return {"error": str(e)}

def main():
    print("🔢 CAN-Graph Model Parameter Analysis for Paper")
    print("=" * 60)
    
    # Collect model files
    saved_models_dir = Path("saved_models")
    model_archive_dir = Path("model_archive/quick_archive_20260114_1642")
    
    all_results = []
    
    for directory in [saved_models_dir, model_archive_dir]:
        if not directory.exists():
            continue
            
        for model_file in sorted(directory.glob("*.pth")):
            # Analyze filename
            model_info = analyze_model_name(model_file.name)
            
            # Count parameters
            param_info = count_parameters(str(model_file))
            
            if "error" not in param_info:
                all_results.append({
                    **model_info,
                    **param_info
                })
    
    # Group and organize results
    model_summary = {}
    
    for result in all_results:
        key = (result["model_type"], result["mode"])
        if key not in model_summary:
            model_summary[key] = {
                "examples": [],
                "param_counts": [],
                "datasets": set()
            }
        
        model_summary[key]["examples"].append(result)
        model_summary[key]["param_counts"].append(result["total_parameters"])
        model_summary[key]["datasets"].add(result["dataset"])
    
    # Print organized summary
    print("\\n📊 MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    architecture_table = []
    
    for (model_type, mode), info in sorted(model_summary.items()):
        param_counts = info["param_counts"]
        avg_params = sum(param_counts) / len(param_counts)
        min_params = min(param_counts)
        max_params = max(param_counts)
        
        print(f"\\n{model_type} ({mode}):")
        print(f"  Models: {len(param_counts)}")
        print(f"  Datasets: {', '.join(sorted(info['datasets']))}")
        print(f"  Parameters: {avg_params:,.0f} (avg), {min_params:,}-{max_params:,} (range)")
        
        architecture_table.append({
            "Architecture": f"{model_type} ({mode})",
            "Parameters": f"{avg_params:,.0f}",
            "Range": f"{min_params:,} - {max_params:,}" if min_params != max_params else f"{min_params:,}",
            "Models": len(param_counts)
        })
    
    # Clean paper table
    print("\\n\\n📋 PAPER TABLE - MAIN ARCHITECTURES")
    print("=" * 70)
    print(f"{'Architecture':<25} {'Avg Parameters':<15} {'Model Size (MB)':<15}")
    print("-" * 70)
    
    paper_architectures = [
        ("VGAE (Autoencoder)", "VGAE", "Autoencoder"),
        ("GAT (Normal)", "GAT", "Normal"),
        ("GAT (Teacher)", "GAT (Teacher)", "Curriculum"),
        ("GAT (Student)", "GAT (Student)", "Curriculum"),
    ]
    
    paper_results = []
    
    for display_name, model_type, mode in paper_architectures:
        key = (model_type, mode)
        if key in model_summary:
            info = model_summary[key]
            avg_params = sum(info["param_counts"]) / len(info["param_counts"])
            avg_size = (avg_params * 4) / (1024 * 1024)
            
            print(f"{display_name:<25} {avg_params:<15,.0f} {avg_size:<15.1f}")
            paper_results.append((display_name, avg_params, avg_size))
    
    # LaTeX table for paper
    print("\\n\\n📄 LATEX TABLE FOR PAPER")
    print("=" * 50)
    print("\\\\begin{table}[ht]")
    print("\\\\centering")
    print("\\\\caption{CAN-Graph Model Architectures and Parameter Counts}")
    print("\\\\label{tab:model-parameters}")
    print("\\\\begin{tabular}{lrr}")
    print("\\\\toprule")
    print("Architecture & Parameters & Size (MB) \\\\\\\\")
    print("\\\\midrule")
    
    for display_name, avg_params, avg_size in paper_results:
        latex_name = display_name.replace('(', '\\textit{').replace(')', '}')
        print(f"{latex_name} & {avg_params:,.0f} & {avg_size:.1f} \\\\\\\\")
    
    print("\\\\bottomrule")
    print("\\\\end{tabular}")
    print("\\\\end{table}")
    
    # Detailed breakdown for appendix
    print("\\n\\n📋 DETAILED BREAKDOWN (for appendix)")
    print("=" * 60)
    
    for result in sorted(all_results, key=lambda x: (x["model_type"], x["dataset"])):
        print(f"{result['model_type']:<20} {result['dataset']:<8} {result['total_parameters']:>10,} params ({result['size_mb']:>4.1f} MB)")
    
    print("\\n✅ Analysis complete! Use the LaTeX table above in your paper.")

if __name__ == "__main__":
    main()