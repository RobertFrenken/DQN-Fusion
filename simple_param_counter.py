#!/usr/bin/env python3
"""
Simple Model Parameter Counter

Quick script to count parameters in saved models for paper documentation.
This version works with saved model files directly.

Usage:
    python simple_param_counter.py
"""

import torch
import os
from pathlib import Path

def count_parameters(model_path: str) -> dict:
    """Count parameters in a saved model file."""
    if not os.path.exists(model_path):
        return {"error": f"Model file not found: {model_path}"}
    
    try:
        # Load model state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Handle different save formats
        if 'model_state_dict' in state_dict:
            model_params = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            model_params = state_dict['state_dict']
        else:
            model_params = state_dict
        
        # Count parameters
        total_params = 0
        param_details = {}
        
        for name, param in model_params.items():
            if isinstance(param, torch.Tensor):
                param_count = param.numel()
                total_params += param_count
                param_details[name] = param_count
        
        # Estimate size in MB (32-bit floats)
        size_mb = (total_params * 4) / (1024 * 1024)
        
        return {
            "total_parameters": total_params,
            "size_mb": round(size_mb, 2),
            "details": param_details
        }
        
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

def main():
    """Analyze all saved models in the project."""
    
    print("🔢 CAN-Graph Saved Model Parameter Analysis")
    print("=" * 60)
    
    # Check saved_models directory
    saved_models_dir = Path("saved_models")
    model_archive_dir = Path("model_archive/quick_archive_20260114_1642")
    
    model_files = []
    
    # Collect all model files
    for directory in [saved_models_dir, model_archive_dir]:
        if directory.exists():
            for model_file in directory.glob("*.pth"):
                model_files.append(model_file)
    
    if not model_files:
        print("❌ No .pth model files found in saved_models/ or model_archive/")
        return
    
    results = []
    
    for model_path in sorted(model_files):
        print(f"\\n📊 Analyzing: {model_path.name}")
        print("-" * 40)
        
        analysis = count_parameters(str(model_path))
        
        if "error" in analysis:
            print(f"  ❌ {analysis['error']}")
            continue
        
        total_params = analysis["total_parameters"]
        size_mb = analysis["size_mb"]
        
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Model Size: {size_mb} MB")
        
        # Categorize model type
        model_name = model_path.name
        if "vgae" in model_name.lower():
            model_type = "VGAE"
        elif "gat" in model_name.lower():
            model_type = "GAT"
        else:
            model_type = "Unknown"
        
        # Extract dataset from filename
        dataset = "Unknown"
        for ds in ["hcrl_sa", "hcrl_ch", "set_01", "set_02", "set_03", "set_04"]:
            if ds in model_name:
                dataset = ds
                break
        
        results.append({
            "Model File": model_name,
            "Model Type": model_type,
            "Dataset": dataset,
            "Parameters": total_params,
            "Size (MB)": size_mb
        })
    
    # Print summary table
    print("\\n\\n📋 SUMMARY TABLE FOR PAPER")
    print("=" * 80)
    print(f"{'Model Type':<15} {'Dataset':<10} {'Parameters':<15} {'Size (MB)':<10}")
    print("-" * 60)
    
    for result in sorted(results, key=lambda x: (x["Model Type"], x["Dataset"])):
        print(f"{result['Model Type']:<15} {result['Dataset']:<10} {result['Parameters']:<15,} {result['Size (MB)']:<10}")
    
    # Group by model type for statistics
    print("\\n\\n📊 STATISTICS BY MODEL TYPE")
    print("=" * 40)
    
    model_types = {}
    for result in results:
        model_type = result["Model Type"]
        if model_type not in model_types:
            model_types[model_type] = []
        model_types[model_type].append(result["Parameters"])
    
    for model_type, param_counts in model_types.items():
        if param_counts:
            avg_params = sum(param_counts) / len(param_counts)
            min_params = min(param_counts)
            max_params = max(param_counts)
            
            print(f"\\n{model_type}:")
            print(f"  Count: {len(param_counts)} models")
            print(f"  Average: {avg_params:,.0f} parameters")
            print(f"  Range: {min_params:,} - {max_params:,} parameters")
    
    # Create simple LaTeX table
    print("\\n\\n📄 LATEX TABLE (copy for paper)")
    print("=" * 50)
    print("\\\\begin{table}[h]")
    print("\\\\centering")
    print("\\\\caption{CAN-Graph Model Parameter Counts}")
    print("\\\\begin{tabular}{lllr}")
    print("\\\\hline")
    print("Model Type & Dataset & Parameters & Size (MB) \\\\\\\\")
    print("\\\\hline")
    
    for result in sorted(results, key=lambda x: (x["Model Type"], x["Dataset"])):
        print(f"{result['Model Type']} & {result['Dataset']} & {result['Parameters']:,} & {result['Size (MB)']} \\\\\\\\")
    
    print("\\\\hline")
    print("\\\\end{tabular}")
    print("\\\\end{table}")
    
    print("\\n✅ Analysis complete!")

if __name__ == "__main__":
    main()