#!/usr/bin/env python3
"""
Project Cleanup Script

Removes redundant directories and files, keeping the organized hierarchical structure.
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up redundant directories and temporary files."""
    
    base_dir = Path(".")
    
    # Items to remove (with confirmation)
    cleanup_items = {
        "outputs/": "Old Lightning logs (replaced by osc_jobs hierarchical structure)",
        "saved_models/": "Old flat model storage (models now in osc_jobs/)",
        "cleanup_job_dirs.py": "One-time cleanup script",
        "count_model_parameters.py": "Development parameter counting script",
        "model_architecture_configs.py": "Development configuration script", 
        "paper_param_counter.py": "Development parameter analysis script",
        "realistic_model_configs.py": "Development configuration script",
        "simple_param_counter.py": "Development parameter counting script",
    }
    
    # Optional items (keep by default)
    optional_items = {
        "datasets/cache/": "Preprocessed data cache (48M) - can regenerate",
        "model_archive/": "Archived models from Jan 14 (125M) - backup",
        "__pycache__/": "Python cache directories",
        ".venv/": "Virtual environment (if using system conda)",
    }
    
    print("🧹 CAN-Graph Project Cleanup")
    print("=" * 40)
    
    # Calculate current disk usage
    dirs_to_check = ["outputs/", "osc_jobs/", "saved_models/", "model_archive/", "datasets/cache/"]
    total_size = 0
    for dir_name in dirs_to_check:
        if Path(dir_name).exists():
            size = get_dir_size(Path(dir_name))
            print(f"{dir_name:15} {size:>10}")
            if dir_name in ["outputs/", "saved_models/"]:
                total_size += size
    
    print(f"\n💾 Total space to reclaim: ~{total_size/1024/1024:.0f}M")
    
    print("\n📋 Items to remove:")
    for item, reason in cleanup_items.items():
        if Path(item).exists():
            print(f"  ❌ {item:25} - {reason}")
        else:
            print(f"  ✅ {item:25} - Already removed")
    
    print("\n📋 Optional cleanup (keeping by default):")
    for item, reason in optional_items.items():
        if Path(item).exists():
            print(f"  🔵 {item:25} - {reason}")
    
    # Interactive cleanup
    response = input(f"\n🤔 Remove redundant files and save ~{total_size/1024/1024:.0f}M space? (y/N): ")
    
    if response.lower() == 'y':
        removed_count = 0
        for item, reason in cleanup_items.items():
            item_path = Path(item)
            if item_path.exists():
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                    print(f"🗑️  Removed directory: {item}")
                else:
                    item_path.unlink()
                    print(f"🗑️  Removed file: {item}")
                removed_count += 1
        
        print(f"\n✅ Cleanup completed! Removed {removed_count} items.")
        print("\n📁 Remaining organized structure:")
        print("  osc_jobs/           - Hierarchical job outputs (KEEP)")
        print("  model_archive/      - Archived models (KEEP)")  
        print("  datasets/cache/     - Preprocessed cache (KEEP)")
        print("  src/                - Source code (KEEP)")
        print("  conf/               - Configuration files (KEEP)")
    else:
        print("❌ Cleanup cancelled. No files removed.")

def get_dir_size(path: Path) -> int:
    """Get directory size in bytes."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    
    total = 0
    for item in path.rglob('*'):
        if item.is_file():
            total += item.stat().st_size
    return total

if __name__ == "__main__":
    cleanup_project()