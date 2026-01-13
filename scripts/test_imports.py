#!/usr/bin/env python3
"""
Import Test Script
Tests if all imports in the reorganized CAN-Graph project work correctly.
"""
import sys
from pathlib import Path

# Add project root to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("🧪 Testing CAN-Graph Import Structure...")

try:
    # Test core model imports
    print("1️⃣ Testing model imports...")
    from src.models import GATWithJK, GraphAutoencoderNeighborhood, EnhancedDQNFusionAgent
    from src.models.pipeline import GATPipeline
    print("   ✅ Models imported successfully")
    
    # Test config imports
    print("2️⃣ Testing config imports...")
    from src.config.fusion_config import DATASET_PATHS, FUSION_WEIGHTS
    from src.config.plotting_config import COLOR_SCHEMES, apply_publication_style
    print("   ✅ Config imported successfully")
    
    # Test utility imports  
    print("3️⃣ Testing utility imports...")
    # GPU utilities are now handled by PyTorch Lightning
    from src.utils.utils_logging import setup_gpu_optimization, log_memory_usage
    print("   ✅ Utilities imported successfully")
    
    # Test training imports
    print("4️⃣ Testing training imports...")
    from src.training.gpu_monitor import GPUMonitor
    from src.training.fusion_extractor import FusionDataExtractor  
    from src.training.trainers import PyTorchTrainer
    print("   ✅ Training modules imported successfully")
    
    # Test evaluation imports
    print("5️⃣ Testing evaluation imports...")
    from src.evaluation.evaluation import create_student_models, create_teacher_models
    print("   ✅ Evaluation imported successfully")
    
    print("\n🎉 All imports working correctly!")
    print("✅ Project reorganization successful!")
    
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("❗ Check file paths and import statements")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Unexpected Error: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("\n📋 Import Summary:")
    print("   - All core models accessible ✅")
    print("   - Configuration management working ✅") 
    print("   - Utilities properly imported ✅")
    print("   - Training pipelines ready ✅")
    print("   - Evaluation framework accessible ✅")
    print("\n🚀 Your CAN-Graph project is ready for development!")