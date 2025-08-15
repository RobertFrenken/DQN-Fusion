import os
import sys

print("=== CUDA Runtime Check ===")
print(f"Python executable: {sys.executable}")

# Check if CUDA DLLs are in PATH
cuda_paths = [
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"
]

for path in cuda_paths:
    if os.path.exists(path):
        print(f"Found CUDA path: {path}")
        dlls = [f for f in os.listdir(path) if f.endswith('.dll') and ('cuda' in f.lower() or 'cublas' in f.lower() or 'curand' in f.lower())]
        print(f"  CUDA DLLs: {dlls[:10]}...")  # Show first 10

# Force fresh import
if 'torch' in sys.modules:
    del sys.modules['torch']

# Check PyTorch's CUDA compilation
try:
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA compiled version: {torch.version.cuda}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU device count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA tensor operations
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            y = x * 2
            print(f"✅ CUDA tensor operations working! Result: {y}")
        except Exception as e:
            print(f"❌ CUDA tensor operations failed: {e}")
    else:
        print("❌ CUDA not available")
    
except Exception as e:
    print(f"Error checking CUDA: {e}")

# Check environment variables
print(f"\nEnvironment variables:")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")