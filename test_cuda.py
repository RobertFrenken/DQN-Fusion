import torch
import torch_geometric

print("=== CUDA Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test tensor operations
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.mm(x, x.t())
        print("✅ CUDA tensor operations working!")
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
else:
    print("❌ CUDA not available")

print(f"PyTorch Geometric version: {torch_geometric.__version__}")