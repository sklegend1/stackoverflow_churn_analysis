import torch
import sys

print("=" * 60)
print("CUDA DIAGNOSTICS")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    device = torch.device('cuda')
    print(f"\nDevice set to: {device}")
    
    # Test tensor creation
    x = torch.randn(3, 3).to(device)
    print(f"\nTest tensor on GPU: {x.device}")
    print("SUCCESS: GPU is working!")
    
else:
    print("\nCUDA NOT AVAILABLE")
    print("\nPossible causes:")
    print("1. PyTorch installed without CUDA support (CPU-only version)")
    print("2. NVIDIA drivers not installed or outdated")
    print("3. CUDA toolkit version mismatch")
    
    print("\n" + "=" * 60)
    print("SOLUTION:")
    print("=" * 60)
    
    print("\nStep 1: Check NVIDIA driver")
    print("Run in CMD: nvidia-smi")
    print("You should see your RTX 3050 listed")
    
    print("\nStep 2: Uninstall current PyTorch")
    print("pip uninstall torch torchvision torchaudio")
    
    print("\nStep 3: Install PyTorch with CUDA 11.8 (recommended for RTX 3050)")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\nOR with CUDA 12.1:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("\nStep 4: Verify installation")
    print("python -c \"import torch; print(torch.cuda.is_available())\"")
    
    print("\n" + "=" * 60)