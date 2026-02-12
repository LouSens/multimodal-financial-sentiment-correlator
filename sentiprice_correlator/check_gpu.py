import torch

print("=" * 40)
print("  GPU VERIFICATION")
print("=" * 40)
print(f"  CUDA available : {torch.cuda.is_available()}")
print(f"  PyTorch        : {torch.__version__}")
print(f"  CUDA version   : {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  VRAM           : {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute cap    : {props.major}.{props.minor}")
else:
    print("  ⚠️  No CUDA GPU detected!")
print("=" * 40)
