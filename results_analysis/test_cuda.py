import torch

print("✅ CUDA disponibile:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🎯 Dispositivo CUDA:", torch.cuda.get_device_name(0))
else:
    print("⚠️  Stai usando solo la CPU")
