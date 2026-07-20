"""
Run this FIRST in your DGX Jupyter notebook / terminal to confirm the environment
is correctly set up before touching any data.

Usage:
    python 00_verify_env.py
"""
import subprocess
import sys


def check(name, fn):
    try:
        result = fn()
        print(f"[OK]   {name}: {result}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return False


def check_torch():
    import torch
    assert torch.cuda.is_available(), "CUDA not available to torch"
    dev = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc = torch.cuda.get_device_capability(0)
    return f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU={dev}, VRAM~{vram_gb:.1f}GB, compute_cap={cc}"


def check_nvidia_smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                           "--format=csv,noheader"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def check_amp():
    import torch
    x = torch.randn(4, 3, 224, 224, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = x * 2
    assert y.dtype == torch.float16
    return "mixed precision autocast OK"


def check_pkg(pkg_name, import_name=None):
    import importlib
    mod = importlib.import_module(import_name or pkg_name)
    ver = getattr(mod, "__version__", "unknown")
    return f"{pkg_name} {ver}"


def check_disk():
    import shutil
    total, used, free = shutil.disk_usage("/")
    return f"free={free/1e9:.1f}GB / total={total/1e9:.1f}GB"


def check_opencv_video():
    import cv2
    return f"opencv {cv2.__version__} (VideoCapture backend available: {cv2.getBuildInformation().count('FFMPEG') > 0})"


if __name__ == "__main__":
    print("=" * 70)
    print("DGX / A100 Environment Check for Deepfake Detection Project")
    print("=" * 70)

    all_ok = True
    all_ok &= check("nvidia-smi", check_nvidia_smi)
    all_ok &= check("torch + CUDA", check_torch)
    all_ok &= check("mixed precision (AMP)", check_amp)
    all_ok &= check("disk space", check_disk)

    print("-" * 70)
    print("Third-party packages (install missing ones with pip, see requirements.txt):")
    for pkg, imp in [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("timm", "timm"),
        ("open_clip_torch", "open_clip"),
        ("facenet-pytorch", "facenet_pytorch"),
        ("albumentations", "albumentations"),
        ("opencv-python", "cv2"),
        ("decord", "decord"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("wandb", "wandb"),
    ]:
        check(pkg, lambda p=pkg, i=imp: check_pkg(p, i))

    check("opencv video backend", check_opencv_video)

    print("=" * 70)
    print("PASS: core GPU/CUDA/AMP checks OK" if all_ok else "FAIL: fix the errors above before proceeding")
    print("=" * 70)
