"""
TriConsistencyNet — Dataset

Reads from the same manifest.csv used by the main codebase.
Columns used: path, label (0=real, 1=fake), split, method.
Mask columns are intentionally ignored (TriConsistencyNet is classification-only).
"""

from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def _get_image_compression():
    try:
        return A.ImageCompression(quality_range=(60, 100), p=1.0)
    except Exception:
        return A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0)


def _train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            _get_image_compression(),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _eval_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ── Dataset class ─────────────────────────────────────────────────────────────

class FFPPDataset(Dataset):
    """
    FaceForensics++ dataset reader backed by the shared manifest.csv.

    Args:
        manifest_path : path to data/processed/manifest.csv
        split         : "train" | "val" | "test"
        image_size    : resize target (default 224)
        augment       : whether to apply training augmentations
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: int = 224,
        augment: bool = False,
    ):
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transforms = (
            _train_transforms(image_size) if augment else _eval_transforms(image_size)
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        img = Image.open(row["path"]).convert("RGB")
        img_np = np.array(img)

        out = self.transforms(image=img_np)
        image = out["image"]                    # (3, H, W) float32

        label = float(row["label"])             # 0.0 = real, 1.0 = fake

        return {
            "image":  image,
            "label":  torch.tensor(label, dtype=torch.float32),
            "method": str(row.get("method", "unknown")),
        }


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_loaders(
    manifest_path: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Return (train_loader, val_loader, test_loader)."""
    train_ds = FFPPDataset(manifest_path, "train", image_size, augment=True)
    val_ds   = FFPPDataset(manifest_path, "val",   image_size, augment=False)
    test_ds  = FFPPDataset(manifest_path, "test",  image_size, augment=False)

    kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, **kw)

    return train_loader, val_loader, test_loader
