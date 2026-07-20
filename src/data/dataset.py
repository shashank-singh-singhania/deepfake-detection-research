"""
PyTorch Dataset + DataLoader built on top of the `manifest.csv` produced by
scripts/02_run_preprocessing.py.

Handles:
  - train/val/test split filtering directly from the manifest (no re-scanning disk)
  - ImageNet-style normalization + train-time augmentation (albumentations)
  - JPEG re-compression augmentation at train time, since FF++ C23 is already
    compressed and the model should be robust to further real-world re-encoding
  - optional loading of the aligned ground-truth manipulation mask (for fake
    frames only) — used later by the explainability/localization head
  - class imbalance handling via a WeightedRandomSampler (real:fake is 1:4 in
    frame count because FF++ has 4 manipulation methods per real video)

Usage:
    from src.data.dataset import build_dataloaders
    loaders = build_dataloaders("data/processed/manifest.csv", image_size=299, batch_size=32)
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_albumentations():
    if not _HAS_ALBUMENTATIONS:
        raise ImportError(
            "albumentations is required for src.data.dataset — "
            "install with `pip install -r requirements.txt`."
        )


def build_train_transform(image_size: int):
    _require_albumentations()
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ImageCompression(quality_range=(60, 100), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.ISONoise(p=1.0),
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


def build_eval_transform(image_size: int):
    _require_albumentations()
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})


class FFPPDataset(Dataset):
    """
    Reads rows straight from manifest.csv (no filesystem re-scan).

    Columns expected (written by src/data/preprocess_ffpp.py):
        path, label, method, split, video_id, frame_idx, mask_path
    """

    def __init__(self, manifest_csv: str, split: str, image_size: int = 299,
                 transform=None, return_mask: bool = False, methods: list = None):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if methods is not None:
            keep = self.df["method"].isin(list(methods) + ["youtube"])
            self.df = self.df[keep].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {manifest_csv} "
                              f"(methods filter={methods}). Check the manifest was built correctly.")

        self.image_size = image_size
        self.return_mask = return_mask
        self.transform = transform or (
            build_train_transform(image_size) if split == "train" else build_eval_transform(image_size)
        )

    def __len__(self):
        return len(self.df)

    def class_counts(self) -> dict:
        return self.df["label"].value_counts().to_dict()

    def sample_weights(self) -> np.ndarray:
        """Inverse-frequency weights per sample, for WeightedRandomSampler (balances real vs fake)."""
        counts = self.df["label"].value_counts()
        weight_per_class = {lbl: 1.0 / cnt for lbl, cnt in counts.items()}
        return self.df["label"].map(weight_per_class).values.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["path"])
        if img is None:
            raise FileNotFoundError(f"Could not read image at {row['path']} (manifest row {idx})")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = None
        if self.return_mask:
            mask_path = row.get("mask_path", "")
            if isinstance(mask_path, str) and mask_path and Path(mask_path).exists():
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)

        if mask is not None:
            transformed = self.transform(image=img, mask=mask)
            mask_t = transformed["mask"].float() / 255.0
        else:
            transformed = self.transform(image=img)
            mask_t = torch.zeros(self.image_size, self.image_size)

        sample = {
            "image": transformed["image"],
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "method": row["method"],
            "video_id": row["video_id"],
            "frame_idx": int(row["frame_idx"]),
        }
        if self.return_mask:
            sample["mask"] = mask_t
        return sample


def build_dataloaders(manifest_csv: str, image_size: int = 299, batch_size: int = 32,
                       num_workers: int = 8, return_mask: bool = False,
                       balance_train: bool = True, methods: list = None,
                       test_methods="all") -> dict:
    """
    Returns {"train": DataLoader, "val": DataLoader, "test": DataLoader}.
    Train loader uses a WeightedRandomSampler (balance_train=True) to counter
    the 1:4 real:fake frame-count imbalance instead of relying on shuffle=True.

    `methods` filters which manipulation methods appear in the TRAIN and VAL
    splits (None = all methods). Use this for cross-manipulation leave-one-out
    training: pass the method list with one excluded.

    `test_methods` filters the TEST split independently. Default "all" means
    the test split is NEVER restricted by `methods` — this is deliberate: a
    leave-one-out experiment needs the excluded method still present at test
    time to measure whether the model generalizes to it. Pass an explicit
    list only if you specifically want to restrict the test set too.
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        if split == "test":
            split_methods = None if test_methods == "all" else test_methods
        else:
            split_methods = methods
        ds = FFPPDataset(manifest_csv, split=split, image_size=image_size,
                         return_mask=return_mask, methods=split_methods)
        if split == "train" and balance_train:
            weights = ds.sample_weights()
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            loaders[split] = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                                         num_workers=num_workers, pin_memory=True, drop_last=True)
        else:
            loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=True)
    return loaders
