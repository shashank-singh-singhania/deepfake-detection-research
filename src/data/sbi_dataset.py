"""
SBI training dataset: wraps the REAL-only rows of manifest.csv (label == 0)
and, for each one, returns both the original image (label 0) and a freshly
generated self-blended pseudo-fake (label 1) — see src/data/sbi_blend.py for
the blending algorithm.

Following the original SBI paper's protocol: training uses ONLY self-blended
real images, never actual FF++ manipulated videos. Validation/test still use
the normal FFPPDataset (src/data/dataset.py) with real forgeries, so the
evaluation genuinely measures generalization to unseen manipulation types.
"""
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.dataset import build_eval_transform, build_train_transform
from src.data.sbi_blend import LandmarkDetector, generate_sbi_sample


class SBIDataset(Dataset):
    """
    Each __getitem__ call returns a dict with a real/fake PAIR:
        {"img_a": tensor, "label_a": 0, "img_b": tensor, "label_b": 1}
    Use `sbi_collate_fn` with the DataLoader to flatten pairs into a normal
    batch of individual (image, label) samples.

    If landmark detection fails on a given image (rare, e.g. extreme pose),
    falls back to returning the real image twice (both label 0) rather than
    fabricating an incorrect fake label.
    """

    def __init__(self, manifest_csv: str, split: str, image_size: int,
                 landmark_detector: LandmarkDetector, transform=None, seed: Optional[int] = None):
        df = pd.read_csv(manifest_csv)
        self.df = df[(df["split"] == split) & (df["label"] == 0)].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No real (label=0) rows found for split='{split}' in {manifest_csv}")

        self.image_size = image_size
        self.landmark_detector = landmark_detector
        self.transform = transform or (
            build_train_transform(image_size) if split == "train" else build_eval_transform(image_size)
        )
        self._rng = np.random.default_rng(seed)
        self._n_landmark_failures = 0  # tracked for diagnostics; check after an epoch if unexpectedly high

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["path"])
        if img is None:
            raise FileNotFoundError(f"Could not read image at {row['path']} (manifest row {idx})")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        landmarks = self.landmark_detector.get_landmarks(img)
        if landmarks is None:
            self._n_landmark_failures += 1
            blended = img.copy()
            label_b = 0  # no fake label without a valid blend
        else:
            blended = generate_sbi_sample(img, landmarks, rng=self._rng)
            label_b = 1

        img_a = self.transform(image=img)["image"]
        img_b = self.transform(image=blended)["image"]

        return {
            "img_a": img_a, "label_a": 0,
            "img_b": img_b, "label_b": label_b,
        }


def sbi_collate_fn(batch: list) -> dict:
    """Flattens (img_a,label_a,img_b,label_b) pairs into a normal batch dict
    matching the same shape/keys used by FFPPDataset's collation, so the
    training engine (src/training/engine.py) works unchanged for either."""
    images, labels = [], []
    for item in batch:
        images.append(item["img_a"])
        labels.append(item["label_a"])
        images.append(item["img_b"])
        labels.append(item["label_b"])
    return {
        "image": torch.stack(images),
        "label": torch.tensor(labels, dtype=torch.long),
    }


def build_sbi_dataloaders(manifest_csv: str, image_size: int, batch_size: int,
                           landmark_detector: LandmarkDetector, num_workers: int = 8) -> dict:
    """
    Returns {"train": sbi_loader, "val": normal_loader, "test": normal_loader}.
    `batch_size` is the TOTAL effective batch size after pairing — the
    underlying SBIDataset DataLoader uses batch_size//2 real images per step,
    each expanded to 2 samples (real + blended) by sbi_collate_fn.
    """
    from src.data.dataset import FFPPDataset  # local import avoids a cycle at module load time

    train_ds = SBIDataset(manifest_csv, split="train", image_size=image_size,
                           landmark_detector=landmark_detector)
    train_loader = DataLoader(
        train_ds, batch_size=max(1, batch_size // 2), shuffle=True,
        collate_fn=sbi_collate_fn, num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    loaders = {"train": train_loader}
    for split in ["val", "test"]:
        ds = FFPPDataset(manifest_csv, split=split, image_size=image_size, return_mask=False)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)
    return loaders
