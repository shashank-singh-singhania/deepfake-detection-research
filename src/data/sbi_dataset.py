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
        self._n_landmark_failures = 0
        # CAVEAT: this counter is only meaningful when the DataLoader uses
        # num_workers=0. With num_workers>0, PyTorch forks a separate copy of
        # this dataset object into each worker PROCESS — each worker increments
        # its OWN copy of this counter, which is never synced back to the main
        # process and is discarded when workers respawn each epoch. Checking
        # `dataset._n_landmark_failures` from the main process after a
        # multi-worker training run will silently under-report (often reads
        # as 0 or near-0) regardless of the true failure rate. Use
        # estimate_landmark_failure_rate() below for a reliable measurement
        # instead — it deliberately runs single-process.

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


def estimate_landmark_failure_rate(manifest_csv: str, split: str, landmark_detector: LandmarkDetector,
                                    sample_size: Optional[int] = 500, seed: int = 0) -> dict:
    """
    Reliably measures how often landmark detection fails on the real (label=0)
    training images — i.e. how often SBIDataset silently falls back to a
    real/real pair instead of a real/blended pair (reducing effective fake
    training signal). Deliberately single-process (no DataLoader/workers) so
    the count is trustworthy, unlike SBIDataset._n_landmark_failures under
    multi-worker training (see caveat on that attribute).

    Run this once as a quick preflight check before a long training run —
    it's a plain Python loop over images, not itself something to run every
    epoch. `sample_size=None` checks every real image in the split; a few
    hundred is usually enough to estimate the rate.

    Returns {"n_checked": int, "n_failed": int, "failure_rate": float}.
    """
    df = pd.read_csv(manifest_csv)
    real_df = df[(df["split"] == split) & (df["label"] == 0)].reset_index(drop=True)
    if len(real_df) == 0:
        raise ValueError(f"No real (label=0) rows found for split='{split}' in {manifest_csv}")

    if sample_size is not None and sample_size < len(real_df):
        real_df = real_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    n_failed = 0
    for _, row in real_df.iterrows():
        img = cv2.imread(row["path"])
        if img is None:
            continue  # unreadable file, not a landmark failure — skip rather than miscount
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if landmark_detector.get_landmarks(img) is None:
            n_failed += 1

    n_checked = len(real_df)
    return {
        "n_checked": n_checked,
        "n_failed": n_failed,
        "failure_rate": n_failed / n_checked if n_checked > 0 else float("nan"),
    }
