"""
Synthetic smoke test for src/data/dataset.py — builds a tiny fake manifest.csv +
dummy JPEGs (+ dummy masks) and validates FFPPDataset / build_dataloaders end to
end: correct split filtering, correct tensor shapes, mask loading, class-balance
sampling, and DataLoader iteration.

Run from the repo root:
    python tests/test_dataset.py
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH = REPO_ROOT / "tests" / "_scratch_dataset"
IMAGE_SIZE = 64


def build_synthetic_manifest():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    rows = []
    rng = np.random.default_rng(0)
    for split, n_real, n_fake in [("train", 8, 32), ("val", 4, 16), ("test", 4, 16)]:
        for i in range(n_real):
            img_path = SCRATCH / f"{split}_real_{i}.jpg"
            cv2.imwrite(str(img_path), rng.integers(0, 255, (100, 100, 3), dtype=np.uint8))
            rows.append({"path": str(img_path), "label": 0, "method": "youtube",
                         "split": split, "video_id": f"real_{i}", "frame_idx": 0, "mask_path": ""})
        for i in range(n_fake):
            img_path = SCRATCH / f"{split}_fake_{i}.jpg"
            mask_path = SCRATCH / f"{split}_fake_mask_{i}.jpg"
            cv2.imwrite(str(img_path), rng.integers(0, 255, (100, 100, 3), dtype=np.uint8))
            cv2.imwrite(str(mask_path), rng.integers(0, 255, (100, 100), dtype=np.uint8))
            rows.append({"path": str(img_path), "label": 1, "method": "Deepfakes",
                         "split": split, "video_id": f"fake_{i}", "frame_idx": 0, "mask_path": str(mask_path)})

    manifest_path = SCRATCH / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_dataset_basic(manifest_path):
    from src.data.dataset import FFPPDataset
    ds = FFPPDataset(str(manifest_path), split="train", image_size=IMAGE_SIZE)
    assert len(ds) == 40, f"expected 8 real + 32 fake = 40 train rows, got {len(ds)}"
    sample = ds[0]
    assert sample["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE), sample["image"].shape
    assert sample["label"].item() in (0, 1)
    print("[PASS] FFPPDataset basic loading + shapes correct")


def test_dataset_mask(manifest_path):
    from src.data.dataset import FFPPDataset
    ds = FFPPDataset(str(manifest_path), split="train", image_size=IMAGE_SIZE, return_mask=True)
    # find a fake sample (has a real mask) and a real sample (should get zero mask)
    fake_idx = ds.df[ds.df["label"] == 1].index[0]
    real_idx = ds.df[ds.df["label"] == 0].index[0]
    fake_sample = ds[fake_idx]
    real_sample = ds[real_idx]
    assert fake_sample["mask"].shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert real_sample["mask"].sum().item() == 0, "real frames should have all-zero mask"
    assert fake_sample["mask"].sum().item() > 0, "fake frame should have a non-trivial mask"
    print("[PASS] mask loading correct (zero for real, populated for fake)")


def test_dataloaders(manifest_path):
    from src.data.dataset import build_dataloaders
    loaders = build_dataloaders(str(manifest_path), image_size=IMAGE_SIZE, batch_size=8,
                                 num_workers=0, return_mask=True, balance_train=True)
    assert set(loaders.keys()) == {"train", "val", "test"}

    batch = next(iter(loaders["train"]))
    assert batch["image"].shape == (8, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["label"].shape == (8,)
    assert batch["mask"].shape == (8, IMAGE_SIZE, IMAGE_SIZE)

    # class-balance sanity check: sample a few batches, label distribution should
    # not be as skewed as the raw 8:32 (1:4) real:fake ratio in the source data
    labels_seen = []
    for i, b in enumerate(loaders["train"]):
        labels_seen.extend(b["label"].tolist())
        if i >= 10:
            break
    frac_real = sum(1 for l in labels_seen if l == 0) / len(labels_seen)
    assert frac_real > 0.25, f"expected WeightedRandomSampler to noticeably balance classes, got frac_real={frac_real:.2f}"
    print(f"[PASS] DataLoaders iterate correctly; balanced train sampling frac_real={frac_real:.2f} (raw was 0.20)")

    val_batch = next(iter(loaders["val"]))
    assert val_batch["image"].shape[0] <= 8
    print("[PASS] val/test loaders (no sampler) iterate correctly")


if __name__ == "__main__":
    manifest_path = build_synthetic_manifest()
    test_dataset_basic(manifest_path)
    test_dataset_mask(manifest_path)
    test_dataloaders(manifest_path)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print("\nALL DATASET TESTS PASSED")
