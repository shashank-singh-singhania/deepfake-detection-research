"""
Synthetic smoke test for SBI (Self-Blended Images) — Phase 3b part 2.

Uses a MockLandmarkDetector returning a fixed oval of synthetic points (no
dlib/model file needed) to validate: mask generation shape/range, that
blending actually changes pixels within the face region, that the SBIDataset
produces correctly-paired (real, blended) samples, and that the full
build_sbi_dataloaders pipeline iterates correctly with the right shapes.

Run from the repo root:
    python tests/test_sbi.py
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH = REPO_ROOT / "tests" / "_scratch_sbi"
IMAGE_SIZE = 96


def make_synthetic_face_landmarks(w=100, h=100, n_points=68):
    """Roughly oval arrangement of points inside the image, standing in for a
    real 68-point face landmark layout — good enough to exercise convex-hull
    mask logic without needing an actual face or detector."""
    cx, cy = w / 2, h / 2
    rx, ry = w * 0.35, h * 0.4
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.stack([cx + rx * np.cos(a) for a in angles] +
                    [cy + ry * np.sin(a) for a in angles], axis=-1)
    # build properly as (n,2)
    pts = np.stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)], axis=1)
    return pts.astype(np.float32)


class MockLandmarkDetector:
    def __init__(self, always_fail=False):
        self.always_fail = always_fail

    def get_landmarks(self, rgb_image):
        if self.always_fail:
            return None
        h, w = rgb_image.shape[:2]
        return make_synthetic_face_landmarks(w, h)


def test_blend_mask_shape_and_range():
    from src.data.sbi_blend import build_blend_mask
    rng = np.random.default_rng(0)
    landmarks = make_synthetic_face_landmarks(100, 100)
    mask = build_blend_mask((100, 100, 3), landmarks, rng)
    assert mask.shape == (100, 100)
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert mask.sum() > 0, "mask should cover a non-trivial face region"
    print("[PASS] build_blend_mask shape/range correct")


def test_generate_sbi_sample_changes_pixels():
    from src.data.sbi_blend import generate_sbi_sample
    rng = np.random.default_rng(0)
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    landmarks = make_synthetic_face_landmarks(100, 100)
    blended = generate_sbi_sample(img, landmarks, rng=rng)
    assert blended.shape == img.shape
    assert blended.dtype == np.uint8
    diff = np.abs(blended.astype(int) - img.astype(int))
    assert diff.sum() > 0, "blended image should differ from the original within the face region"
    print(f"[PASS] generate_sbi_sample produces a modified image (mean abs diff={diff.mean():.2f})")


def build_synthetic_manifest():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    rows = []
    rng = np.random.default_rng(2)
    # SBI only needs REAL rows for train; val/test need both real+fake for genuine eval
    for i in range(12):
        p = SCRATCH / f"train_real_{i}.jpg"
        cv2.imwrite(str(p), rng.integers(80, 180, (100, 100, 3), dtype=np.uint8))
        rows.append({"path": str(p), "label": 0, "method": "youtube", "split": "train",
                     "video_id": f"r{i}", "frame_idx": 0, "mask_path": ""})
    for split, n_real, n_fake in [("val", 4, 16), ("test", 4, 16)]:
        for i in range(n_real):
            p = SCRATCH / f"{split}_real_{i}.jpg"
            cv2.imwrite(str(p), rng.integers(80, 180, (100, 100, 3), dtype=np.uint8))
            rows.append({"path": str(p), "label": 0, "method": "youtube", "split": split,
                         "video_id": f"r{i}", "frame_idx": 0, "mask_path": ""})
        for i in range(n_fake):
            p = SCRATCH / f"{split}_fake_{i}.jpg"
            cv2.imwrite(str(p), rng.integers(80, 180, (100, 100, 3), dtype=np.uint8))
            rows.append({"path": str(p), "label": 1, "method": "Deepfakes", "split": split,
                         "video_id": f"f{i}", "frame_idx": 0, "mask_path": ""})

    manifest_path = SCRATCH / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_sbi_dataset_pairing():
    from src.data.sbi_dataset import SBIDataset
    manifest_path = build_synthetic_manifest()
    ds = SBIDataset(str(manifest_path), split="train", image_size=IMAGE_SIZE,
                     landmark_detector=MockLandmarkDetector(), seed=0)
    assert len(ds) == 12, f"expected 12 real train rows, got {len(ds)}"

    item = ds[0]
    assert item["label_a"] == 0
    assert item["label_b"] == 1  # mock detector always succeeds
    assert item["img_a"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert item["img_b"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    print("[PASS] SBIDataset returns correctly-labeled (real, blended) pairs")


def test_sbi_dataset_landmark_failure_fallback():
    from src.data.sbi_dataset import SBIDataset
    manifest_path = build_synthetic_manifest()
    ds = SBIDataset(str(manifest_path), split="train", image_size=IMAGE_SIZE,
                     landmark_detector=MockLandmarkDetector(always_fail=True), seed=0)
    item = ds[0]
    assert item["label_b"] == 0, "landmark failure should fall back to label 0, never fabricate a fake label"
    assert ds._n_landmark_failures == 1
    print("[PASS] landmark-detection-failure fallback behaves safely (no incorrect fake label)")


def test_build_sbi_dataloaders():
    from src.data.sbi_dataset import build_sbi_dataloaders
    manifest_path = build_synthetic_manifest()
    loaders = build_sbi_dataloaders(str(manifest_path), image_size=IMAGE_SIZE, batch_size=8,
                                     landmark_detector=MockLandmarkDetector(), num_workers=0)
    assert set(loaders.keys()) == {"train", "val", "test"}

    train_batch = next(iter(loaders["train"]))
    assert train_batch["image"].shape == (8, 3, IMAGE_SIZE, IMAGE_SIZE)  # batch_size//2=4 pairs -> 8 samples
    assert train_batch["label"].shape == (8,)
    # every pair contributes exactly one real (0) and one blended (1) -> should be exactly balanced
    frac_real = (train_batch["label"] == 0).float().mean().item()
    assert abs(frac_real - 0.5) < 1e-6, f"SBI pairing should give exactly 0.5 real fraction, got {frac_real}"
    print(f"[PASS] SBI train loader shapes correct, perfectly balanced (frac_real={frac_real})")

    val_batch = next(iter(loaders["val"]))
    assert val_batch["image"].shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
    print("[PASS] SBI val/test loaders (real FF++-style manifest rows) iterate correctly")

    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    test_blend_mask_shape_and_range()
    test_generate_sbi_sample_changes_pixels()
    test_sbi_dataset_pairing()
    test_sbi_dataset_landmark_failure_fallback()
    test_build_sbi_dataloaders()
    print("\nALL SBI TESTS PASSED")
