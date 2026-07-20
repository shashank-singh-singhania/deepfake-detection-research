"""
Synthetic smoke test: builds a tiny fake FF++-like directory (real + 4 manipulation
methods, dummy short videos + dummy masks) and runs the split-resolution and
preprocessing logic end-to-end with a MOCKED face detector (no network/model
download needed). Validates directory conventions, split parsing, frame sampling,
cropping, mask alignment, resume behavior, and manifest writing.

Run from the repo root:
    python tests/test_pipeline_synthetic.py

This is NOT meant to run on real FF++ data — it's a correctness check for the
pipeline code itself. Safe to re-run any time you modify src/data/*.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH = REPO_ROOT / "tests" / "_scratch"
ROOT = SCRATCH / "synthetic_ffpp"
SPLITS_DIR = ROOT / "splits"
OUT_ROOT = SCRATCH / "synthetic_out"

from src.data.ffpp_splits import METHODS  # noqa: E402


def make_dummy_video(path: Path, n_frames=20, size=(320, 240), with_face_square=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10, size)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
        if with_face_square:
            cv2.rectangle(frame, (100, 60), (220, 180), (200, 180, 160), -1)
        cv2.putText(frame, str(i), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def build_synthetic_dataset():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)

    pairs = [("000", "003"), ("001", "982")]
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_DIR / "train.json", "w") as f:
        json.dump([list(p) for p in pairs], f)
    with open(SPLITS_DIR / "val.json", "w") as f:
        json.dump([], f)
    with open(SPLITS_DIR / "test.json", "w") as f:
        json.dump([], f)

    real_ids = sorted({vid for pair in pairs for vid in pair})
    for vid in real_ids:
        make_dummy_video(ROOT / "original_sequences" / "youtube" / "c23" / "videos" / f"{vid}.mp4")

    for method in METHODS:
        for a, b in pairs:
            for x, y in ((a, b), (b, a)):
                make_dummy_video(ROOT / "manipulated_sequences" / method / "c23" / "videos" / f"{x}_{y}.mp4")
                make_dummy_video(ROOT / "manipulated_sequences" / method / "masks" / "videos" / f"{x}_{y}.mp4",
                                  with_face_square=True)

    print(f"Synthetic dataset built at {ROOT}")


def test_splits():
    from src.data.ffpp_splits import build_video_list
    items = build_video_list(str(ROOT), str(SPLITS_DIR), "train", compression="c23")
    n_real = sum(1 for i in items if i.label == 0)
    n_fake = sum(1 for i in items if i.label == 1)
    print(f"Resolved {len(items)} videos -> real={n_real}, fake={n_fake}")
    assert n_real == 4, f"expected 4 unique real ids, got {n_real}"
    assert n_fake == 16, f"expected 2 pairs * 2 orderings * 4 methods = 16, got {n_fake}"
    print("[PASS] split resolution logic correct (both orderings resolved)")


class MockExtractor:
    """Stands in for FaceExtractor without needing MTCNN weights/network."""
    def __init__(self, *a, **k):
        pass

    def detect_boxes(self, bgr_frames):
        return [(100, 60, 220, 180) for _ in bgr_frames]


def test_preprocess():
    import src.data.preprocess_ffpp as pp
    pp.FaceExtractor = MockExtractor  # monkeypatch to avoid network/model download

    from src.data.ffpp_splits import build_video_list
    items = build_video_list(str(ROOT), str(SPLITS_DIR), "train", compression="c23", include_masks=True)

    extractor = MockExtractor()
    manifest_rows = []
    for item in items:
        pp.process_video(item, extractor, OUT_ROOT, frames_per_video=5, image_size=112,
                          margin=0.2, extract_masks=True, manifest_rows=manifest_rows)

    n_jpgs = len(list(OUT_ROOT.rglob("frame_*.jpg")))
    print(f"Wrote {len(manifest_rows)} manifest rows, {n_jpgs} jpg files on disk")
    assert len(manifest_rows) > 0
    assert n_jpgs > 0

    mask_rows = [r for r in manifest_rows if r["mask_path"]]
    assert len(mask_rows) > 0, "expected some mask crops for fake videos"
    assert Path(mask_rows[0]["mask_path"]).exists()
    print("[PASS] preprocessing + mask alignment produced expected files")

    manifest_rows2 = []
    for item in items:
        pp.process_video(item, extractor, OUT_ROOT, frames_per_video=5, image_size=112,
                          margin=0.2, extract_masks=True, manifest_rows=manifest_rows2)
    assert len(manifest_rows2) == len(manifest_rows), "resume path should recover same number of rows"
    print("[PASS] resume/idempotency check correct")


if __name__ == "__main__":
    build_synthetic_dataset()
    test_splits()
    test_preprocess()
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print("\nALL SYNTHETIC PIPELINE TESTS PASSED")
