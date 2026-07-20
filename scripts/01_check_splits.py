"""
Step 01 — Sanity-check that the official FF++ train/val/test splits resolve
correctly against YOUR local copy of the dataset, before running any heavy
preprocessing.

Run from the repo root:
    python scripts/01_check_splits.py \
        --ffpp_root /data/FaceForensics++ \
        --splits_dir /data/FaceForensics++/splits \
        --compression c23
"""
import argparse
import sys
from pathlib import Path

# Make the repo root importable as `src.*` regardless of where this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ffpp_splits import build_video_list  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ffpp_root", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    args = ap.parse_args()

    for split in ["train", "val", "test"]:
        items = build_video_list(args.ffpp_root, args.splits_dir, split, args.compression)
        n_real = sum(1 for i in items if i.label == 0)
        n_fake = sum(1 for i in items if i.label == 1)
        print(f"{split}: {len(items)} videos total  (real={n_real}, fake={n_fake})")


if __name__ == "__main__":
    main()
