"""
Step 02 — Run the FF++ (C23) face-crop preprocessing pipeline.

Run from the repo root. Recommended: do a small test run first (few frames,
train split only) before committing to the full job.

Small test run:
    python scripts/02_run_preprocessing.py \
        --ffpp_root /data/FaceForensics++ \
        --splits_dir /data/FaceForensics++/splits \
        --output_root data/processed \
        --compression c23 \
        --split train \
        --frames_per_video 8 \
        --image_size 299

Full run (all splits, full frame count, with ground-truth masks for later
explainability evaluation):
    python scripts/02_run_preprocessing.py \
        --ffpp_root /data/FaceForensics++ \
        --splits_dir /data/FaceForensics++/splits \
        --output_root data/processed \
        --compression c23 \
        --split train val test \
        --frames_per_video 32 \
        --image_size 299 \
        --extract_masks

Safe to Ctrl-C and re-run — already-processed videos are skipped automatically.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess_ffpp import run_preprocessing  # noqa: E402
from src.data.ffpp_splits import METHODS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ffpp_root", required=True, help="Path to FaceForensics++ root directory")
    ap.add_argument("--splits_dir", required=True, help="Path to folder with train.json/val.json/test.json")
    ap.add_argument("--output_root", required=True, help="Where to write processed face crops + manifest.csv")
    ap.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--split", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--frames_per_video", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=299)
    ap.add_argument("--margin", type=float, default=0.3, help="Fractional margin added around detected face box")
    ap.add_argument("--extract_masks", action="store_true", help="Also crop+save ground-truth manipulation masks")
    args = ap.parse_args()

    run_preprocessing(
        ffpp_root=args.ffpp_root,
        splits_dir=args.splits_dir,
        output_root=args.output_root,
        compression=args.compression,
        methods=args.methods,
        splits=args.split,
        frames_per_video=args.frames_per_video,
        image_size=args.image_size,
        margin=args.margin,
        extract_masks=args.extract_masks,
    )


if __name__ == "__main__":
    main()
