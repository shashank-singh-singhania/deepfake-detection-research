"""
Script 08 — Index & Process DeepFakeFace (DFF) Dataset for Zero-Shot Cross-Dataset Evaluation.

Scans data/dff_raw/ and builds a clean test manifest data/processed_dff/manifest.csv:
  - wiki: label=0 (real)
  - insight: label=1 (fake, InsightFace)
  - text2img: label=1 (fake, Stable Diffusion v1.5)
  - inpainting: label=1 (fake, SD Inpainting)

Usage:
    python scripts/08_process_dff_dataset.py --max_samples_per_class 2000
"""
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw_dir", default="data/dff_raw", help="Path to unzipped DFF raw dataset")
    ap.add_argument("--output_manifest", default="data/processed_dff/manifest.csv", help="Where to save DFF test manifest")
    ap.add_argument("--max_samples_per_class", type=int, default=2500,
                    help="Limit samples per category for fast zero-shot evaluation (0 for all)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist. Run python external/download_dff.py first!")
        return

    categories = {
        "wiki": (0, "real"),
        "insight": (1, "insight"),
        "text2img": (1, "text2img"),
        "inpainting": (1, "inpainting"),
    }

    records = []
    print("=== Scanning DeepFakeFace (DFF) Dataset ===")
    for cat_name, (label, method_name) in categories.items():
        cat_dir = raw_dir / cat_name
        if not cat_dir.exists():
            print(f"Warning: {cat_dir} not found, skipping...")
            continue

        image_files = sorted(list(cat_dir.rglob("*.jpg")) + list(cat_dir.rglob("*.png")) + list(cat_dir.rglob("*.jpeg")))
        if args.max_samples_per_class > 0 and len(image_files) > args.max_samples_per_class:
            image_files = image_files[:args.max_samples_per_class]

        print(f"Found {len(image_files)} images for {cat_name} (label={label}, method={method_name})")
        for img_path in tqdm(image_files, desc=cat_name):
            records.append({
                "path": str(img_path.resolve()).replace("\\", "/"),
                "filepath": str(img_path.resolve()).replace("\\", "/"),
                "label": label,
                "split": "test",
                "method": method_name,
                "video_id": img_path.stem,
                "frame_idx": 0,
                "mask_path": "",
            })

    df = pd.DataFrame(records)
    out_path = Path(args.output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nManifest successfully created: {out_path}")
    print(f"Total DFF evaluation samples: {len(df)}")
    print(f"Class breakdown:\n{df['label'].value_counts().to_dict()}")
    print(f"Method breakdown:\n{df['method'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
