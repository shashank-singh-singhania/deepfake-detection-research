"""
Step 03 — Quick sanity check of the Dataset/DataLoader against your REAL
manifest.csv (after scripts/02_run_preprocessing.py has finished).

Prints class counts per split, pulls one batch from each loader, checks
tensor shapes, and reports the balanced-sampling class ratio actually seen
over a few train batches. Run this before writing any model/training code —
if this doesn't look right, nothing downstream will.

Run from the repo root:
    python scripts/03_check_dataloader.py --manifest data/processed/manifest.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FFPPDataset, build_dataloaders  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="Path to manifest.csv from preprocessing")
    ap.add_argument("--image_size", type=int, default=299)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--check_masks", action="store_true", help="Also verify mask loading (requires --extract_masks was used during preprocessing)")
    ap.add_argument("--n_batches_for_balance_check", type=int, default=20)
    args = ap.parse_args()

    print("=== Per-split class counts (from manifest) ===")
    for split in ["train", "val", "test"]:
        try:
            ds = FFPPDataset(args.manifest, split=split, image_size=args.image_size, return_mask=args.check_masks)
        except ValueError as e:
            print(f"[WARN] {split}: {e}")
            continue
        counts = ds.class_counts()
        n_real = counts.get(0, 0)
        n_fake = counts.get(1, 0)
        print(f"{split}: real={n_real}, fake={n_fake}, ratio real:fake = 1:{n_fake / max(n_real,1):.2f}")

    print("\n=== DataLoader sanity check ===")
    loaders = build_dataloaders(args.manifest, image_size=args.image_size, batch_size=args.batch_size,
                                 num_workers=args.num_workers, return_mask=args.check_masks, balance_train=True)

    batch = next(iter(loaders["train"]))
    print(f"train batch: image={tuple(batch['image'].shape)}, label={tuple(batch['label'].shape)}"
          + (f", mask={tuple(batch['mask'].shape)}" if args.check_masks else ""))
    print(f"image dtype={batch['image'].dtype}, min={batch['image'].min():.3f}, max={batch['image'].max():.3f}"
          " (should look roughly normalized, not raw 0-255)")

    val_batch = next(iter(loaders["val"]))
    print(f"val batch:   image={tuple(val_batch['image'].shape)}, label={tuple(val_batch['label'].shape)}")

    print(f"\nSampling {args.n_batches_for_balance_check} train batches to check class balance from WeightedRandomSampler...")
    labels_seen = []
    for i, b in enumerate(loaders["train"]):
        labels_seen.extend(b["label"].tolist())
        if i >= args.n_batches_for_balance_check:
            break
    frac_real = sum(1 for l in labels_seen if l == 0) / len(labels_seen)
    print(f"Observed real fraction over sampled batches: {frac_real:.2f} (target ~0.5; raw manifest ratio is much lower)")

    print("\nIf all shapes/dtypes above look right and frac_real is close to 0.5, "
          "the Dataset/DataLoader are ready — move on to Phase 3b (baseline reproduction).")


if __name__ == "__main__":
    main()
