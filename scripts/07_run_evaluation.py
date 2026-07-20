"""
Step 07 — Multi-protocol evaluation of a trained checkpoint.

Runs the checkpoint (from scripts/04, 05, or 06) through the evaluation
protocols in src/evaluation/evaluate.py and writes one consolidated JSON:
  - in_dataset: full test-split AUC/ACC/EER/AP, plus per-method breakdown
  - cross_dataset (only if --cross_dataset_manifest given): same, on a
    completely different manifest.csv, zero-shot
  - explainability (fusion architecture only): pointing-game accuracy, mask IoU

Usage (Xception or SBI checkpoint — same architecture, both use "xception"):
    python scripts/07_run_evaluation.py \
        --architecture xception \
        --checkpoint experiments/xception_baseline_c23/best_model.pt \
        --manifest data/processed/manifest.csv \
        --image_size 299

Usage (fusion checkpoint — pass the SAME model hyperparameters used to train it,
found in experiments/<run_name>/config.json):
    python scripts/07_run_evaluation.py \
        --architecture fusion \
        --checkpoint experiments/fusion_v1_c23/best_model.pt \
        --manifest data/processed/manifest.csv \
        --image_size 224 \
        --clip_model_name ViT-B-32 --clip_proj_dim 256 --freq_feat_dim 128 --fusion_hidden_dim 256

Add cross-dataset zero-shot evaluation (once such a manifest exists):
    ... --cross_dataset_manifest data/processed_celebdf/manifest.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.evaluation.evaluate import load_model, run_multi_protocol_evaluation  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--architecture", required=True, choices=["xception", "fusion"])
    ap.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image_size", type=int, default=299, help="299 for xception, 224 for fusion")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--cross_dataset_manifest", default=None,
                    help="Optional: path to a different manifest.csv for zero-shot cross-dataset evaluation")
    ap.add_argument("--cross_dataset_split", default="test")
    ap.add_argument("--output", default=None, help="Where to write the results JSON (default: alongside the checkpoint)")

    # fusion-only architecture args — must match what the checkpoint was trained with
    ap.add_argument("--clip_model_name", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="openai")
    ap.add_argument("--n_unfrozen_clip_blocks", type=int, default=2)
    ap.add_argument("--clip_proj_dim", type=int, default=256)
    ap.add_argument("--freq_feat_dim", type=int, default=128)
    ap.add_argument("--fusion_hidden_dim", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Architecture: {args.architecture} | Checkpoint: {args.checkpoint}")

    model_kwargs = {}
    if args.architecture == "fusion":
        clip_pretrained = None if args.clip_pretrained == "None" else args.clip_pretrained
        model_kwargs = dict(
            clip_model_name=args.clip_model_name, clip_pretrained=clip_pretrained,
            n_unfrozen_clip_blocks=args.n_unfrozen_clip_blocks, clip_proj_dim=args.clip_proj_dim,
            freq_feat_dim=args.freq_feat_dim, fusion_hidden_dim=args.fusion_hidden_dim,
        )

    model = load_model(args.architecture, args.checkpoint, device, **model_kwargs)

    results = run_multi_protocol_evaluation(
        model, args.architecture, args.manifest, image_size=args.image_size,
        batch_size=args.batch_size, device=device, num_workers=args.num_workers,
        cross_dataset_manifest=args.cross_dataset_manifest, cross_dataset_split=args.cross_dataset_split,
    )

    print("\n=== In-dataset (test split) ===")
    print(f"All methods combined: {results['in_dataset']['in_dataset_all_methods']}")
    print("Per-method breakdown:")
    for method, m in results["in_dataset"]["per_method"].items():
        print(f"  {method}: {m}")
    if "explainability" in results["in_dataset"]:
        print(f"Explainability: {results['in_dataset']['explainability']}")

    if "cross_dataset" in results:
        print(f"\n=== Cross-dataset zero-shot ({args.cross_dataset_manifest}) ===")
        print(f"All methods combined: {results['cross_dataset']['in_dataset_all_methods']}")
        for method, m in results["cross_dataset"]["per_method"].items():
            print(f"  {method}: {m}")

    output_path = Path(args.output) if args.output else Path(args.checkpoint).parent / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
