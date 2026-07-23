"""
Step 05 — Train the SBI (Self-Blended Images) baseline on FF++ C23.

Unlike scripts/04 (Xception baseline), this trains ONLY on self-blended real
images (generated on the fly from train-split real frames) — no actual
manipulated videos are used during training at all, per the original SBI
protocol. Validation/test still use real FF++ forgeries, so this genuinely
measures generalization to unseen manipulation types, and is directly
comparable to the Xception baseline's val/test numbers.

Requires a facial landmark detector. Default: dlib's 68-point predictor.
    pip install dlib
    Download shape_predictor_68_face_landmarks.dat (see README.md Phase 3b-SBI
    section for the link) and pass its path via --dlib_predictor_path.

Two checkpoint files are written, same convention as scripts/04:
  - best_model.pt        : weights only, for evaluation (scripts/07)
  - latest_checkpoint.pt : full training state, for --resume_from_checkpoint

Usage:
    python scripts/05_train_sbi_baseline.py \
        --manifest data/processed/manifest.csv \
        --dlib_predictor_path /path/to/shape_predictor_68_face_landmarks.dat \
        --image_size 299 \
        --batch_size 32 \
        --epochs 30 \
        --lr 1e-4 \
        --lr_scheduler cosine \
        --run_name sbi_baseline_c23
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.optim import AdamW

from src.data.sbi_blend import DlibLandmarkDetector  # noqa: E402
from src.data.sbi_dataset import build_sbi_dataloaders  # noqa: E402
from src.models.baseline import build_baseline_model  # noqa: E402
from src.training.engine import train_one_epoch, evaluate  # noqa: E402
from src.training.checkpoint import (  # noqa: E402
    build_scheduler, step_scheduler, save_full_checkpoint, load_full_checkpoint,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dlib_predictor_path", required=True,
                    help="Path to shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--model", default="xception", choices=["xception"],
                    help="Backbone architecture — same choice as scripts/04 for a fair comparison")
    ap.add_argument("--image_size", type=int, default=299)
    ap.add_argument("--batch_size", type=int, default=32,
                    help="Effective batch size after real+blended pairing (must be even)")
    ap.add_argument("--epochs", type=int, default=30, help="TOTAL epochs for the run (including any already completed before a resume)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr_scheduler", default="cosine", choices=["none", "cosine", "plateau"])
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--early_stopping_patience", type=int, default=5)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true", help="Skip ImageNet-pretrained weights (debugging/offline only)")
    ap.add_argument("--resume_from_checkpoint", default=None, help="Path to a latest_checkpoint.pt to resume from")
    args = ap.parse_args()

    run_name = args.run_name or f"sbi_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("experiments") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Run dir: {run_dir}")

    landmark_detector = DlibLandmarkDetector(args.dlib_predictor_path)
    loaders = build_sbi_dataloaders(args.manifest, image_size=args.image_size, batch_size=args.batch_size,
                                     landmark_detector=landmark_detector, num_workers=args.num_workers)
    print(f"train (SBI, real-only source): {len(loaders['train'].dataset)} real images -> "
          f"{len(loaders['train']) * args.batch_size} pseudo samples/epoch (approx)")
    print(f"val: {len(loaders['val'].dataset)} samples | test: {len(loaders['test'].dataset)} samples "
          f"(real FF++ forgeries, for genuine generalization measurement)")

    model = build_baseline_model(args.model, pretrained=not args.no_pretrained).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args.lr_scheduler, optimizer, epochs=args.epochs)
    use_amp = (device == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"Mixed precision: {use_amp} | LR scheduler: {args.lr_scheduler}")

    start_epoch = 0
    best_val_auc = -1.0
    patience_counter = 0
    history = []

    if args.resume_from_checkpoint:
        state = load_full_checkpoint(args.resume_from_checkpoint, model, optimizer, scheduler, device)
        start_epoch = state["start_epoch"]
        best_val_auc = state["best_val_auc"]
        history = state["history"]
        patience_counter = state["patience_counter"]
        print(f"Resumed from {args.resume_from_checkpoint}: starting at epoch {start_epoch}, "
              f"best_val_auc so far={best_val_auc:.4f}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device, scaler=scaler,
                                      log_prefix=f"epoch{epoch} train(SBI)")
        val_metrics, _, _ = evaluate(model, loaders["val"], device, log_prefix=f"epoch{epoch} val")
        step_scheduler(scheduler, args.lr_scheduler, val_auc=val_metrics["auc"])
        dt = time.time() - t0

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} | "
              f"val_auc={val_metrics['auc']:.4f} val_acc={val_metrics['acc']:.4f} "
              f"val_eer={val_metrics['eer']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss, "epoch_seconds": dt,
                         "lr": optimizer.param_groups[0]["lr"],
                         **{f"val_{k}": v for k, v in val_metrics.items()}})
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            print(f"  -> new best val_auc={best_val_auc:.4f}, checkpoint saved")
        else:
            patience_counter += 1

        save_full_checkpoint(run_dir / "latest_checkpoint.pt", model, optimizer, scheduler,
                              epoch=epoch, best_val_auc=best_val_auc, history=history,
                              patience_counter=patience_counter)

        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping: no val AUC improvement for {args.early_stopping_patience} epochs.")
            break

    best_ckpt = run_dir / "best_model.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics, _, _ = evaluate(model, loaders["test"], device, log_prefix="final test")
    print(f"Final test metrics (best val checkpoint): {test_metrics}")
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
