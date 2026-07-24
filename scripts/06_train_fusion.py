"""
Step 06 — Train the novel fusion model (Phase 4): CLIP semantic branch +
compression-aware frequency-forensics branch + localization head, trained
jointly on classification + mask loss.

Requires manifest.csv built WITH --extract_masks (scripts/02_run_preprocessing.py),
since the localization head needs ground-truth manipulation masks to train
against. If you preprocessed without masks, re-run scripts/02 with
--extract_masks before using this script.

Two checkpoint files are written, same convention as scripts/04 and 05:
  - best_model.pt        : weights only, for evaluation (scripts/07)
  - latest_checkpoint.pt : full training state, for --resume_from_checkpoint

Usage:
    python scripts/06_train_fusion.py \
        --manifest data/processed/manifest.csv \
        --image_size 224 \
        --batch_size 32 \
        --epochs 30 \
        --lr 1e-4 \
        --lr_scheduler cosine \
        --mask_weight 1.0 \
        --run_name fusion_v1_c23
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.optim import AdamW

from src.data.dataset import build_dataloaders  # noqa: E402
from src.data.ffpp_splits import METHODS  # noqa: E402
from src.models.fusion_model import build_fusion_model  # noqa: E402
from src.training.train_fusion import train_one_epoch_fusion, evaluate_with_localization  # noqa: E402
from src.training.checkpoint import (  # noqa: E402
    build_scheduler, step_scheduler, save_full_checkpoint, load_full_checkpoint,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clip_model_name", default="ViT-B-32")
    ap.add_argument("--clip_pretrained", default="openai", help="'openai' for real pretrained weights; use 'None' string only for debugging")
    ap.add_argument("--n_unfrozen_clip_blocks", type=int, default=2)
    ap.add_argument("--clip_proj_dim", type=int, default=256)
    ap.add_argument("--freq_feat_dim", type=int, default=128)
    ap.add_argument("--fusion_hidden_dim", type=int, default=256)
    ap.add_argument("--image_size", type=int, default=224, help="224 recommended to match CLIP's native positional embeddings")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30, help="TOTAL epochs for the run (including any already completed before a resume)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr_scheduler", default="cosine", choices=["none", "cosine", "plateau"])
    ap.add_argument("--mask_weight", type=float, default=1.0, help="Weight on the auxiliary localization/mask loss")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--early_stopping_patience", type=int, default=5)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--resume_from_checkpoint", default=None, help="Path to a latest_checkpoint.pt to resume from")
    ap.add_argument("--exclude_methods", nargs="+", default=[], choices=METHODS,
                    help="Cross-manipulation leave-one-out: exclude these method(s) from TRAIN/VAL only. "
                         "Test split always includes all methods — see scripts/07_run_evaluation.py.")
    args = ap.parse_args()

    clip_pretrained = None if args.clip_pretrained == "None" else args.clip_pretrained

    run_name = args.run_name or f"fusion_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("experiments") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Run dir: {run_dir}")

    loaders = build_dataloaders(
        args.manifest, image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, return_mask=True, balance_train=True,
        methods=[m for m in METHODS if m not in args.exclude_methods] if args.exclude_methods else None,
    )
    if args.exclude_methods:
        print(f"Cross-manipulation leave-one-out: excluding {args.exclude_methods} from train/val "
              f"(test split still includes all methods for generalization measurement)")
    for split, loader in loaders.items():
        print(f"{split}: {len(loader.dataset)} samples, {len(loader)} batches")

    model = build_fusion_model(
        clip_model_name=args.clip_model_name, clip_pretrained=clip_pretrained,
        n_unfrozen_clip_blocks=args.n_unfrozen_clip_blocks, clip_proj_dim=args.clip_proj_dim,
        freq_feat_dim=args.freq_feat_dim, fusion_hidden_dim=args.fusion_hidden_dim,
    ).to(device)
    print(f"Total params: {model.total_parameter_count():,} | "
          f"Trainable: {model.trainable_parameter_count():,} "
          f"({model.trainable_parameter_count() / model.total_parameter_count():.1%})")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=args.lr, weight_decay=args.weight_decay)
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
        train_stats = train_one_epoch_fusion(model, loaders["train"], optimizer, device, scaler=scaler,
                                              mask_weight=args.mask_weight, log_prefix=f"epoch{epoch} train")
        val_metrics = evaluate_with_localization(model, loaders["val"], device, log_prefix=f"epoch{epoch} val")
        step_scheduler(scheduler, args.lr_scheduler, val_auc=val_metrics["auc"])
        dt = time.time() - t0

        print(f"[epoch {epoch}] loss={train_stats['loss']:.4f} (cls={train_stats['cls_loss']:.4f} "
              f"mask={train_stats['mask_loss']:.4f}) | val_auc={val_metrics['auc']:.4f} "
              f"val_acc={val_metrics['acc']:.4f} val_pointing_game={val_metrics['pointing_game_acc']:.4f} "
              f"val_mask_iou={val_metrics['mask_iou']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s")
        hs = val_metrics["heatmap_stats"]
        bti = val_metrics["best_threshold_iou"]
        print(f"           heatmap: mean={hs['mean']:.4f} p90={hs['p90']:.4f} p99={hs['p99']:.4f} "
              f"max={hs['max']:.4f} frac>0.5={hs['frac_above_0.5']:.4f} | "
              f"best_iou={bti['best_iou']:.4f} @thresh={bti['best_threshold']:.2f} "
              f"(iou@0.5={bti['iou_at_0.5']:.4f})")
        history.append({"epoch": epoch, "epoch_seconds": dt, "lr": optimizer.param_groups[0]["lr"],
                         **train_stats, **{f"val_{k}": v for k, v in val_metrics.items()}})
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
    test_metrics = evaluate_with_localization(model, loaders["test"], device, log_prefix="final test")
    print(f"Final test metrics (best val checkpoint): {test_metrics}")
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
