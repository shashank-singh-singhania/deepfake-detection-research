"""
Tests for src/evaluation/evaluate.py (Phase 6) and the build_dataloaders
train/val vs test method-filtering split (used for cross-manipulation
leave-one-out training).

Run from the repo root:
    python tests/test_evaluate.py
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH = REPO_ROOT / "tests" / "_scratch_evaluate"
IMAGE_SIZE = 64


def build_synthetic_manifest_multi_method():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    rng = np.random.default_rng(4)
    rows = []
    methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    for split in ["train", "val", "test"]:
        for i in range(4):
            p = SCRATCH / f"{split}_real_{i}.jpg"
            cv2.imwrite(str(p), rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
            rows.append({"path": str(p), "label": 0, "method": "youtube", "split": split,
                         "video_id": f"r{i}", "frame_idx": 0, "mask_path": ""})
        for method in methods:
            for i in range(4):
                p = SCRATCH / f"{split}_{method}_{i}.jpg"
                cv2.imwrite(str(p), rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
                rows.append({"path": str(p), "label": 1, "method": method, "split": split,
                             "video_id": f"{method}_{i}", "frame_idx": 0, "mask_path": ""})

    manifest_path = SCRATCH / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_leave_one_out_train_val_exclude_but_test_keeps_all():
    from src.data.dataset import build_dataloaders
    manifest_path = build_synthetic_manifest_multi_method()

    methods_keep = ["Face2Face", "FaceSwap", "NeuralTextures"]  # excluding Deepfakes
    loaders = build_dataloaders(str(manifest_path), image_size=IMAGE_SIZE, batch_size=4,
                                 num_workers=0, methods=methods_keep, balance_train=True)

    train_methods = set(loaders["train"].dataset.df["method"])
    val_methods = set(loaders["val"].dataset.df["method"])
    test_methods = set(loaders["test"].dataset.df["method"])

    assert "Deepfakes" not in train_methods, f"train should exclude Deepfakes, got {train_methods}"
    assert "Deepfakes" not in val_methods, f"val should exclude Deepfakes, got {val_methods}"
    assert "Deepfakes" in test_methods, \
        f"test must ALWAYS include all methods (for leave-one-out generalization measurement), got {test_methods}"
    print(f"[PASS] leave-one-out filtering correct: train/val methods={train_methods}, "
          f"test methods={test_methods} (Deepfakes correctly held out of train/val only)")


def test_evaluate_load_and_run_xception():
    from src.models.baseline import build_baseline_model
    from src.evaluation.evaluate import load_model, run_multi_protocol_evaluation

    manifest_path = build_synthetic_manifest_multi_method()
    ckpt_path = SCRATCH / "xception_ckpt.pt"

    # build + save a real (untrained) model, to test the actual save/load round-trip
    model = build_baseline_model("xception", pretrained=False)
    torch.save(model.state_dict(), ckpt_path)

    loaded = load_model("xception", str(ckpt_path), device="cpu")
    results = run_multi_protocol_evaluation(loaded, "xception", str(manifest_path), image_size=IMAGE_SIZE,
                                             batch_size=4, device="cpu", num_workers=0)

    assert "in_dataset" in results
    assert "in_dataset_all_methods" in results["in_dataset"]
    per_method = results["in_dataset"]["per_method"]
    assert set(per_method.keys()) == {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}
    assert "explainability" not in results["in_dataset"], "xception has no localization head — must not appear"
    print(f"[PASS] xception checkpoint load + multi-protocol eval correct, "
          f"per-method breakdown covers all 4 methods: {list(per_method.keys())}")

    shutil.rmtree(SCRATCH, ignore_errors=True)


def test_evaluate_fusion_includes_explainability_and_cross_dataset():
    from src.models.fusion_model import build_fusion_model
    from src.evaluation.evaluate import load_model, run_multi_protocol_evaluation

    manifest_path = build_synthetic_manifest_multi_method()

    # give fake rows a mask_path so the localization/explainability protocol has something to evaluate
    df = pd.read_csv(manifest_path, keep_default_na=False)  # keep_default_na=False: "" stays a string, not NaN
    for idx, row in df[df["label"] == 1].iterrows():
        mask_path = SCRATCH / f"mask_{idx}.jpg"
        cv2.imwrite(str(mask_path), np.random.default_rng(idx).integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
        df.at[idx, "mask_path"] = str(mask_path)
    df.to_csv(manifest_path, index=False)

    ckpt_path = SCRATCH / "fusion_ckpt.pt"
    model_kwargs = dict(clip_model_name="ViT-B-32", clip_pretrained=None, n_unfrozen_clip_blocks=1,
                        clip_proj_dim=32, freq_feat_dim=16, fusion_hidden_dim=32)
    model = build_fusion_model(**model_kwargs)
    torch.save(model.state_dict(), ckpt_path)

    loaded = load_model("fusion", str(ckpt_path), device="cpu", **model_kwargs)
    # use the same manifest as both "main" and "cross-dataset" to test the cross_dataset code path
    # NOTE: image_size=224 here (not the module IMAGE_SIZE=64) because CLIP ViT-B-32's positional
    # embeddings are sized for 224x224 input; the stored JPEGs are still 64x64, the transform
    # pipeline resizes them to whatever image_size is requested at evaluation time.
    results = run_multi_protocol_evaluation(loaded, "fusion", str(manifest_path), image_size=224,
                                             batch_size=4, device="cpu", num_workers=0,
                                             cross_dataset_manifest=str(manifest_path))

    assert "explainability" in results["in_dataset"], "fusion checkpoint must produce explainability metrics"
    assert set(results["in_dataset"]["explainability"].keys()) == {"pointing_game_acc", "mask_iou"}
    assert "cross_dataset" in results
    assert "in_dataset_all_methods" in results["cross_dataset"]
    print(f"[PASS] fusion checkpoint load + multi-protocol eval correct, "
          f"explainability={results['in_dataset']['explainability']}, "
          f"cross_dataset protocol present")

    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    test_leave_one_out_train_val_exclude_but_test_keeps_all()
    test_evaluate_load_and_run_xception()
    test_evaluate_fusion_includes_explainability_and_cross_dataset()
    print("\nALL EVALUATION SUITE TESTS PASSED")
