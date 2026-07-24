"""
TriConsistencyNet

TriConsistencyNet Model Evaluation Script

Author: Shashank Singh
"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataloader import get_dataloaders
from src.models.triconsistencynet import TriConsistencyNet
from src.trainer.metrics import Metrics
from src.trainer.trainer import Trainer
from src.utils.config import ConfigLoader
from src.utils.logger import project_logger


def main():
    project_logger.info("Initializing TriConsistencyNet model evaluation on the test split...")

    # 1. Setup Directories
    experiment_dir = PROJECT_ROOT / "experiments" / "triconsistency"
    checkpoint_path = experiment_dir / "checkpoints" / "best_triconsistencynet.pth"
    metrics_dir = experiment_dir / "metrics"
    visualizations_dir = experiment_dir / "visualizations"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        project_logger.error(f"Best checkpoint not found at: {checkpoint_path}")
        return

    # 2. Load Configurations
    training_config = ConfigLoader().load("training.yaml")

    # 3. Create DataLoaders
    _, _, test_loader = get_dataloaders()
    project_logger.info(f"Loaded test split with {len(test_loader.dataset)} face samples ({len(test_loader)} batches).")

    if len(test_loader.dataset) == 0:
        project_logger.error("Test dataset is empty. Ensure you preprocessed the test split.")
        return

    # 4. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_logger.info(f"Using device: {device}")

    # 5. Instantiate Model and Load Checkpoint
    model = TriConsistencyNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    project_logger.success(f"Successfully loaded best checkpoint from Epoch {checkpoint['epoch']} (Val Acc: {checkpoint['metric'] * 100:.2f}%)")

    # 6. Instantiate Trainer for Evaluation
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        optimizer=None,
        criterion=criterion,
        device=device,
        mixed_precision=training_config.training.mixed_precision,
    )

    # 7. Run Test Evaluation
    project_logger.info("Running inference on test split...")
    test_results = trainer.validate(test_loader)
    
    # 8. Calculate Metrics
    test_metrics = Metrics.calculate(
        targets=test_results["labels"],
        predictions=test_results["predictions"],
        probabilities=test_results["probabilities"],
    )

    # Print Results
    project_logger.success("=== Test Split Metrics ===")
    project_logger.info(f"Test Loss: {test_results['loss']:.4f}")
    project_logger.info(f"Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    project_logger.info(f"Precision: {test_metrics['precision'] * 100:.2f}%")
    project_logger.info(f"Recall:    {test_metrics['recall'] * 100:.2f}%")
    project_logger.info(f"F1-score:  {test_metrics['f1'] * 100:.2f}%")
    project_logger.info(f"ROC-AUC:   {test_metrics['roc_auc'] * 100:.4f}")
    project_logger.info(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # 9. Save Metrics to JSON (convert numpy array to list)
    test_metrics_json = test_metrics.copy()
    test_metrics_json["confusion_matrix"] = test_metrics_json["confusion_matrix"].tolist()
    test_metrics_json["loss"] = test_results["loss"]
    
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics_json, f, indent=4)
    project_logger.success(f"Test metrics saved to: {metrics_dir / 'test_metrics.json'}")

    # 10. Generate and Save ROC / PR Curves
    labels = test_results["labels"]
    probs = np.asarray(test_results["probabilities"])

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {test_metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve - TriConsistencyNet")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(metrics_dir / "roc_curve.png", dpi=300)
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.figure()
    plt.plot(recall, precision, color="blue", lw=2, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - TriConsistencyNet")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(metrics_dir / "pr_curve.png", dpi=300)
    plt.close()
    
    project_logger.success(f"ROC and PR curves saved to: {metrics_dir}")

    # 11. Visualize and Save Attention Maps for Sample Test Images
    project_logger.info("Generating attention map visualizations for sample test images...")
    model.eval()
    
    # Get a single batch of test images
    sample_batch = next(iter(test_loader))
    images = sample_batch["image"].to(device)
    labels_batch = sample_batch["label"].to(device)
    
    with torch.no_grad():
        _ = model(images)
        # Shape: (B, 1280, 7, 7)
        attention_maps = model.last_attention_map
        
    # Plot first 5 samples
    num_samples = min(5, images.size(0))
    for i in range(num_samples):
        # original image (denormalize from Albumentations default)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        # Scale to 0-1 range for plotting
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # average attention across channels
        attn = attention_maps[i].mean(dim=0).cpu().numpy()
        # normalize heatmap
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Original Image
        axes[0].imshow(img)
        axes[0].set_title(f"Original Face (Label: {'FAKE' if labels_batch[i].item() == 1 else 'REAL'})")
        axes[0].axis("off")
        
        # Overlay Heatmap
        axes[1].imshow(img)
        # Interpolate the heatmap to match original image size
        axes[1].imshow(attn, cmap="jet", alpha=0.5, extent=(0, 224, 224, 0))
        axes[1].set_title("FGE Attention Heatmap Overlay")
        axes[1].axis("off")
        
        plt.tight_layout()
        save_path = visualizations_dir / f"attention_sample_{i}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    project_logger.success(f"Sample attention visualizations saved under: {visualizations_dir}")


if __name__ == "__main__":
    main()
