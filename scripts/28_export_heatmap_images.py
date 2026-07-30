"""
Script 28 — Export standalone Forgery Heatmap & Overlay Images for all showcase samples.

Output Directory:
  paper_figures_600dpi/architecture_assets/heatmap_images/

Usage:
    python scripts/28_export_heatmap_images.py
"""
from pathlib import Path
import cv2

OUT_DIR = Path("paper_figures_600dpi/architecture_assets/heatmap_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def export_heatmap_images():
    samples = [
        ("sample_02_fake_pred_fake_100pct.png", "heatmap_architecture_diagram.png", "overlay_architecture_diagram.png"),
        ("sample_12_real_pred_real_100pct.png", "heatmap_real_sample_12.png", "overlay_real_sample_12.png"),
        ("sample_10_fake_pred_fake_100pct.png", "heatmap_fake_sample_10.png", "overlay_fake_sample_10.png"),
        ("sample_05_real_pred_real_100pct.png", "heatmap_real_sample_05.png", "overlay_real_sample_05.png"),
        ("sample_13_fake_pred_fake_100pct.png", "heatmap_fake_sample_13.png", "overlay_fake_sample_13.png"),
        ("sample_08_real_pred_real_100pct.png", "heatmap_real_sample_08.png", "overlay_real_sample_08.png"),
        ("sample_14_fake_pred_fake_100pct.png", "heatmap_fake_sample_14.png", "overlay_fake_sample_14.png"),
    ]

    for filename, out_heatmap_name, out_overlay_name in samples:
        src = Path("evaluation_results/gradcam_visualizations_fusion") / filename
        if src.exists():
            full_img = cv2.imread(str(src))
            h, w, c = full_img.shape
            pw = w // 3

            person_bgr = full_img[:, :pw]
            heatmap_bgr = full_img[:, 2*pw:]
            overlay_bgr = cv2.addWeighted(person_bgr, 0.5, heatmap_bgr, 0.5, 0)

            p_h = OUT_DIR / out_heatmap_name
            p_o = OUT_DIR / out_overlay_name

            cv2.imwrite(str(p_h), heatmap_bgr)
            cv2.imwrite(str(p_o), overlay_bgr)
            print(f"Saved Heatmap: {p_h} & Overlay: {p_o}")


if __name__ == "__main__":
    export_heatmap_images()
