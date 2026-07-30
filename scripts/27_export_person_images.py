"""
Script 27 — Export exact raw Face Image of the person used in the Architecture Diagram & Showcase Figures.

Output Directory:
  paper_figures_600dpi/architecture_assets/person_images/

Usage:
    python scripts/27_export_person_images.py
"""
from pathlib import Path
import cv2

OUT_DIR = Path("paper_figures_600dpi/architecture_assets/person_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def export_person_images():
    # 1. Architecture Diagram Person Image (from sample_02)
    sample_02_path = Path("evaluation_results/gradcam_visualizations_fusion/sample_02_fake_pred_fake_100pct.png")
    if sample_02_path.exists():
        full_img = cv2.imread(str(sample_02_path))
        h, w, c = full_img.shape
        pw = w // 3
        person_face = full_img[:, :pw]
        
        p1 = OUT_DIR / "person_image_architecture_diagram.png"
        cv2.imwrite(str(p1), person_face)
        print(f"Saved Architecture Diagram Person Image: {p1}")

    # 2. Extract top showcase person images (sample_12 real, sample_10 fake, sample_05 real, sample_13 fake)
    samples = [
        ("sample_12_real_pred_real_100pct.png", "person_real_sample_12.png"),
        ("sample_10_fake_pred_fake_100pct.png", "person_fake_sample_10.png"),
        ("sample_05_real_pred_real_100pct.png", "person_real_sample_05.png"),
        ("sample_13_fake_pred_fake_100pct.png", "person_fake_sample_13.png"),
        ("sample_08_real_pred_real_100pct.png", "person_real_sample_08.png"),
        ("sample_14_fake_pred_fake_100pct.png", "person_fake_sample_14.png"),
    ]

    for filename, out_name in samples:
        src = Path("evaluation_results/gradcam_visualizations_fusion") / filename
        if src.exists():
            full_img = cv2.imread(str(src))
            h, w, c = full_img.shape
            pw = w // 3
            person_face = full_img[:, :pw]
            
            p_out = OUT_DIR / out_name
            cv2.imwrite(str(p_out), person_face)
            print(f"Saved Person Image: {p_out}")


if __name__ == "__main__":
    export_person_images()
