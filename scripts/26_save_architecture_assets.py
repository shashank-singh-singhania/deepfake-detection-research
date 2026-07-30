"""
Script 26 — Extract and Save Individual Standalone Architecture Assets at 600 DPI.

Saves individual high-resolution PNG image assets used in the Architecture Diagram:
  1. asset_01_input_face_crop_rgb_600dpi.png
  2. asset_02_srm_highpass_noise_residual_600dpi.png
  3. asset_03_gt_manipulation_mask_600dpi.png
  4. asset_04_predicted_forgery_heatmap_600dpi.png
  5. asset_05_overlay_gradcam_on_face_600dpi.png

Output Directory:
  paper_figures_600dpi/architecture_assets/

Usage:
    python scripts/26_save_architecture_assets.py
"""
from pathlib import Path
import numpy as np
import cv2

OUT_DIR = Path("paper_figures_600dpi/architecture_assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_and_save_assets():
    sample_path = Path("evaluation_results/gradcam_visualizations_fusion/sample_02_fake_pred_fake_100pct.png")
    if not sample_path.exists():
        sample_path = Path("evaluation_results/gradcam_visualizations_fusion/sample_04_fake_pred_fake_100pct.png")

    full_img = cv2.imread(str(sample_path))
    h, w, c = full_img.shape
    pw = w // 3

    input_crop_bgr = full_img[:, :pw]
    gt_mask_bgr = full_img[:, pw:2*pw]
    heatmap_crop_bgr = full_img[:, 2*pw:]

    # Compute actual SRM Filter Residual Map
    gray = cv2.cvtColor(input_crop_bgr, cv2.COLOR_BGR2GRAY)
    srm_kernel = np.array([
        [-1,  2, -6,  2, -1],
        [ 2, -8, 12, -8,  2],
        [-6, 12,-12, 12, -6],
        [ 2, -8, 12, -8,  2],
        [-1,  2, -6,  2, -1]
    ], dtype=np.float32) / 12.0

    srm_filtered = cv2.filter2D(gray, -1, srm_kernel)
    srm_colored_bgr = cv2.applyColorMap(srm_filtered, cv2.COLORMAP_VIRIDIS)

    # 1. Save RGB Face Crop
    p1 = OUT_DIR / "asset_01_input_face_crop_rgb_600dpi.png"
    cv2.imwrite(str(p1), input_crop_bgr)
    print(f"Saved: {p1}")

    # 2. Save SRM Noise Residual Map
    p2 = OUT_DIR / "asset_02_srm_highpass_noise_residual_600dpi.png"
    cv2.imwrite(str(p2), srm_colored_bgr)
    print(f"Saved: {p2}")

    # 3. Save GT Manipulation Mask
    p3 = OUT_DIR / "asset_03_gt_manipulation_mask_600dpi.png"
    cv2.imwrite(str(p3), gt_mask_bgr)
    print(f"Saved: {p3}")

    # 4. Save Predicted Forgery Heatmap
    p4 = OUT_DIR / "asset_04_predicted_forgery_heatmap_600dpi.png"
    cv2.imwrite(str(p4), heatmap_crop_bgr)
    print(f"Saved: {p4}")

    # 5. Create Heatmap Overlay on RGB Face
    overlay_bgr = cv2.addWeighted(input_crop_bgr, 0.5, heatmap_crop_bgr, 0.5, 0)
    p5 = OUT_DIR / "asset_05_overlay_gradcam_on_face_600dpi.png"
    cv2.imwrite(str(p5), overlay_bgr)
    print(f"Saved: {p5}")


if __name__ == "__main__":
    extract_and_save_assets()
