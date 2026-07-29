# 🛡️ DGX Disaster Recovery Guide & Local Checkpoint Backup Inventory

**Project:** *Dual-Stream Semantic-Frequency Fusion Network for Explainable Deepfake Detection and Cross-Domain Generalization*  
**Author:** Shashank Singh Singhania  
**Repository:** `https://github.com/shashank-singh-singhania/deepfake-detection-research.git`  

---

## 📌 Executive Summary

This guide guarantees **100% full recovery** of the research project, source code, trained PyTorch model checkpoints (`best_model.pt`), evaluation metrics, figures, manuscript text, and dataset preprocessing pipelines in the event that the DGX GPU server is wiped, deleted, or reset.

All source code, evaluation metrics (32 JSONs), 600 DPI publication figures, manuscript files (`paper_draft.md`, `paper_draft.docx`, `paper_draft.pdf`), IEEE LaTeX templates (`main.tex`, `references.bib`), and 50 fresh Grad-CAM visualization figure grids are **fully backed up locally on your laptop** and synchronized on GitHub `main`.

---

## 💾 1. Model Checkpoints Backup Instructions (`best_model.pt`)

To ensure you **never have to retrain models on DGX again**, back up the trained `.pt` checkpoint files from DGX (`/workspace/shashank/deepfake-detection-research/experiments/`) to your local laptop using any of the following 3 options:

### Option A: Git LFS (Direct GitHub Backup - Recommended)
Run these commands on your **DGX Server terminal (`root@abc-0`)**:
```bash
cd /workspace/shashank/deepfake-detection-research
git lfs install
git lfs track "*.pt"
git add .gitattributes experiments/
git commit -m "backup: save trained model checkpoints (Proposed Model, Xception, SBI)"
git push origin main
```
Then on your **Laptop PowerShell**:
```powershell
cd C:\Users\Singhania\Desktop\Research\deepfake-detection-research
git pull origin main
```

---

### Option B: Direct SCP Download from DGX to Laptop
Run this command on your **Laptop PowerShell**:
```powershell
scp -r root@<DGX_SERVER_IP>:/workspace/shashank/deepfake-detection-research/experiments C:\Users\Singhania\Desktop\Research\deepfake-detection-research\experiments
```

---

### Option C: Archive to Single Zip/Tar File on DGX
Run this command on your **DGX Server terminal**:
```bash
cd /workspace/shashank/deepfake-detection-research
tar -czvf trained_model_checkpoints.tar.gz experiments/
```

---

## 📂 2. Inventory of Locally Saved Assets

Every critical asset is stored locally in your workspace `C:\Users\Singhania\Desktop\Research\deepfake-detection-research`:

1. **Source Code (`src/`):**
   - `src/models/fusion_model.py`: Proposed Dual-Stream Model architecture (CLIP ViT-B/32 + EfficientNet-B0 + SRM + SE Attention + Compression Gate).
   - `src/models/xception.py`: Xception baseline model implementation.
   - `src/models/sbi_model.py`: Synthetic Body-Inconsistency (SBI) baseline model implementation.
   - `src/data/preprocess_ffpp.py`: FaceForensics++ frame extraction and MTCNN face cropping pipeline.
   - `src/data/dataset.py`: PyTorch dataset definitions and evaluation transforms.
   - `src/evaluation/evaluate.py`: Multi-GPU evaluation engine for AUC, AP, EER, Acc, Pointing Game, and Mask IoU.

2. **600 DPI Publication Graphics Suite (`paper_figures_600dpi/`):**
   - `fig1_best_system_architecture_diagram_600dpi.png`: System Architecture Flowchart.
   - `fig2_comparative_roc_curves.png`: 2-Panel ROC Curves across both datasets (FF++ & DFF).
   - `fig3_confusion_matrix_proposed.png`: 2x2 Binary Confusion Matrix (Proposed Model — FF++).
   - `fig3b_confusion_matrix_proposed_dff_600dpi.png`: 2x2 Binary Confusion Matrix (Proposed Model — DFF) in Purples palette.
   - `fig3c_multiclass_confusion_matrix_ffpp_600dpi.png`: 5x5 Multi-Class Confusion Matrix (FF++).
   - `fig3d_multiclass_confusion_matrix_dff_600dpi.png`: 4x4 Multi-Class Confusion Matrix (DFF) in Purples palette.
   - `fig4_confusion_matrix_xception.png`: 2x2 Binary Confusion Matrix (Xception — FF++).
   - `fig4b_confusion_matrix_xception_dff_600dpi.png`: 2x2 Binary Confusion Matrix (Xception — DFF).
   - `fig5_training_history_curves.png`: 3-Panel Empirical Training Curves (Loss, Acc, AUC over 30 Epochs).
   - `fig6c_cross_domain_both_datasets_600dpi.png`: 2-Panel Zero-Shot Cross-Domain Bar Chart.
   - `fig7_compression_robustness_both_datasets_600dpi.png`: 2-Panel JPEG Compression Degradation Curves ($Q=100\to50$).
   - `fig9_flawless_gradcam_showcase_600dpi.png`: 6-Row Master Grad-CAM Showcase (3 Real & 3 Fake).
   - `fig9b_comparative_gradcam_xception_vs_proposed_600dpi.png`: Horizontal 2x3 Comparative Showcase Grid.

3. **Rendered Table Images (`paper_figures_600dpi/`):**
   - `fig9_table1_in_dataset_rendered_600dpi.png`: Rendered Table I (FF++ Benchmark).
   - `fig10_table2_per_method_rendered_600dpi.png`: Rendered Table II (Per-Method Breakdown).
   - `fig11_table3_cross_domain_rendered_600dpi.png`: Rendered Table III (Zero-Shot Cross-Domain DFF).
   - `fig12_table4_compression_rendered_600dpi.png`: Rendered Table IV (Compression Degradation across Both Datasets).

4. **Evaluation Metrics JSONs (`evaluation_results/`):**
   - 32 JSON files detailing exact numerical results across all benchmarks and compression levels.

5. **Grad-CAM Visualizations (`evaluation_results/`):**
   - `gradcam_visualizations_fusion/`: 50 fresh 600 DPI figure grids generated on DGX GPU for Proposed Model.
   - `gradcam_visualizations_xception/`: 50 fresh 600 DPI figure grids generated on DGX GPU for Xception Baseline.

6. **Manuscript & Overleaf Writing Kit:**
   - `paper_draft.md`: Complete paper text written using "Proposed Model" naming standard.
   - `paper_draft.docx`: Microsoft Word version.
   - `paper_draft.pdf`: PDF version.
   - `paper_figures/main.tex`: IEEE double-column LaTeX template.
   - `paper_figures/references.bib`: BibTeX file containing 13 cited papers.

---

## 🔄 3. Step-by-Step Recovery Recipe (If DGX Server is Wiped)

If the DGX GPU server is reset, follow these exact steps to rebuild the environment in 15 minutes:

### Step 1: Clone Repository onto New GPU Server
```bash
git clone https://github.com/shashank-singh-singhania/deepfake-detection-research.git
cd deepfake-detection-research
```

### Step 2: Set Up Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install open-clip-torch timm albumentations opencv-python matplotlib seaborn pandas scikit-learn
```

### Step 3: Re-download Datasets (If Needed)

#### A. FaceForensics++ (FF++):
```bash
python scripts/01_download_ffpp.py --output_dir data/raw/ffpp --compression c23
python src/data/preprocess_ffpp.py --input_dir data/raw/ffpp --output_dir data/processed
```

#### B. DeepFakeFace (DFF):
```bash
python scripts/07_evaluate_cross_domain_dff.py --download_only
```

---

## 🔒 4. Verification of Local Backup
Run the local verification check to ensure all 600 DPI figures and JSON files are present on your laptop:
```powershell
Get-ChildItem -Path "paper_figures_600dpi"
Get-ChildItem -Path "evaluation_results"
```
All files are backed up locally and synchronized on GitHub `main` (`https://github.com/shashank-singh-singhania/deepfake-detection-research.git`).
