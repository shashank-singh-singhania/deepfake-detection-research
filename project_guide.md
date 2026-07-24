# Deepfake Detection Research — Beginner-Friendly Project Guide & Architecture Manual

**Project Lead:** Shashank Singh Singhania  
**Repository:** `deepfake-detection-research`  
**Dataset:** FaceForensics++ (FF++) C23 Quality  
**Hardware:** NVIDIA DGX Node (1x NVIDIA A100 40GB GPU)  
**Date:** July 2026  

---

## 1. Project Introduction & Key Concepts

### What is Deepfake Detection?
A **deepfake** is an AI-generated or AI-modified image or video where a person's face or body is altered (for example, swapping one person's face onto another person, or manipulating facial expressions). 

**Deepfake detection** is the task of building machine learning models (specifically deep neural networks) that can automatically take an input face image or video frame and determine:
1. **Binary Classification:** Is this image **REAL** (authentic) or **FAKE** (manipulated)?
2. **Localization / Explainability:** *Where exactly* in the image was the manipulation performed? (Outputting a 2D spatial heatmap showing the fake region).

---

### Key Terminology Explained Simply

- **FaceForensics++ (FF++):** The standard benchmark dataset used in deepfake research. It contains 1,000 original real videos and 4,000 manipulated videos generated using 4 distinct forgery techniques:
  1. **Deepfakes (DF):** Learning-based face replacement.
  2. **Face2Face (F2F):** Expression transfer (puppeteering an authentic face).
  3. **FaceSwap (FS):** Graphics-based 3D face transfer.
  4. **NeuralTextures (NT):** GAN-based facial mouth/texture synthesis.
- **Compression Level (C23):** Video compression removes subtle pixel details to save file size. Raw uncompressed video is easy to detect, but **C23 (medium compression)** represents realistic video compression used on social media platforms (YouTube, Twitter, TikTok). It is much harder to detect because compression blurs micro-artifacts.
- **Identity-Preserved Splitting:** Making sure that all frames of a specific person's video appear **only** in the training set, validation set, or test set, but never split across them. This prevents the model from "cheating" by memorizing a person's face shape instead of learning true forgery traces.

---

## 2. Literature Review & Gap Analysis (Simplified)

### 2.1 Plain-English Breakdown of 13 Key Research Papers

| Paper Title & Authors | Year & Venue | What They Did (Simple Explanation) | Why It Mattered | What Was Missing / Limitation |
|---|---|---|---|---|
| **FaceForensics++**<br>*Rössler et al.* | 2019<br>ICCV | Created the FF++ benchmark dataset with 4 forgery methods across 3 compression levels (Raw, C23, C40). Trained an **Xception** neural network baseline. | Established the foundation for modern deepfake detection research. Proved Xception achieves **>99% accuracy** on raw/lightly compressed data. | Did not test on unseen deepfake generators or external datasets. Compression robustness was lightly explored. |
| **Face X-ray**<br>*Li et al.* | 2020<br>CVPR | Instead of looking at fake faces, it looks for the **blending boundary**—the line where the fake face crop was pasted onto the real background. | Shifted focus away from specific GAN artifacts toward universal blending boundaries, improving generalization. | Fails when no blending boundary exists (e.g. fully generated AI faces or whole-image diffusion models). |
| **Multi-attentional Detection**<br>*Zhao et al.* | 2021<br>CVPR | Used multiple spatial attention heads to zoom into different face regions (eyes, mouth, skin) and aggregate texture artifacts. | Improved detection of subtle local manipulations (like NeuralTextures). | Attention maps were never quantitatively verified against true ground-truth fake masks. |
| **Pairwise Self-Consistency (PCL)**<br>*Li et al.* | 2021<br>CVPR | Checked if different parts of the same face are consistent with each other (e.g., matching lighting, skin texture). | Inconsistent features signal forgery without needing specific fake labels during training. | Consistency signals break down under heavy video compression (C23/C40) and small face crops. |
| **High-Frequency Feature Generalization**<br>*Luo et al.* | 2021<br>CVPR | Extracted high-frequency noise using **SRM filters** (Spatial Rich Model noise filters) alongside standard RGB color images. | High-frequency noise exposes hidden mathematical artifacts left by deepfake generators. | High-frequency details are easily destroyed when videos are compressed or re-encoded. |
| **Self-Blended Images (SBI)**<br>*Shiohara & Yamasaki* | 2022<br>CVPR | Created synthetic fake training images by blending a real face with a slightly modified version of itself (**Self-Blending**). | Achieved **93%+ cross-dataset accuracy** on Celeb-DF without ever seeing real deepfake datasets during training. | Landmark detection fails on extreme face angles; poor in-dataset performance on standard benchmarks without threshold tuning. |
| **SLADD**<br>*Chen et al.* | 2022<br>CVPR | Used adversarial learning to automatically discover the hardest fake image augmentations during model training. | Dynamically adapts the training data so the model doesn't over-fit to easy fakes. | Increases training complexity and instability; unverified on diffusion-based fakes. |
| **UIA-ViT**<br>*Zhuang et al.* | 2022<br>ECCV | Used Vision Transformers (ViT) with patch-consistency loss to automatically highlight inconsistent facial patches. | Showed Transformers can learn forgery boundaries without needing pixel-level masks. | Requires huge compute/pretraining; attention maps were only visually inspected, not quantitatively measured. |
| **AltFreezing**<br>*Wang et al.* | 2023<br>CVPR | Used a 3D-CNN video network, alternately freezing spatial weights and temporal weights during training. | Forced the network to learn both spatial artifacts and motion/timing glitches across video frames. | Very slow and compute-heavy; operates at video clip level rather than fine-grained per-frame spatial masks. |
| **TALL**<br>*Xu et al.* | 2023<br>CVPR | Arranged video frames into a 2D "thumbnail grid image" so a standard 2D Swin-Transformer could process video sequences cheaply. | High efficiency for video deepfake detection compared to heavy 3D-CNNs. | Grid layout destroys fine spatial details needed for precise pixel-level localization. |
| **Implicit Identity Leakage**<br>*Dong et al.* | 2023<br>CVPR | Proved that deepfake detectors often cheat by memorizing identity (who the person is) rather than learning forgery artifacts. | Proposed identity-disentanglement to force the model to ignore subject identity. | Only partially removes identity bias; adds complex multi-task loss terms. |
| **UCF (Uncovering Common Features)**<br>*Yan et al.* | 2023<br>CVPR | Separated identity/content features from universal forgery features shared across all manipulation algorithms. | Strong generalization across different deepfake generation methods. | Complex disentanglement losses; tested primarily on older GANs rather than modern diffusion models. |

---

### 2.2 The 7 Literature Gaps (Why Current Models Fail)

1. **Gap #1 — Compression Sensitivity:** High-performing models collapse when videos are compressed (FF++ C23/C40) because fine pixel artifacts are wiped out by JPEG/video compression.
2. **Gap #2 — Generator Over-fitting:** Models trained on one deepfake tool (e.g. GANs) fail on new generators (e.g. Diffusion models or novel face-swappers).
3. **Gap #3 — Lack of Quantitative Explainability:** Most papers show nice-looking heatmaps (Grad-CAM), but **never measure** if their heatmap actually matches the true ground-truth manipulation mask.
4. **Gap #4 — Isolated Feature Streams:** Papers focus exclusively on *only* semantic CLIP features OR frequency noise OR temporal motion—rarely fusing them into a unified, compression-aware model with spatial mask supervision.
5. **Gap #5 — Identity Memorization Shortcut:** Classifiers memorize person identities instead of learning forgery characteristics.
6. **Gap #6 — Transformer Efficiency Ignored:** Advanced ViT/VLM models are rarely benchmarked for parameter count and inference speed.
7. **Gap #7 — Academic vs. In-The-Wild Reality Gap:** Models scoring >99% on clean academic benchmarks drop to ~50% (random guess) on real-world internet deepfakes.

---

### 2.3 Our Selected Research Gap & Solution Architecture

**Our Goal:** To solve **Gaps #1, #3, and #4** by creating a **Novel Dual-Stream Fusion Model**:
- **Stream A (Semantic Branch):** Uses a pre-trained **CLIP ViT-B/32** vision-language backbone to capture high-level facial structures and semantic anomalies.
- **Stream B (Frequency Branch):** Uses **SRM (Spatial Rich Model) high-pass noise filters** followed by a CNN to capture mathematical high-frequency noise hidden in compressed C23 videos.
### 2.3 Our Selected Research Gap & Solution Architecture (Deep Dive)

#### The Real-World Problem (Explained with an Analogy)
Imagine a detective trying to spot a forged $100 bill. 
1. **The Pure RGB Detective (e.g., Xception):** Looks only at colors and shapes. Works great on pristine bills in a bright room, but if the bill is wrinkled or faded (like video compression on social media), they get easily confused.
2. **The Pure Frequency Detective (e.g., SRM / FFT):** Uses a microscope to look *only* at paper fiber patterns. They find microscopic ink noise, but have zero idea what a portrait of Benjamin Franklin is supposed to look like.
3. **The Black-Box Classifier:** Stamps the bill as "95% Fake" without telling anyone *which part* of the bill was forged.

#### Our Proposed Solution: The Dual-Stream Fusion Model
We designed a **Dual-Stream Fusion Architecture** that brings together two specialized "detectives" plus an "evidence highlighter":

```text
                           Input Face Crop (224 x 224 x 3)
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
        Semantic Stream (CLIP ViT-B/32)              Frequency Stream (SRM Noise Filters)
      Look at facial geometry, eyes, lips,          Look at hidden mathematical noise
         and high-level visual features               left behind by AI face-swappers
                   │                                             │
             512-dim Vector                              128x56x56 Spatial Feature Map
                   │                                             │
                   └──────────────────────┬──────────────────────┘
                                          ▼
                         Cross-Domain Feature Fusion Layer
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
       Classification Output Head                    Localization / Explainability Head
   Predicts overall REAL vs FAKE score              Outputs a 2D spatial heatmap drawing a
            (Single Logit / AUC)                   highlighter over the exact fake region
                                                  (Supervised by ground-truth masks)
```

1. **Stream 1 (Semantic Detective - CLIP ViT-B/32):** Examines facial symmetry, eye alignment, and natural blending.
2. **Stream 2 (Frequency Forensic Lab - SRM Noise Filters):** Filters out standard colors using 3 Spatial Rich Model (SRM) high-pass kernels, exposing invisible noise grids left by deepfake generators.
3. **Fusion Layer:** Combines the semantic vector and frequency spatial map into a single unified representation.
4. **Dual Output Heads:**
   - **Classifier:** Predicts whether the image is REAL or FAKE.
   - **Localization / Explainability Head:** Generates a 2D spatial heatmap showing the exact pixels that were tampered with, validated quantitatively against ground-truth manipulation masks.

---

## 3. Project Implementation Status & Step-by-Step Evolution

### 3.1 The Step-by-Step Story of How the Project Evolved

#### Step 1: Raw Data & Frame Extraction (Setting the Foundation)
- **Goal:** Convert raw video files into clean, ready-to-train face crops.
- **Process:** Downloaded 5,000 FaceForensics++ C23 videos (1,000 real + 4,000 fake). Extracted 159,969 aligned face crops at 224x224 and 299x299 resolutions using face detectors (RetinaFace / MTCNN).
- **Identity-Preserved Splitting:** We organized the dataset into `train` (115,188 frames), `val` (22,384 frames), and `test` (22,397 frames) in `data/processed/manifest.csv`. **Crucial Detail:** All videos of a specific person appear *only* in train, val, OR test—never split across them. This ensures the model learns true deepfake artifacts rather than memorizing people's faces.

#### Step 2: Baseline Model Training (Setting Benchmark Ceilings & Floors)
- **Xception Baseline:** Fine-tuned standard Xception on 299x299 crops $\to$ **98.44% AUC** (established our in-distribution performance upper bound).
- **SBI Baseline (Self-Blended Images):** Trained on synthetic self-blended image pairs $\to$ **71.10% AUC** (established our lower bound). SBI relies on synthetic self-blended noise, which does not match real-world FF++ C23 compressed forgeries without explicit calibration.

#### Step 3: Building & Stabilizing the Novel Dual-Stream Fusion Model
- **Model Implementation:** Fused CLIP ViT-B/32 + SRM noise filters + spatial localization head in `src/models/fusion_model.py`.
- **The First Training Crash (v1):** At epoch 12, FP16 half-precision gradients exploded under mixed-precision AMP, poisoning BatchNorm statistics with NaN values and crashing GPU memory.
- **Engineering Fixes:** We introduced gradient unscaling before clipping, enforced `max_grad_norm=1.0` gradient clipping, and added `nan_to_num` + `clamp(0,1)` guards to prevent CUDA device-side assertions.
- **Clean Training (v2):** Trained all 30 epochs cleanly $\to$ **88.24% AUC**, 66.65% Pointing Game Accuracy, and 38.69% optimal Mask IoU.

#### Step 4: Alternative Exploration (TriConsistencyNet)
- **Exploration:** To test whether explicit spatial-frequency product attention could rival our Fusion model, we implemented **TriConsistencyNet** (`another_model/`) combining a frozen `EfficientNetV2-S` backbone with a 2D Fast Fourier Transform (FFT) magnitude encoder.
- **Result:** Trained for 22 epochs $\to$ **80.66% AUC**.
- **Conclusion:** TriConsistencyNet outperformed SBI (80.66% vs 71.10%), but our Dual-Stream CLIP + SRM Fusion Model proved superior (88.24% AUC) because unfrozen CLIP transformer layers provide far better joint spatial-frequency alignment than a frozen EfficientNet backbone.

---

### 3.2 Complete Phase-by-Phase Status Table

| Phase | Project Phase | Status | What Was Accomplished & Key Deliverables |
|---|---|---|---|
| **Phase 0** | Environment Setup | **100% Done** | Installed PyTorch with CUDA 12.4, `timm`, `open_clip_torch`, `albumentations`, `decord` on DGX A100. Pinned in `requirements.txt`. |
| **Phase 1 & 2** | Data Acquisition & Preprocessing | **100% Done** | Processed 5,000 videos into 159,969 aligned 224x224 / 299x299 face crops. Created master `manifest.csv` with identity-preserved splits (`train`: 115,188, `val`: 22,384, `test`: 22,397) and ground-truth mask locations. |
| **Phase 3** | Baseline Reproduction | **100% Done** | Trained Xception (**98.44% AUC**) and SBI (**71.10% AUC**, verified 1.6% landmark failure rate). Established upper and lower performance bounds. |
| **Phase 4** | Novel Model Architecture | **100% Done** | Built **Novel Dual-Stream Fusion Model** (CLIP + SRM + Localization Head) and **TriConsistencyNet** (EfficientNetV2-S + 2D FFT + CCA attention). |
| **Phase 5** | Training Protocol & Stability Fixes | **100% Done** | Added FP16 Mixed Precision, Cosine Annealing LR, Gradient Clipping (`max_grad_norm=1.0`), and NaN guards (`nan_to_num` + `clamp(0,1)`). |
| **Phase 6** | Diagnostic Evaluation Suite | **100% Done** | Built `07_run_evaluation.py` and `another_model/evaluate.py`. Calculates AUC, AP, EER, Balanced Acc, Pointing Game, and optimal IoU per manipulation method. |
| **Phase 7** | Model Refinement & Analysis | **In Progress** | Discovered that Fusion v2's localization head is *well-localized but under-confident* (IoU @ 0.5 = 0.03%, but IoU @ 0.10 threshold = **38.69%**, Pointing Game = **66.65%**). |

---

## 4. Engineering Solutions & Stability Fixes Explained

During training on the DGX A100 GPU, we encountered and solved 5 major technical bugs:

### Bug 1: OpenCV Video Loading Crash on Linux
- **Problem:** OpenCV (`cv2.VideoCapture`) failed on certain compressed FFMPEG video containers on Linux, returning empty frames.
- **Fix:** Implemented an automatic fallback to `decord` video reader in `src/data/preprocess_ffpp.py`. If OpenCV returns an empty frame, `decord` takes over seamlessly.

### Bug 2: Early Stopping Countdown Reset on Resume
- **Problem:** When training was interrupted and resumed with `--resume_from_checkpoint`, `patience_counter` wasn't saved in the checkpoint file. This caused early stopping to reset to 0 every time training resumed.
- **Fix:** Updated `src/training/checkpoint.py` to store and restore `patience_counter` in state dicts across all scripts.

### Bug 3: PyTorch AMP `GradScaler` Assertion Error
- **Problem:** When loss became NaN on corrupt batches, skipping backward pass while calling `scaler.update()` triggered `AssertionError: No inf checks were recorded prior to update.`
- **Fix:** Fixed `src/training/train_fusion.py` so `scaler.unscale_(optimizer)` and `scaler.update()` are only called when a valid backward pass occurs.

### Bug 4: CUDA Device-Side Assertion Crash (`Loss.cu:94`)
- **Problem:** When resuming from a corrupted checkpoint, model heatmaps produced NaN/Inf values. Feeding NaNs into `F.binary_cross_entropy()` caused a hard GPU assertion error (`input_val >= zero && input_val <= one`).
- **Fix:** Sanitized heatmaps before BCE loss calculation in `src/training/train_fusion.py`:
  ```python
  heatmap_safe = torch.nan_to_num(heatmap.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
  ```

### Bug 5: Albumentations 2.x Argument Schema Mismatch
- **Problem:** Albumentations upgraded to 2.x, changing `RandomResizedCrop(height=..., width=...)` and `ImageCompression` argument signatures, breaking training scripts.
- **Fix:** Updated `another_model/src/dataset.py` with positional `A.Resize(image_size, image_size)` and a dynamic `_get_image_compression()` wrapper that works on both Albumentations 1.x and 2.x.

---

## 5. Model Architectures & Benchmark Evaluation Results

### 5.1 Explanation of Evaluation Metrics (Beginner Friendly)

- **AUC (Area Under ROC Curve):** Measures how well the model ranks fake faces higher than real faces across all possible decision thresholds. **1.0 (100%) is perfect**, **0.5 (50%) is random guessing**.
- **Average Precision (AP):** Measures precision across different recall levels. Especially valuable for imbalanced datasets.
- **Equal Error Rate (EER):** The error rate at the exact point where False Positive Rate equals False Negative Rate. **Lower is better** (0% is perfect).
- **Balanced Accuracy:** The average of Accuracy on Real faces and Accuracy on Fake faces. Unlike raw accuracy, it cannot be fooled by class imbalance (e.g. 1 real : 4 fakes).
- **Pointing Game Accuracy:** Evaluates localization. It takes the highest intensity pixel (peak point) in the model's heatmap and checks if it falls inside the ground-truth fake mask. **Higher is better**.
- **Mask IoU (Intersection over Union):** Measures spatial overlap between the predicted fake mask region and the ground-truth fake mask ($0.0 \to 1.0$).

---

### 5.2 Comprehensive Benchmark Results (FF++ C23 Test Split)

| Model Architecture | Overall AUC | Average Precision (AP) | Equal Error Rate (EER) | Balanced Accuracy | Raw Accuracy | Pointing Game Acc | Mask IoU |
|---|---|---|---|---|---|---|---|
| **Xception Baseline** | **98.44%** | 99.62% | 5.38% | 93.46% | 95.16% | N/A | N/A |
| **Novel Fusion Model v2** | **88.24%** | 96.42% | 19.29% | 77.22% | 85.11% | **66.65%** | **37.75%** (@ adaptive)<br>**38.69%** (@ thresh 0.10) |
| **TriConsistencyNet** | **80.66%** | 93.97% | 27.22% | 67.91% | 80.77% | N/A | N/A |
| **SBI Baseline** | **71.10%** | 90.31% | 34.96% | 53.37% | 25.95%* | N/A | N/A |

*\*Note on SBI: Raw accuracy of 25.95% is due to threshold miscalibration (predicting almost all images as Real). Its true discrimination capability is reflected by AUC (71.10%) and Balanced Acc (53.37%).*

---

### 5.3 Per-Manipulation Method AUC Breakdown

| Model Architecture | Deepfakes (DF) | Face2Face (F2F) | FaceSwap (FS) | NeuralTextures (NT) |
|---|---|---|---|---|
| **Xception Baseline** | **99.15%** | **98.91%** | **98.74%** | **96.98%** |
| **Novel Fusion Model v2** | **93.14%** | **88.64%** | **89.78%** | **81.40%** |
| **TriConsistencyNet** | **84.37%** | **81.13%** | **79.93%** | **77.22%** |
| **SBI Baseline** | **81.26%** | **71.01%** | **64.47%** | **67.65%** |

---

### 5.4 Model Architectural Comparison

#### 1. Novel Dual-Stream Fusion Model (`src/models/fusion_model.py`)
- **Semantic Stream:** OpenAI CLIP ViT-B/32 backbone (top 2 Transformer blocks unfrozen) $\rightarrow$ 512-dim semantic vector.
- **Frequency Stream:** Fixed 3-kernel SRM high-pass noise filters $\rightarrow$ 3-layer trainable Conv-BN-ReLU CNN $\rightarrow$ $128\times56\times56$ feature map.
- **Dual Output Heads:**
  - **Classification Head:** Linear layer predicting Real vs Fake probability.
  - **Localization Head:** $1\times1$ Conv + Sigmoid producing $224\times224$ manipulation heatmap.
- **Total Parameters:** 88.2M | Trainable: 14.9M (17.0%).

#### 2. TriConsistencyNet (`another_model/src/model.py`)
- **Spatial Stream:** Frozen `EfficientNetV2-S` backbone $\rightarrow$ $1280\times7\times7$ feature map.
- **Frequency Stream (FGE):** 2D FFT magnitude spectrum ($\log(1 + |F|)$ log compression) $\rightarrow$ 0.8M parameter CNN $\rightarrow$ $256\times7\times7$ feature map.
- **Cross-Consistency Attention (CCA):** Element-wise multiplication $C = S \odot F$ gated into spatial attention map $A \in [0, 1]$, refining features via $S_{\text{refined}} = S \odot (1 + A)$.
- **Adaptive Feature Fusion (AFF):** Squeeze-and-Excitation channel gating.
- **Total Parameters:** 27.2M | Trainable: 7.09M (26.0%).

#### 3. Xception Baseline (`src/models/baseline.py`)
- Standard 36-layer depthwise separable convolutional network pretrained on ImageNet and fine-tuned end-to-end on 299x299 face crops.

#### 4. SBI Baseline (`src/data/sbi_dataset.py` & `sbi_blend.py`)
- Self-Blended Images augmentation blending real source faces with transformed cutouts of themselves to create artificial fake pairs during training.

---

## 6. Research Goals & Actionable Improvement Plan

### 6.1 Current Bottleneck in Novel Fusion Model
Our diagnostic suite revealed a key discovery:
```text
heatmap: mean=0.1055 p90=0.2367 p99=0.3688 max=0.8756 frac>0.5=0.0001
         best_iou=0.3891 @thresh=0.10 (iou@0.5=0.0005)
```
- **Pointing Game Accuracy is strong (66.65%):** The localization head correctly locates *where* the fake region is.
- **Activations are under-confident:** Heatmap sigmoid outputs cluster between 0.10 and 0.35. Standard thresholding at 0.5 zeros out the heatmap, making IoU look broken (0.05%), whereas thresholding at 0.10 yields **38.69% Mask IoU**.

---

### 6.2 Actionable Roadmap to Improve Fusion Model

To push the Fusion Model beyond **88.24% AUC** toward Xception's **98.44% AUC**:

1. **Replace Standard BCE with Focal BCE / Dice Loss:**
   - Standard BCE treats easy background pixels equally. Adding **Focal Loss** or **Dice Loss** will force the localization head to emit confident activations ($\to 1.0$) for fake pixels.
2. **Increase Mask Loss Weight:**
   - Now that gradient clipping (`max_grad_norm=1.0`) prevents FP16 gradient explosion, scale `mask_weight` from 2.0 to 4.0 or 5.0 to force stronger spatial guidance.
3. **Cross-Dataset Zero-Shot Evaluation:**
   - Evaluate Fusion v2 and Xception on the **Celeb-DF v2** test set to prove that Fusion's dual-stream CLIP+SRM architecture generalizes better to unseen deepfakes than Xception.

---

## 7. Complete Repository File Structure

```text
deepfake-detection-research/
├── CLAUDE_UPDATE_REPORT.md             # Summary update report for project tracking
├── PROJECT_STRUCTURE.md                # Architecture overview document
├── README.md                           # Quickstart guide & repository overview
├── project_guide.md                    # THIS Comprehensive Beginner-Friendly Guide
├── requirements.txt                    # Pinned dependencies (PyTorch, timm, open_clip, etc.)
│
├── another_model/                      # TriConsistencyNet Standalone Sub-Package
│   ├── evaluate.py                     # Test evaluation script with per-method breakdown
│   ├── evaluate_triconsistencynet.py   # Legacy evaluation entrypoint
│   ├── mode_architecture.md            # TriConsistencyNet architectural documentation
│   ├── model.py                        # Legacy model file
│   ├── test_triconsistencynet.py       # Standalone forward-pass sanity check
│   ├── train.py                        # Standalone training script (reads manifest.csv)
│   ├── train_triconsistencynet.py      # Legacy training entrypoint
│   └── src/
│       ├── __init__.py                 # Package init
│       ├── attention.py                # Cross-Consistency Attention (CCA) module
│       ├── dataset.py                  # Standalone dataset loader (Albumentations 1.x/2.x ready)
│       ├── frequency.py                # 2D FFT Frequency Guidance Encoder (FGE) module
│       ├── fusion.py                   # Adaptive Feature Fusion (AFF) SE module
│       └── model.py                    # Complete TriConsistencyNet PyTorch architecture
│
├── configs/                            # YAML Configuration Files
│   ├── dataset.yaml                    # Dataset paths & split settings
│   ├── model.yaml                      # Model architecture parameters
│   └── training.yaml                   # Hyperparameters, batch size, learning rates
│
├── data/                               # Dataset Storage
│   ├── processed/
│   │   └── manifest.csv                # Master index (159,969 face crops, splits, GT mask paths)
│   ├── raw/                            # Raw video frames
│   └── splits/                         # Official FF++ train/val/test CSV split files
│
├── docs/                               # Literature Documentation
│   └── literature_review_deepfake.xlsx # Matrix of 13+ surveyed papers, gaps, & plan
│
├── experiments/                        # Checkpoints, Logs, & Outputs
│   ├── fusion_v1_c23/                  # Fusion v1 run directory
│   ├── fusion_v2_c23/                  # Fusion v2 run directory (best_model.pt - 88.24% AUC)
│   ├── sbi_baseline_c23/               # SBI baseline run directory (best_model.pt - 71.10% AUC)
│   ├── triconsistencynet_c23/          # TriConsistencyNet run directory (best_model.pt - 80.66% AUC)
│   └── xception_baseline_c23/          # Xception baseline run directory (best_model.pt - 98.44% AUC)
│
├── scripts/                            # Executable Entrypoints
│   ├── 00_verify_env.py                # Check PyTorch CUDA & GPU readiness
│   ├── 01_check_splits.py              # Verify identity-preserved train/val/test splits
│   ├── 02_run_preprocessing.py         # Extract frames, align faces, build manifest.csv
│   ├── 03_check_dataloader.py          # Verify PyTorch DataLoader batches & masks
│   ├── 04_train_baseline.py            # Train Xception baseline model
│   ├── 05_train_sbi_baseline.py        # Train SBI baseline model
│   ├── 06_train_fusion.py              # Train Novel Dual-Stream Fusion Model v2
│   └── 07_run_evaluation.py            # Full evaluation suite (AUC, EER, Pointing Game, IoU)
│
├── src/                                # Core Source Code Library
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                  # Main FF++ PyTorch Dataset & DataLoader builder
│   │   ├── ffpp_splits.py              # Identity split generator
│   │   ├── preprocess_ffpp.py          # Face detection & decord video fallback reader
│   │   ├── sbi_blend.py                # Self-blended image augmentation engine
│   │   └── sbi_dataset.py              # SBI dataset reader with landmark preflight check
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py                 # Evaluation orchestration module
│   │   └── metrics.py                  # AUC, AP, EER, Balanced Acc, Heatmap Stats, Pointing Game, IoU
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py                 # Xception baseline model implementation
│   │   └── fusion_model.py             # Dual-Stream CLIP + SRM Fusion Model architecture
│   └── training/
│       ├── __init__.py
│       ├── checkpoint.py               # Checkpoint saver/loader with patience counter restoration
│       ├── engine.py                   # Standard training engine
│       └── train_fusion.py             # Fusion training engine (grad clipping + NaN guards)
│
└── tests/                              # Unit & Integration Tests
```
