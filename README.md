# Deepfake Detection Research Project (FF++ C23, DGX A100)

Research project: generalizable deepfake detection trained on FaceForensics++
(C23), evaluated across in-dataset, cross-manipulation, and cross-dataset
protocols, targeting the gap identified in `docs/literature_review_deepfake.xlsx`
(Gap Analysis tab) — a fusion of vision-language semantic features with a
compression-aware frequency-forensics branch, with a quantitatively-evaluated
explainability head (not just qualitative Grad-CAM).

See **`PROJECT_STRUCTURE.md`** for the full directory map and phase checklist —
that file is the authoritative reference for where everything lives.

## Setup
```bash
cd deepfake-detection-research
pip install -r requirements.txt   # torch itself should already be on the DGX image
```

## Phase -1 — download the dataset (if not already done)
Use the official downloader saved at `external/faceforensics_download_v4.py`
(confirmed correct against the file you got from the Google Form — see
`external/README.md` for why the `v1` script some people find is NOT this one).
```bash
python external/faceforensics_download_v4.py /data/FaceForensics++ -d all -c c23 -t videos
python external/faceforensics_download_v4.py /data/FaceForensics++ -d all -c c23 -t masks
```
Then get the official split files:
https://github.com/ondyari/FaceForensics/tree/master/dataset/splits
→ put `train.json`/`val.json`/`test.json` in `/data/FaceForensics++/splits/` (or `data/splits/` in this repo if you copy them here).

## Run order (current phases — updated as the project progresses)

**Phase 0 — verify environment**
```bash
python scripts/00_verify_env.py
```

**Phase 1 — sanity-check FF++ splits against your downloaded data**
1. Confirm `train.json`/`val.json`/`test.json` are in your splits folder (see Phase -1 above).
2. Run:
```bash
python scripts/01_check_splits.py \
  --ffpp_root /path/to/FaceForensics++ \
  --splits_dir /path/to/FaceForensics++/splits \
  --compression c23
```
Expect roughly (standard FF++ split sizes: 720/140/140 unique real videos for
train/val/test; each real id appears as the "target" in exactly one fake video
per method, per pair-ordering — see `external/README.md` for why both `a_b.mp4`
and `b_a.mp4` exist):
train ≈ 720 real + 2880 fake, val ≈ 140 real + 560 fake, test ≈ 140 real + 560 fake.
Many `[WARN] missing ...` lines mean your folder layout doesn't match FF++'s
standard structure — check `PROJECT_STRUCTURE.md` and the docstring in
`src/data/ffpp_splits.py`.

**Phase 2 — preprocess: face crops + manifest**

Small test run first (check speed/storage before committing):
```bash
python scripts/02_run_preprocessing.py \
  --ffpp_root /path/to/FaceForensics++ \
  --splits_dir /path/to/FaceForensics++/splits \
  --output_root data/processed \
  --compression c23 \
  --split train \
  --frames_per_video 8 \
  --image_size 299
```
Then the full run:
```bash
python scripts/02_run_preprocessing.py \
  --ffpp_root /path/to/FaceForensics++ \
  --splits_dir /path/to/FaceForensics++/splits \
  --output_root data/processed \
  --compression c23 \
  --split train val test \
  --frames_per_video 32 \
  --image_size 299 \
  --extract_masks
```
Safe to Ctrl-C and re-run — already-processed videos are skipped.
Result: `data/processed/manifest.csv` + face-crop JPEGs (+ aligned ground-truth
masks for fake videos, used later for quantitative explainability evaluation).

**Sanity check before moving on:** open 5-10 saved crops from
`data/processed/train/real/...` and `.../fake/...` and confirm faces are
actually centered, not blank/black.

**Phase 3 — Dataset/DataLoader sanity check**
```bash
python scripts/03_check_dataloader.py \
  --manifest data/processed/manifest.csv \
  --image_size 299 \
  --batch_size 32 \
  --check_masks
```
Confirms: per-split class counts, correct tensor shapes/dtypes from the
DataLoader, and that the `WeightedRandomSampler` actually balances the
1:4 real:fake frame ratio (target ~0.5 real fraction over sampled batches,
not the raw ~0.2). If this all looks right, the data pipeline is done and
we move to baseline reproduction.

**Phase 3b — train the Xception baseline**
```bash
python scripts/04_train_baseline.py \
  --manifest data/processed/manifest.csv \
  --model xception \
  --image_size 299 \
  --batch_size 32 \
  --epochs 30 \
  --lr 1e-4 \
  --lr_scheduler cosine \
  --run_name xception_baseline_c23
```
Trains with mixed precision, early-stops on validation AUC (default patience 5
epochs), saves the best checkpoint to `experiments/xception_baseline_c23/best_model.pt`,
and writes `history.json` (per-epoch metrics) + `test_metrics.json` (final
AUC/ACC/EER/AP on the held-out test split, using the best val checkpoint).
`--lr_scheduler` accepts `none`/`cosine`/`plateau` (default `cosine`).

This reproduces the original FF++ paper's baseline on your exact split —
**do this before writing any novel-architecture code.** It's your only fair
comparison point, and if something in the data pipeline is subtly wrong,
this is where you'll see it (e.g. near-chance AUC, NaN loss, wildly unstable
training).

**Phase 3b — train the SBI (Self-Blended Images) baseline**

SBI needs facial landmarks. Install dlib and get the 68-point predictor file:
```bash
pip install dlib   # compiles from source; needs cmake + build-essential if missing
# Download shape_predictor_68_face_landmarks.dat, e.g.:
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```
Then train:
```bash
python scripts/05_train_sbi_baseline.py \
  --manifest data/processed/manifest.csv \
  --dlib_predictor_path /path/to/shape_predictor_68_face_landmarks.dat \
  --image_size 299 \
  --batch_size 32 \
  --epochs 30 \
  --lr 1e-4 \
  --run_name sbi_baseline_c23
```
Key difference from the Xception baseline: SBI trains **only** on self-blended
real images (generated on the fly, never touching actual FF++ forgeries during
training) but is still evaluated on real val/test forgeries — so its numbers
tell you how well *pure self-supervised blending* generalizes, versus Xception
trained directly on real manipulated videos. Both should end up in the same
results table later.

**Phase 4 — train the novel fusion model**

This is the actual novel contribution: a CLIP semantic branch (frozen except
the last few transformer blocks + a small adapter) fused with a
compression-aware frequency-forensics branch (fixed SRM filters + a learned
gate that down-weights frequency cues when the input looks heavily
compressed), plus a localization head trained against FF++'s real
ground-truth manipulation masks — so explainability gets a genuine
quantitative score (pointing-game accuracy, mask IoU), not just a Grad-CAM
picture.

Requires manifest.csv built **with** `--extract_masks` (re-run
`scripts/02_run_preprocessing.py --extract_masks` if you skipped that flag
earlier — the localization head has nothing to train against otherwise).

```bash
python scripts/06_train_fusion.py \
  --manifest data/processed/manifest.csv \
  --image_size 224 \
  --batch_size 32 \
  --epochs 30 \
  --lr 1e-4 \
  --mask_weight 1.0 \
  --run_name fusion_v1_c23
```
`--image_size 224` is recommended (CLIP ViT-B-32's native positional embedding
size) rather than the 299 used for the Xception/SBI baselines — this is a
different image size convention between the baselines and the novel model,
worth keeping in mind if you build a shared results-plotting script later.

Prints, per epoch: combined loss (classification + mask), val AUC/ACC as
before, plus `val_pointing_game` and `val_mask_iou` — the two quantitative
explainability numbers. Final test metrics (best val checkpoint) are written
to `experiments/<run_name>/test_metrics.json`, same as the other two scripts.

**Resuming an interrupted run (any of scripts 04/05/06)**

Every run writes `experiments/<run_name>/latest_checkpoint.pt` after each
epoch (full training state: model + optimizer + scheduler + history) —
separate from `best_model.pt` (weights only, used for evaluation). If a run
gets interrupted, resume it with the same command plus:
```bash
  --resume_from_checkpoint experiments/<run_name>/latest_checkpoint.pt
```
`--epochs` should be the TOTAL epoch count for the whole run (not just the
remaining epochs) — e.g. if it died after epoch 12 of a planned 30, resume
with `--epochs 30` again and it'll correctly continue from epoch 13.

**Not yet wired in:** `TemporalAttentionHead` (in `src/models/fusion_model.py`)
is implemented and unit-tested standalone, but the current Dataset/DataLoader
is frame-level, not clip-level — a multi-frame Dataset variant is needed
before the temporal head can plug into the main model. Flagged in
PROJECT_STRUCTURE.md as a known gap, not a blocker for the current pipeline.

**Phase 6 — multi-protocol evaluation**

Once you have a trained checkpoint (from scripts/04, 05, or 06), evaluate it
across all protocols in one pass:
```bash
# Xception or SBI checkpoint (same architecture class):
python scripts/07_run_evaluation.py \
  --architecture xception \
  --checkpoint experiments/xception_baseline_c23/best_model.pt \
  --manifest data/processed/manifest.csv \
  --image_size 299

# Fusion checkpoint (pass the SAME hyperparameters used to train it — see
# experiments/<run_name>/config.json):
python scripts/07_run_evaluation.py \
  --architecture fusion \
  --checkpoint experiments/fusion_v1_c23/best_model.pt \
  --manifest data/processed/manifest.csv \
  --image_size 224
```
Prints and saves (to `evaluation_results.json` next to the checkpoint, unless
`--output` is given):
- **In-dataset**: full test-split AUC/ACC/EER/AP, plus a per-method breakdown
  (Deepfakes/Face2Face/FaceSwap/NeuralTextures each vs. real) — this single
  pass computes both without needing separate dataloaders per method.
- **Cross-dataset zero-shot** (only if `--cross_dataset_manifest` is given —
  e.g. a manifest built from Celeb-DF; that preprocessing path isn't built
  yet, see "What's NOT built yet" below).
- **Explainability** (fusion checkpoints only, automatic): pointing-game
  accuracy + mask IoU.

**Cross-manipulation leave-one-out training:** to genuinely test
generalization to an unseen manipulation method (not just report a per-method
breakdown on a model trained on everything), retrain with a method excluded:
```bash
python scripts/04_train_baseline.py --manifest data/processed/manifest.csv \
  --exclude_methods Deepfakes --run_name xception_loo_deepfakes
```
Train/val exclude `Deepfakes`; the test split still includes it (this is
handled automatically), so `scripts/07_run_evaluation.py`'s per-method
breakdown on that checkpoint tells you how well it generalizes to a
manipulation method it never saw during training. Same flag works on
`scripts/06_train_fusion.py`.

## What's NOT built yet (see PROJECT_STRUCTURE.md checklist)
- Clip-level (multi-frame) Dataset variant + wiring in `TemporalAttentionHead`
- Cross-dataset preprocessing (e.g. a Celeb-DF downloader + manifest builder, so `--cross_dataset_manifest` above has something real to point at)
- Ablations, efficiency report, results consolidation, paper draft

## Testing
```bash
python tests/test_pipeline_synthetic.py   # splits + preprocessing pipeline
python tests/test_dataset.py              # Dataset/DataLoader
python tests/test_baseline.py             # model + metrics + training engine
python tests/test_sbi.py                  # SBI blending + dataset (mock landmark detector, no dlib needed)
python tests/test_fusion_model.py         # fusion architecture: shapes, gradients, freezing, localization metrics
python tests/test_evaluate.py             # leave-one-out filtering, checkpoint load, multi-protocol evaluation
python tests/test_checkpoint.py           # full training-state checkpoint save/load resume, LR scheduler stepping
```
`test_fusion_model.py` builds a real (randomly-initialized) CLIP ViT-B-32, so
it's slower (~10-30s) and heavier on memory than the others — that's expected.
Self-contained synthetic smoke tests — build tiny fake data + mocked
detector, validate pipeline logic without touching real data or needing
network/model downloads. Re-run after any change to `src/data/*.py`.

## Reference documents
- `docs/literature_review_deepfake.xlsx` — 30-paper Literature Matrix, Gap
  Analysis, and the original Project Plan (phase-by-phase task breakdown)
- `configs/ffpp_c23.yaml` — central reference for dataset paths and hyperparameters
