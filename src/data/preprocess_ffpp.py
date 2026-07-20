"""
Preprocess FaceForensics++ (C23) into aligned face crops ready for training.

For each video (real + all 4 manipulation methods, per the official split):
  1. Sample N frames evenly across the video.
  2. Run batched GPU face detection (facenet-pytorch MTCNN).
  3. Crop with a margin, resize to target resolution, save as JPEG.
  4. (Optional) Apply the identical crop box to the corresponding ground-truth
     manipulation mask frame, for later quantitative explainability evaluation.
  5. Write a manifest CSV of every saved crop with label/method/split/video_id/frame_idx.

Resumable: if a video's output folder already has the expected number of frames,
it is skipped. Safe to Ctrl-C and re-run.

This is a LIBRARY module (src/data/preprocess_ffpp.py) — do not run it directly.
The CLI entry point is scripts/02_run_preprocessing.py, which imports
`run_preprocessing()` from here. See PROJECT_STRUCTURE.md / README.md at the repo
root for the full run order.
"""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .ffpp_splits import build_video_list, VideoItem, METHODS


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_frame_indices(total_frames: int, n: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= n:
        return list(range(total_frames))
    return list(np.linspace(0, total_frames - 1, n, dtype=int))


def read_frames(video_path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Read specific frame indices from a video via OpenCV. Returns {idx: BGR frame}."""
    cap = cv2.VideoCapture(str(video_path))
    frames = {}
    idx_set = set(indices)
    max_idx = max(indices) if indices else -1
    i = 0
    while i <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if i in idx_set:
            frames[i] = frame
        i += 1
    cap.release()
    return frames


def crop_with_margin(frame: np.ndarray, box, margin: float, out_size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return crop, (x1, y1, x2, y2)


class FaceExtractor:
    """Thin wrapper around facenet-pytorch MTCNN for batched GPU face detection."""

    def __init__(self, device: str, image_size: int, margin: float):
        from facenet_pytorch import MTCNN
        self.device = device
        self.image_size = image_size
        self.margin = margin
        # keep_all=False -> largest/most confident face only (matches FF++ single-subject videos)
        self.mtcnn = MTCNN(keep_all=False, device=device, post_process=False)

    def detect_boxes(self, bgr_frames: list[np.ndarray]):
        """Batched detection. Returns list of box (x1,y1,x2,y2) or None per frame."""
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr_frames]
        boxes_batch, probs_batch = self.mtcnn.detect(rgb_frames)
        results = []
        for boxes in boxes_batch:
            if boxes is None or len(boxes) == 0:
                results.append(None)
            else:
                results.append(tuple(boxes[0]))  # highest-confidence face
        return results


def video_output_dir(output_root: Path, split: str, label: int, method: str, video_id: str) -> Path:
    label_str = "real" if label == 0 else "fake"
    return Path(output_root) / split / label_str / method / video_id


def already_done(out_dir: Path, expected_n: int) -> bool:
    if not out_dir.exists():
        return False
    n = len(list(out_dir.glob("frame_*.jpg")))
    return n >= max(1, int(expected_n * 0.8))  # tolerate a few undetected frames


def process_video(item: VideoItem, extractor: FaceExtractor, output_root: Path,
                   frames_per_video: int, image_size: int, margin: float,
                   extract_masks: bool, manifest_rows: list):
    out_dir = video_output_dir(output_root, item.split, item.label, item.method, item.video_id)

    if already_done(out_dir, frames_per_video):
        # still add existing frames to manifest when resuming
        for fp in sorted(out_dir.glob("frame_*.jpg")):
            manifest_rows.append({
                "path": str(fp), "label": item.label, "method": item.method,
                "split": item.split, "video_id": item.video_id,
                "frame_idx": int(fp.stem.split("_")[1]),
                "mask_path": str(fp.parent / "masks" / fp.name) if extract_masks and item.label == 1 else "",
            })
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks"
    if extract_masks and item.label == 1 and item.mask_path is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(item.path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    indices = sample_frame_indices(total, frames_per_video)
    if not indices:
        print(f"[WARN] unreadable/empty video: {item.path}")
        return

    frames = read_frames(item.path, indices)
    mask_frames = {}
    if extract_masks and item.label == 1 and item.mask_path is not None and item.mask_path.exists():
        mask_frames = read_frames(item.mask_path, indices)

    valid_idx = [i for i in indices if i in frames]
    if not valid_idx:
        return
    batch = [frames[i] for i in valid_idx]
    boxes = extractor.detect_boxes(batch)

    for idx, frame, box in zip(valid_idx, batch, boxes):
        if box is None:
            continue
        result = crop_with_margin(frame, box, margin, image_size)
        if result is None:
            continue
        crop, crop_box = result
        out_path = out_dir / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        mask_out_str = ""
        if idx in mask_frames:
            x1, y1, x2, y2 = crop_box
            mcrop = mask_frames[idx][y1:y2, x1:x2]
            if mcrop.size > 0:
                mcrop = cv2.resize(mcrop, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                mask_out_path = mask_dir / f"frame_{idx:04d}.jpg"
                cv2.imwrite(str(mask_out_path), mcrop)
                mask_out_str = str(mask_out_path)

        manifest_rows.append({
            "path": str(out_path), "label": item.label, "method": item.method,
            "split": item.split, "video_id": item.video_id, "frame_idx": idx,
            "mask_path": mask_out_str,
        })


def run_preprocessing(ffpp_root: str, splits_dir: str, output_root: str, compression: str = "c23",
                       methods: list = None, splits: list = None, frames_per_video: int = 32,
                       image_size: int = 299, margin: float = 0.3, extract_masks: bool = False):
    """Library entry point used by scripts/02_run_preprocessing.py (kept import-only, no argparse here)."""
    methods = methods or METHODS
    splits = splits or ["train", "val", "test"]

    device = get_device()
    print(f"Using device: {device}")
    extractor = FaceExtractor(device=device, image_size=image_size, margin=margin)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for split_name in splits:
        items = build_video_list(
            ffpp_root, splits_dir, split_name,
            compression=compression, methods=methods,
            include_masks=extract_masks,
        )
        print(f"\n=== Split '{split_name}': {len(items)} videos to process ===")
        for item in tqdm(items, desc=f"{split_name}"):
            try:
                process_video(item, extractor, output_root, frames_per_video,
                               image_size, margin, extract_masks, manifest_rows)
            except Exception as e:
                print(f"[ERROR] {item.path}: {e}")

    manifest_path = output_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "method", "split", "video_id", "frame_idx", "mask_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. Manifest written to {manifest_path} ({len(manifest_rows)} frame rows).")
    return manifest_path
