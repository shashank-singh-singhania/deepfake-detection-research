"""
Utilities to load the OFFICIAL FaceForensics++ train/val/test splits and resolve
them to actual video file paths on disk.

Official split files (train.json / val.json / test.json) come from:
    https://github.com/ondyari/FaceForensics/tree/master/dataset/splits
Each file is a JSON list of 2-element lists of zero-padded video IDs, e.g.:
    [["000", "003"], ["001", "892"], ...]

- Both IDs in a pair are valid REAL videos (original_sequences/youtube/<c>/videos/<id>.mp4)
- Each pair corresponds to TWO manipulated videos per method: "<a>_<b>.mp4" AND
  "<b>_<a>.mp4" — confirmed directly from the official download script
  (ondyari/FaceForensics download.py), which builds its fake-video filelist as:
      filelist.append('_'.join(pair))
      filelist.append('_'.join(pair[::-1]))
  i.e. both orderings are separate, real files on the server — NOT "either one
  or the other". A pair therefore yields 2 real videos + 2 fake videos per
  method (one fake video per real id, with that id as the "target" identity).
  This is why aggregate FF++ counts work out to 1000 real / 1000 fake per
  method (~500 pairs x 2 orderings), and why train/val/test have equal real
  and per-method fake video counts (e.g. 720 real == 720 fake per method).

METHODS = the 4 standard FF++ manipulation methods.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
REAL_SOURCE = "youtube"


@dataclass
class VideoItem:
    path: Path
    label: int          # 0 = real, 1 = fake
    method: str         # "youtube" for real, else manipulation method name
    video_id: str        # e.g. "000" or "000_003"
    split: str           # train / val / test
    mask_path: Optional[Path] = None


def load_split_ids(splits_dir: Path, split_name: str) -> list[tuple[str, str]]:
    """Load a split json and return list of (id_a, id_b) string tuples."""
    fp = Path(splits_dir) / f"{split_name}.json"
    with open(fp, "r") as f:
        pairs = json.load(f)
    return [(str(a), str(b)) for a, b in pairs]


def _resolve_fake_video(root: Path, method: str, compression: str, x: str, y: str) -> Optional[Path]:
    """Resolve one specific ordering "<x>_<y>.mp4" (caller tries both orderings separately)."""
    vid_dir = root / "manipulated_sequences" / method / compression / "videos"
    p = vid_dir / f"{x}_{y}.mp4"
    return p if p.exists() else None


def _resolve_mask_video(root: Path, method: str, x: str, y: str) -> Optional[Path]:
    # Masks are stored at raw resolution (no compression variants) for most methods
    mask_dir = root / "manipulated_sequences" / method / "masks" / "videos"
    p = mask_dir / f"{x}_{y}.mp4"
    return p if p.exists() else None


def _resolve_real_video(root: Path, compression: str, vid_id: str) -> Optional[Path]:
    p = root / "original_sequences" / REAL_SOURCE / compression / "videos" / f"{vid_id}.mp4"
    return p if p.exists() else None


def build_video_list(
    ffpp_root: str,
    splits_dir: str,
    split_name: str,
    compression: str = "c23",
    methods: list[str] = None,
    include_masks: bool = False,
) -> list[VideoItem]:
    """
    Returns a de-duplicated list of VideoItem for the given split, covering
    real videos (once each) and fake videos (BOTH orderings per pair, per
    requested method — see module docstring for why both exist).
    """
    root = Path(ffpp_root)
    methods = methods or METHODS
    pairs = load_split_ids(Path(splits_dir), split_name)

    items: list[VideoItem] = []
    seen_real_ids = set()
    seen_fake_ids: set[tuple[str, str]] = set()  # (method, video_id) to avoid dupes if a pair repeats

    for a, b in pairs:
        # --- real videos: each unique id used once ---
        for vid_id in (a, b):
            if vid_id in seen_real_ids:
                continue
            seen_real_ids.add(vid_id)
            rp = _resolve_real_video(root, compression, vid_id)
            if rp is None:
                print(f"[WARN] missing real video id={vid_id} (split={split_name})")
                continue
            items.append(VideoItem(path=rp, label=0, method=REAL_SOURCE,
                                    video_id=vid_id, split=split_name))

        # --- fake videos: BOTH orderings per pair, per method ---
        for method in methods:
            for x, y in ((a, b), (b, a)):
                vid_id = f"{x}_{y}"
                if (method, vid_id) in seen_fake_ids:
                    continue
                seen_fake_ids.add((method, vid_id))
                fp = _resolve_fake_video(root, method, compression, x, y)
                if fp is None:
                    print(f"[WARN] missing fake video id={vid_id} method={method} (split={split_name})")
                    continue
                mask_p = _resolve_mask_video(root, method, x, y) if include_masks else None
                items.append(VideoItem(path=fp, label=1, method=method,
                                        video_id=vid_id, split=split_name,
                                        mask_path=mask_p))

    return items
