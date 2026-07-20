# external/

Unmodified third-party scripts we depend on but don't own. Never edit files
here directly — if a fix is needed, wrap it (see below) rather than patching
the vendor file, so re-downloading a fresh copy from upstream is always safe.

## faceforensics_download_v4.py
Official FF++ downloader from https://github.com/ondyari/FaceForensics
(confirmed correct — this is what the Google Form access grants; it is NOT
the same as the older FaceForensics v1 downloader, which only covers Face2Face
and lacks c23/c40 compression options).

Basic usage (see script's own `-h` for full options):
```bash
python external/faceforensics_download_v4.py <output_dir> -d all -c c23 -t videos
python external/faceforensics_download_v4.py <output_dir> -d all -c c23 -t masks
```
This populates `<output_dir>/original_sequences/...` and
`<output_dir>/manipulated_sequences/...` in exactly the layout
`src/data/ffpp_splits.py` expects.

**Important (confirmed by reading this script's filelist logic):** for every
pair `(a, b)` in the dataset, BOTH `a_b.mp4` and `b_a.mp4` are separate real
files on the server — not "either one or the other". Our `ffpp_splits.py`
resolves both orderings as distinct fake videos per pair, per method.
