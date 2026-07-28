"""
Download script for the DeepFakeFace (DFF) dataset from Hugging Face:
https://huggingface.co/datasets/OpenRL/DeepFakeFace

This dataset evaluates zero-shot cross-domain generalization from GAN-based fakes (FF++)
to modern Diffusion Model-based fakes (Stable Diffusion v1.5, SD Inpainting, InsightFace).

Files downloaded into data/dff_raw/:
  - wiki.zip (Real images from IMDB-WIKI)
  - insight.zip (Fake images via InsightFace face-swapping)
  - text2img.zip (Fake images via Stable Diffusion v1.5)
  - inpainting.zip (Fake images via SD Inpainting)
"""
import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

DFF_URLS = {
    "wiki": "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/wiki.zip",
    "insight": "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/insight.zip",
    "text2img": "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/text2img.zip",
    "inpainting": "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/inpainting.zip",
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"File already exists: {output_path} (skipping download)")
        return
    print(f"Downloading {url} to {output_path}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def unzip_file(zip_path: Path, extract_to: Path):
    print(f"Unzipping {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Done unzipping {zip_path.name}!")


def main():
    target_dir = Path("data/dff_raw")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("=== Downloading DeepFakeFace (DFF) Dataset from Hugging Face ===")
    for category, url in DFF_URLS.items():
        zip_path = target_dir / f"{category}.zip"
        download_file(url, zip_path)
        
        extract_path = target_dir / category
        if not extract_path.exists():
            unzip_file(zip_path, extract_path)
        else:
            print(f"Already unzipped: {extract_path}")

    print("\nDeepFakeFace (DFF) raw dataset ready in data/dff_raw/!")


if __name__ == "__main__":
    main()
