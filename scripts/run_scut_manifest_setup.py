"""
One-shot: build manifest CSV for SCUT-FBP5500 training.

Priority:
  1) If data/scut_images/SCUT-FBP5500_v2/ is present (common Kaggle/HCI v2 layout),
     use fold-1 train_1.txt and Images/ as dataset root (filenames: AF*.jpg, AM*.jpg, ...).
  2) Else download official train_1.txt from GitHub (mty*.jpg naming) and use
     FBP_SCUT_IMAGE_ROOT or data/scut_images/ with --skip-missing.

From project root:
  python scripts/run_scut_manifest_setup.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT_URL = (
    "https://raw.githubusercontent.com/HCIILAB/SCUT-FBP5500-Database-Release/"
    "master/data/1/train_1.txt"
)
DEFAULT_SPLIT = ROOT / "data" / "scut_splits" / "train_1.txt"
V2_TRAIN1 = (
    ROOT
    / "data"
    / "scut_images"
    / "SCUT-FBP5500_v2"
    / "train_test_files"
    / "5_folders_cross_validations_files"
    / "cross_validation_1"
    / "train_1.txt"
)
V2_IMAGES = ROOT / "data" / "scut_images" / "SCUT-FBP5500_v2" / "Images"
DEFAULT_IMAGE_ROOT = Path(os.environ.get("FBP_SCUT_IMAGE_ROOT", str(ROOT / "data" / "scut_images")))
DEFAULT_OUT = ROOT / "data" / "scut_train_fold1.csv"


def main() -> int:
    DEFAULT_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    build = ROOT / "scripts" / "build_scut_fbp5500_manifest.py"

    if V2_TRAIN1.is_file() and V2_IMAGES.is_dir():
        split_path = V2_TRAIN1
        img_root = V2_IMAGES
        extra = []
        print("Using SCUT-FBP5500 v2 layout:")
        print(f"  split: {split_path}")
        print(f"  images: {img_root}")
    else:
        split_path = DEFAULT_SPLIT
        img_root = DEFAULT_IMAGE_ROOT
        extra = ["--skip-missing"]
        if not split_path.is_file():
            print(f"Downloading split file -> {split_path}")
            try:
                req = urllib.request.Request(
                    SPLIT_URL, headers={"User-Agent": "FBP-setup/1.0"}
                )
                with urllib.request.urlopen(req, timeout=120) as r:
                    split_path.write_bytes(r.read())
            except (urllib.error.URLError, OSError) as e:
                curl = shutil.which("curl")
                if curl is None:
                    raise RuntimeError(
                        f"Download failed ({e}). Install curl or download manually:\n  {SPLIT_URL}"
                    ) from e
                subprocess.check_call(
                    [curl, "-fsSL", "-o", str(split_path), SPLIT_URL], cwd=str(ROOT)
                )
        else:
            print(f"Using existing split: {split_path}")

    cmd = [
        sys.executable,
        str(build),
        "--split-file",
        str(split_path),
        "--dataset-root",
        str(img_root),
        "--out-csv",
        str(DEFAULT_OUT),
        "--score-scale",
        "linear_1_to_100",
    ] + extra
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))

    n_data = max(0, sum(1 for _ in DEFAULT_OUT.open(encoding="utf-8")) - 1)
    print()
    if n_data == 0:
        print(
            "Note: manifest has only the header (no images matched). "
            f"For v2, extract SCUT-FBP5500_v2 under data/scut_images/. "
            f"For GitHub split, copy *.jpg into:\n  {DEFAULT_IMAGE_ROOT}\n"
            "Then run this script again."
        )
    print(f"Manifest: {DEFAULT_OUT} ({n_data} samples)")
    print("Train example:")
    print(
        f'  python train_pipeline.py --manifest data/scut_train_fold1.csv '
        f'--image-root "{img_root}" '
        f"--landmark-mode face_alignment --num-workers 0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
