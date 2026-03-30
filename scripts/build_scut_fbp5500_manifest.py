"""
Build train_pipeline-compatible manifest CSV from SCUT-FBP5500 official split files.

Prerequisites:
  1. Download the SCUT-FBP5500 image archive (see HCIILAB README / Google Drive).
  2. Optionally clone split lists from GitHub:
     https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/tree/master/data/1
     Files: train_1.txt, test_1.txt, ... (5-fold) or use 60/40 splits in other folders.

Usage (from project root):

  python scripts/build_scut_fbp5500_manifest.py ^
    --split-file path/to/train_1.txt ^
    --dataset-root path/to/where/images/are ^
    --out-csv data/scut_train_fold1.csv ^
    --score-scale linear_1_to_100

Then train:

  python train_pipeline.py --manifest data/scut_train_fold1.csv --image-root . ^
    --landmark-mode face_alignment --num-workers 0 --num-classes 100

Or with precomputed 68-point .npy (dlib order) under data/landmarks/:

  python train_pipeline.py --manifest ... --landmark-dir data/landmarks --landmark-mode npy --num-classes 100

Note: SCUT provides 86-point .pts landmarks; this project uses 68 dlib-order points.
Use face_alignment at train time, or precompute with scripts/precompute_landmarks_npy.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.scut_fbp5500 import (
    iter_scut_split_file,
    map_scut_score_to_training_scale,
    resolve_image_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SCUT-FBP5500 -> FBP manifest CSV")
    parser.add_argument(
        "--split-file",
        type=str,
        required=True,
        help="Official train_*.txt or test_*.txt from SCUT-FBP5500-Database-Release/data/",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root folder containing image files listed in the split (e.g. unpacked Images/).",
    )
    parser.add_argument("--out-csv", type=str, required=True, help="Output manifest path")
    parser.add_argument(
        "--score-scale",
        type=str,
        choices=("round_1_to_5", "linear_1_to_100"),
        default="linear_1_to_100",
        help="round_1_to_5: integer 1..5 for --num-classes 5. "
        "linear_1_to_100: map [1,5] mean to 1..100 for --num-classes 100 (default in this repo).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip rows whose image file does not exist under --dataset-root",
    )
    parser.add_argument(
        "--require-exists",
        action="store_true",
        help="Fail if any image path is missing (after --dataset-root join)",
    )
    args = parser.parse_args()

    split_path = Path(args.split_file)
    root = Path(args.dataset_root)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_raw = iter_scut_split_file(split_path)
    written = 0
    skipped = 0
    missing = 0

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "score"])
        for fname, mean_s in rows_raw:
            img_path = resolve_image_path(root, fname)
            if not img_path.is_file():
                missing += 1
                if args.require_exists:
                    raise FileNotFoundError(f"Missing image: {img_path}")
                if args.skip_missing:
                    skipped += 1
                    continue
                raise FileNotFoundError(
                    f"Missing image: {img_path}\n"
                    f"Check --dataset-root (got {root}). "
                    f"If your zip uses subfolders, put that structure under dataset-root or symlink."
                )
            score_out = map_scut_score_to_training_scale(mean_s, args.score_scale)
            try:
                rel = img_path.resolve().relative_to(root.resolve())
                path_cell = str(rel).replace("\\", "/")
            except ValueError:
                path_cell = str(img_path).replace("\\", "/")
            w.writerow([path_cell, f"{score_out:g}"])
            written += 1

    print(f"Wrote {written} rows to {out_path}")
    if skipped:
        print(f"Skipped {skipped} missing files")
    if missing and not args.skip_missing and not args.require_exists:
        pass


if __name__ == "__main__":
    main()
