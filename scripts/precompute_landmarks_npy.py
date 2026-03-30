"""
Precompute 68-point face landmarks (face_alignment, dlib-compatible order) as .npy files.

Usage (from project root):
  pip install face_alignment
  python scripts/precompute_landmarks_npy.py --manifest data/train.csv --image-root . --out-dir data/landmarks

Writes one file per image stem: out-dir/<stem>.npy with shape (68, 2) in **original** image pixels.
Training dataset will rescale these to match the resized 224x224 input.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help='CSV with columns path,score')
    parser.add_argument("--image-root", default="", help="Optional root for relative paths")
    parser.add_argument("--out-dir", required=True, help="Directory to write .npy files")
    args = parser.parse_args()

    root = Path(args.image_root) if args.image_root else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import face_alignment

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=dev,
    )

    manifest = Path(args.manifest)
    rows: list[tuple[Path, str]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {h.strip().lower(): h for h in (reader.fieldnames or [])}
        col_path = fields["path"]
        for row in reader:
            p = row[col_path].strip()
            path_obj = Path(p)
            if root is not None and not path_obj.is_absolute():
                path_obj = root / path_obj
            rows.append((path_obj, path_obj.stem))

    ok = 0
    for path_obj, stem in rows:
        if not path_obj.is_file():
            print(f"skip missing: {path_obj}")
            continue
        rgb = np.array(Image.open(path_obj).convert("RGB"))
        preds = fa.get_landmarks(rgb)
        if preds is None or len(preds) == 0:
            print(f"skip no face: {path_obj}")
            continue
        lm = preds[0].astype(np.float32)
        np.save(out_dir / f"{stem}.npy", lm)
        ok += 1
    print(f"Saved {ok} landmark files to {out_dir}")


if __name__ == "__main__":
    main()
