"""
Load PNG/JPG images + scalar scores from a CSV manifest for FBP training.

Requires 68-point landmarks per sample: either precomputed .npy files or runtime
face_alignment (optional dependency).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.ldl_labels import score_to_distribution


def _default_image_transform(image_size: tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _load_landmarks_npy(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if arr.shape != (68, 2):
        raise ValueError(
            f"Landmark file {path} must have shape (68, 2), got {arr.shape}"
        )
    return arr.astype(np.float32)


_face_alignment_instance = None


def _get_face_alignment():
    global _face_alignment_instance
    if _face_alignment_instance is None:
        import face_alignment

        _face_alignment_instance = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _face_alignment_instance


def _landmarks_face_alignment_rgb(rgb: np.ndarray) -> np.ndarray:
    fa = _get_face_alignment()
    preds = fa.get_landmarks(rgb)
    if preds is None or len(preds) == 0:
        raise RuntimeError(
            "face_alignment: no face detected; skip this image or use precomputed .npy landmarks"
        )
    return preds[0].astype(np.float32)


class FBPImageScoreManifestDataset(Dataset):
    """
    CSV columns (header required):
      - path: relative or absolute path to image (png/jpg/...)
      - score: numeric label (default: 1..K when one_based=True, e.g. 1–100)

    Optional columns ignored unless you extend the reader.

    Landmarks:
      - landmark_mode=\"npy\": for image stem \"foo.png\", load
        landmark_dir / \"foo.npy\" with shape (68, 2), same pixel space as the
        **original** image file (before resize). They are scaled to match the
        resized tensor coordinates (224x224) inside __getitem__.
      - landmark_mode=\"face_alignment\": run face_alignment on loaded RGB
        (install: pip install face_alignment). Use num_workers=0 in DataLoader
        to avoid multiprocessing issues with the detector.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        *,
        image_root: str | Path | None = None,
        num_classes: int = 100,
        one_based_score: bool = True,
        soft_sigma: float = 0.0,
        image_size: tuple[int, int] = (224, 224),
        landmark_mode: Literal["npy", "face_alignment"] = "npy",
        landmark_dir: str | Path | None = None,
        transform: transforms.Compose | None = None,
        landmark_scale_fn: (
            Callable[[np.ndarray, tuple[int, int], tuple[int, int]], np.ndarray] | None
        ) = None,
    ) -> None:
        super().__init__()
        self._manifest = Path(manifest_csv)
        self._root = Path(image_root) if image_root is not None else None
        self._num_classes = num_classes
        self._one_based = one_based_score
        self._soft_sigma = float(soft_sigma)
        self._image_size = image_size
        self._landmark_mode = landmark_mode
        self._landmark_dir = Path(landmark_dir) if landmark_dir is not None else None

        self._rows: list[tuple[Path, float]] = []
        with self._manifest.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV manifest is empty or has no header")
            fields = {h.strip().lower(): h for h in reader.fieldnames}
            if "path" not in fields or "score" not in fields:
                raise ValueError(
                    'CSV must have columns "path" and "score" (case-insensitive)'
                )
            col_path = fields["path"]
            col_score = fields["score"]
            for row in reader:
                p = row[col_path].strip()
                s = float(row[col_score])
                path_obj = Path(p)
                if self._root is not None and not path_obj.is_absolute():
                    path_obj = self._root / path_obj
                self._rows.append((path_obj, s))

        if len(self._rows) == 0:
            raise ValueError(f"No rows in manifest: {self._manifest}")

        if landmark_mode == "npy":
            if self._landmark_dir is None:
                raise ValueError('landmark_mode="npy" requires landmark_dir to be set')
        if landmark_mode == "face_alignment" and self._landmark_dir is not None:
            pass

        self._transform = transform if transform is not None else _default_image_transform(
            image_size
        )
        self._landmark_scale_fn = landmark_scale_fn or _scale_landmarks_to_resized

    def __len__(self) -> int:
        return len(self._rows)

    def _resolve_landmarks(
        self,
        image_path: Path,
        pil_before_resize: Image.Image,
    ) -> np.ndarray:
        w0, h0 = pil_before_resize.size
        tw, th = self._image_size

        if self._landmark_mode == "npy":
            assert self._landmark_dir is not None
            stem = image_path.stem
            npy_path = self._landmark_dir / f"{stem}.npy"
            if not npy_path.is_file():
                raise FileNotFoundError(
                    f"Missing landmarks: {npy_path}. "
                    f'Expected landmark_mode="npy" file for stem "{stem}".'
                )
            lm = _load_landmarks_npy(npy_path)
            return self._landmark_scale_fn(lm, (w0, h0), (tw, th))

        if self._landmark_mode == "face_alignment":
            rgb = np.array(pil_before_resize.convert("RGB"))
            lm = _landmarks_face_alignment_rgb(rgb)
            return self._landmark_scale_fn(lm, (w0, h0), (tw, th))

        raise RuntimeError(f"Unknown landmark_mode: {self._landmark_mode}")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path, score = self._rows[idx]
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        pil = Image.open(path).convert("RGB")
        landmarks_np = self._resolve_landmarks(path, pil)
        image_tensor = self._transform(pil)

        target_dist = score_to_distribution(
            score,
            self._num_classes,
            one_based=self._one_based,
            sigma=self._soft_sigma,
        )

        landmarks = torch.from_numpy(landmarks_np).float()

        return {
            "image": image_tensor,
            "landmarks": landmarks,
            "target_dist": target_dist,
        }


def _scale_landmarks_to_resized(
    lm: np.ndarray,
    orig_hw: tuple[int, int],
    new_hw: tuple[int, int],
) -> np.ndarray:
    """Scale (x,y) from original image size (W,H) to resized (W,H)."""
    w0, h0 = orig_hw
    w1, h1 = new_hw
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original image size")
    out = lm.copy()
    out[:, 0] *= w1 / float(w0)
    out[:, 1] *= h1 / float(h0)
    return out.astype(np.float32)
