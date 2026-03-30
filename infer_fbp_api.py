"""
FBP inference interface:
  - Input: a PNG/JPG image path
  - It uses face_alignment to extract 68 2D landmarks
  - Then feeds (image, landmarks) into DualBranchFBPModel
  - Output: predicted beauty score (LDL expectation, score_hat)

This script is meant to be a "direct input image -> score" interface.

Important:
  The model must be loaded from your trained checkpoint for meaningful results.
  If you don't have a trained checkpoint, the score will be near-random.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

_DEFAULT_TORCH_HOME = r"D:\torch_home"
# face_alignment/torch 在 Windows 上下载模型权重时会依赖缓存路径。
# 为了避免中文用户名/路径编码导致的路径解析问题，推理脚本统一用 ASCII 路径。
os.environ.setdefault("TORCH_HOME", _DEFAULT_TORCH_HOME)

# Ensure Chinese output isn't garbled in Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import face_alignment
except Exception as e:  # pragma: no cover
    face_alignment = None
    _face_alignment_import_error = e

from models.dual_branch_fbp import DualBranchFBPModel
from utils.ldl_labels import score_to_distribution


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _read_scores_from_manifest(
    manifest_csv: str | Path,
    *,
    score_col: str = "score",
) -> list[float]:
    """
    Read `score` column from a CSV manifest (header required).
    """
    p = Path(manifest_csv)
    if not p.is_file():
        raise FileNotFoundError(f"prior manifest not found: {p}")

    scores: list[float] = []
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV manifest has no header: {p}")
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        if score_col.lower() not in fields:
            raise ValueError(
                f"CSV manifest missing column {score_col!r}. Columns: {reader.fieldnames}"
            )
        col_name = fields[score_col.lower()]
        for row in reader:
            scores.append(float(row[col_name]))
    return scores


def _compute_prior_val_distribution(
    scores: list[float],
    *,
    num_classes: int,
    val_ratio: float,
    seed: int,
    soft_sigma: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute prior label distribution on the validation subset.

    Split strategy matches training:
      val_len = int(N * val_ratio)
      val selected by random permutation with `seed`
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0,1)")
    n = len(scores)
    if n < 2:
        raise RuntimeError("Manifest needs at least 2 samples to build prior.")

    val_len = int(n * val_ratio)
    if val_len < 1:
        raise RuntimeError(
            f"val_ratio={val_ratio} produces val_len={val_len} < 1; cannot compute prior."
        )
    train_len = n - val_len

    if device is None:
        device = torch.device("cpu")

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    val_idx = perm[train_len:]

    prior = torch.zeros(num_classes, dtype=torch.float32, device=device)
    sigma = float(soft_sigma)

    # Fast path: one-hot (soft_sigma==0) at nearest integer bin.
    if sigma <= 0.0:
        for i in val_idx.tolist():
            s = float(scores[i])
            bin_ = int(round(s)) - 1  # bins are 1..K
            bin_ = max(0, min(num_classes - 1, bin_))
            prior[bin_] += 1.0
    else:
        for i in val_idx.tolist():
            s = float(scores[i])
            prior += score_to_distribution(
                s,
                num_classes,
                one_based=True,
                sigma=sigma,
            ).to(device=device, dtype=torch.float32)

    prior = prior / float(val_len)
    return prior


def _topk_bins(probs_1d: torch.Tensor, k: int) -> list[tuple[int, float]]:
    """
    probs_1d: Tensor (K,) for bins 1..K
    returns [(bin(1..K), prob), ...] sorted by prob desc
    """
    k = int(k)
    k = max(1, min(k, probs_1d.shape[0]))
    vals, inds = torch.topk(probs_1d, k=k, largest=True, sorted=True)
    return [(int(i) + 1, float(v)) for v, i in zip(vals.tolist(), inds.tolist())]


def _preprocess_image_to_tensor(
    image_path: str | Path,
    image_size: tuple[int, int] = (224, 224),
) -> tuple[torch.Tensor, Image.Image, tuple[int, int]]:
    """
    Convert an image to a normalized tensor (1,3,H,W).

    Returns:
      - image_tensor: torch.FloatTensor (1,3,H,W)
      - pil_rgb: resized PIL image (used only to get resized size)
      - orig_wh: original (W,H)
    """
    pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil.size

    resized = pil.resize(image_size, resample=Image.BILINEAR)
    rgb = np.array(resized, dtype=np.float32) / 255.0  # (H,W,3)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))  # (3,H,W)
    image_tensor = torch.from_numpy(chw).unsqueeze(0).contiguous()  # (1,3,H,W)
    return image_tensor, resized, (orig_w, orig_h)


def _scale_landmarks_to_resized(
    lm_xy: np.ndarray,
    orig_wh: tuple[int, int],
    resized_wh: tuple[int, int],
) -> np.ndarray:
    """
    Scale landmarks (x,y) from original image coordinates to resized image coordinates.
    """
    if lm_xy.shape != (68, 2):
        raise ValueError(f"Expected landmarks shape (68,2), got {lm_xy.shape}")

    orig_w, orig_h = orig_wh
    new_w, new_h = resized_wh
    if orig_w <= 0 or orig_h <= 0 or new_w <= 0 or new_h <= 0:
        raise ValueError("Invalid image size for landmark scaling")

    lm = lm_xy.astype(np.float32).copy()
    lm[:, 0] *= float(new_w) / float(orig_w)
    lm[:, 1] *= float(new_h) / float(orig_h)
    return lm


def _load_checkpoint_state_dict(weights_path: str | Path) -> dict[str, Any]:
    ckpt_path = Path(weights_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"weights not found: {ckpt_path}")
    obj = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        # Assume it's already a state_dict.
        return obj
    raise ValueError("Unsupported checkpoint format: expected a dict or dict with model_state_dict")


@torch.no_grad()
def predict_score_png(
    image_path: str | Path,
    *,
    weights_path: str | Path,
    num_classes: int = 100,
    hidden_dim: int = 256,
    backbone: str = "mobilenet_v3_small",
    pretrained_backbone: bool = True,
    device: str | torch.device | None = None,
    image_size: tuple[int, int] = (224, 224),
) -> dict[str, Any]:
    """
    Direct interface: input image -> output predicted score.

    Returns a dict with:
      - score_hat: float in 1..K (expectation of LDL distribution)
      - logits/probs: tensors (optional fields can be removed later)
    """
    if face_alignment is None:
        raise RuntimeError(
            "face_alignment is not available. Please install it first: pip install face-alignment\n"
            f"Original import error: {_face_alignment_import_error}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 1) Build model
    model = DualBranchFBPModel(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        backbone=backbone,  # "mobilenet_v3_small" or "resnet18"
        pretrained=pretrained_backbone,
    ).to(device)

    # 2) Load trained weights
    state_dict = _load_checkpoint_state_dict(weights_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] missing keys when loading weights: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(
            f"[WARN] unexpected keys when loading weights: {unexpected[:10]}"
            f"{'...' if len(unexpected) > 10 else ''}"
        )
    model.eval()

    # 3) Preprocess image
    image_tensor, resized_pil, orig_wh = _preprocess_image_to_tensor(
        image_path=image_path,
        image_size=image_size,
    )
    image_tensor = image_tensor.to(device=device)
    resized_wh = resized_pil.size  # (W,H)

    # 4) Face landmarks via face_alignment
    # face_alignment expects RGB uint8 array.
    rgb_uint8 = np.array(resized_pil, dtype=np.uint8)
    # Note: we run landmark detection on the resized image to reduce compute.
    # Then we scale back to the resized coordinate system (no-op because it's already resized).
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device="cpu",  # keep it stable for Windows; change if you want
    )
    preds = fa.get_landmarks(rgb_uint8)
    if preds is None or len(preds) == 0:
        raise RuntimeError("face_alignment: no face detected")
    lm = preds[0].astype(np.float32)  # (68,2) in resized image coords

    # Ensure shape
    if lm.shape[0] != 68 or lm.shape[1] != 2:
        raise ValueError(f"face_alignment returned landmarks of shape {lm.shape}, expected (68,2)")

    # Because we detect on resized_pil, scaling from resized to resized is identity.
    # But keep scaling code for correctness if you change detection input later.
    lm_scaled = _scale_landmarks_to_resized(lm, orig_wh=resized_wh, resized_wh=resized_wh)
    landmarks = torch.from_numpy(lm_scaled).float().unsqueeze(0).to(device=device)  # (1,68,2)

    # 5) Forward
    out = model(images=image_tensor, landmarks=landmarks)
    score_hat = float(out.score_hat[0].detach().cpu().item())

    return {
        "score_hat": score_hat,
        "logits": out.logits.detach().cpu(),
        "probs": out.probs.detach().cpu(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FBP (Dual-Branch) inference: PNG -> score")
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Path to png/jpg image. If empty, script scans --input-dir for the latest image.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "input"),
        help="Directory used when --image is empty. The script picks the newest png/jpg/jpeg inside it.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to trained model checkpoint (.pt/.pth). If empty, auto-pick checkpoints_gpu/best.pth.",
    )
    parser.add_argument("--num-classes", type=int, default=100, help="K bins for LDL; 100 for 1..100 scale")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dim used in training")
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v3_small",
        choices=("mobilenet_v3_small", "resnet18"),
        help="Backbone used in training",
    )
    parser.add_argument("--no-pretrained-backbone", action="store_true", help="Disable ImageNet pretrained backbone")
    parser.add_argument("--device", type=str, default="", help="e.g. cpu or cuda:0 (empty auto)")

    # Human-population prior (from manifest) for comparing where the score sits.
    parser.add_argument(
        "--prior-manifest",
        type=str,
        default="data/scut_train_fold1.csv",
        help="Manifest CSV used to compute label prior distribution on the validation split.",
    )
    parser.add_argument(
        "--prior-val-ratio",
        type=float,
        default=0.1,
        help="val-ratio used when building the prior distribution.",
    )
    parser.add_argument(
        "--prior-seed",
        type=int,
        default=42,
        help="seed used when splitting manifest into train/val for prior computation.",
    )
    parser.add_argument(
        "--prior-soft-sigma",
        type=float,
        default=0.0,
        help="If >0, build prior using Gaussian soft labels; if 0, one-hot prior.",
    )
    parser.add_argument(
        "--prior-topk",
        type=int,
        default=5,
        help="How many bins to show for prior top-k.",
    )
    parser.add_argument(
        "--model-topk",
        type=int,
        default=5,
        help="How many bins to show for model top-k.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device: str | torch.device | None
    if args.device.strip():
        device = args.device.strip()
    else:
        device = None

    # Default weights path: prefer checkpoints_gpu/best.pth, then checkpoints/best.pth.
    weights_path: str
    if args.weights.strip():
        weights_path = args.weights.strip()
    else:
        repo_root = Path(__file__).resolve().parent
        # infer_fbp_api.py lives at project root; checkpoints_* are under same root.
        cand1 = repo_root / "checkpoints_gpu" / "best.pth"
        cand2 = repo_root / "checkpoints" / "best.pth"
        if cand1.is_file():
            weights_path = str(cand1)
        elif cand2.is_file():
            weights_path = str(cand2)
        else:
            raise FileNotFoundError(
                "Cannot auto-find default checkpoint. Please pass --weights explicitly.\n"
                f"Tried: {cand1} and {cand2}"
            )

    # Pick input image:
    # - If args.image provided: use it
    # - Else: scan args.input_dir for newest png/jpg/jpeg and use it
    if args.image.strip():
        image_path = args.image.strip()
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"--input-dir not found or not a directory: {input_dir}")
        candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
            candidates.extend(input_dir.glob(ext))
        if not candidates:
            raise FileNotFoundError(
                f"No png/jpg/jpeg found under {input_dir}. "
                "Copy an image there and re-run (no need to rename)."
            )
        image_path_obj = max(candidates, key=lambda p: p.stat().st_mtime)
        image_path = str(image_path_obj)
        print(f"[INFO] Auto-selected newest image: {image_path_obj.name}")

    result = predict_score_png(
        image_path,
        weights_path=weights_path,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        backbone=args.backbone,
        pretrained_backbone=not args.no_pretrained_backbone,
        device=device,
    )
    score_hat = float(result["score_hat"])
    probs_np = result["probs"].numpy()  # (1, K)
    probs_1d = torch.from_numpy(probs_np[0]).float()  # (K,)

    topk_model = _topk_bins(probs_1d, args.model_topk)
    top_idx = int(torch.argmax(probs_1d).item())
    top_score = top_idx + 1  # bins are 1..K

    # 1) Model output (score)
    print(f"Expected score_hat: {score_hat:.3f} (discrete bin={top_score})")
    print(
        f"Model output top-{len(topk_model)} bins by probability: "
        + ", ".join([f"bin={b} prob={p:.4f}" for b, p in topk_model])
    )

    # 2) 人群先验对比（val split）
    mu_bin = int(round(score_hat))
    mu_bin = max(1, min(int(args.num_classes), mu_bin))
    mu_bin0 = mu_bin - 1

    prior_manifest = args.prior_manifest.strip()
    if prior_manifest:
        try:
            scores = _read_scores_from_manifest(prior_manifest, score_col="score")
            prior_val = _compute_prior_val_distribution(
                scores,
                num_classes=int(args.num_classes),
                val_ratio=float(args.prior_val_ratio),
                seed=int(args.prior_seed),
                soft_sigma=float(args.prior_soft_sigma),
                device=torch.device("cpu"),
            )

            p_bin = float(prior_val[mu_bin0].item())
            cdf_le = float(prior_val[: mu_bin0 + 1].sum().item())  # P(score<=bin)
            p_ge = float(prior_val[mu_bin0:].sum().item())  # P(score>=bin)

            cdf = torch.cumsum(prior_val, dim=0)
            nonzero = (cdf >= 0.5).nonzero(as_tuple=False)
            if nonzero.numel() == 0:
                median_bin = 50
            else:
                median_bin0 = int(nonzero[0].item())
                median_bin = median_bin0 + 1
            high_or_low = "high" if mu_bin >= median_bin else "low"

            topk_prior = _topk_bins(prior_val, int(args.prior_topk))

            print(
                f"Human prior (Val) probability at bin={mu_bin}: {p_bin*100:.2f}% "
                f"(CDF<=bin: {cdf_le*100:.2f}%, P(score>=bin): {p_ge*100:.2f}%)"
            )
            print(
                "Position within the prior distribution "
                f"(defined as 'score>=bin' = top {p_ge*100:.2f}%): "
                f"top{p_ge*100:.2f}% (relative to prior median bin={median_bin}, {high_or_low})"
            )
            print(
                f"Prior top-{len(topk_prior)} bins by probability: "
                + ", ".join([f"bin={b} prob={p:.4f}" for b, p in topk_prior])
            )
        except Exception as e:  # pragma: no cover
            print(f"[WARN] prior computation skipped: {e}")


if __name__ == "__main__":
    main()

