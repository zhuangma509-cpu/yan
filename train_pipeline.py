"""
FBP training entry: LDL composite loss, DataLoader, and train_epoch loop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from data.ldl_dataset import FBPLDLDataset
from data.manifest_dataset import FBPImageScoreManifestDataset
from models.dual_branch_fbp import DualBranchFBPModel
from utils.fbp_loss import FBPLoss
from utils.training import train_epoch, eval_epoch


def _make_synthetic_batch(
    num_samples: int,
    num_classes: int,
    height: int,
    width: int,
    seed: int,
) -> FBPLDLDataset:
    """Build a small in-memory dataset for integration testing."""
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(num_samples, 3, height, width, generator=g)
    landmarks = torch.zeros(num_samples, 68, 2)
    cx, cy = width * 0.5, height * 0.45
    t = torch.linspace(0, 1, 68)
    for i in range(num_samples):
        phase = torch.rand(1, generator=g).item() * 6.28
        landmarks[i, :, 0] = cx + 35.0 * torch.sin(t * 6.28 + phase)
        landmarks[i, :, 1] = cy + 45.0 * torch.cos(t * 3.14 + phase)
        landmarks[i] += 0.5 * torch.randn(68, 2, generator=g)

    alpha = torch.rand(num_samples, num_classes, generator=g) + 0.2
    target_distributions = alpha / alpha.sum(dim=-1, keepdim=True)

    return FBPLDLDataset(images, landmarks, target_distributions)


def main() -> None:
    parser = argparse.ArgumentParser(description="FBP LDL training pipeline")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=100,
        help="LDL bins (1..K scores). Default 100 for 1–100 scale.",
    )
    parser.add_argument("--w-kl", type=float, default=1.0)
    parser.add_argument("--w-exp", type=float, default=1.0)
    parser.add_argument("--w-geo", type=float, default=0.1)
    parser.add_argument(
        "--geo-threshold",
        type=float,
        default=85.0,
        help="Geometric regularization: penalize high predicted score when S_courts/S_eyes are large (1–100 scale).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of dataset used for validation (0~1). Used to pick best checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save last.pth and best.pth checkpoints.",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Random-init backbone")
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        help='CSV path,score (see data/manifest_dataset.py). SCUT-FBP5500: scripts/build_scut_fbp5500_manifest.py',
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="Optional root directory prepended to relative paths in manifest CSV.",
    )
    parser.add_argument(
        "--landmark-dir",
        type=str,
        default="",
        help='Directory of .npy files (68,2) per image stem; required when --landmark-mode=npy.',
    )
    parser.add_argument(
        "--landmark-mode",
        type=str,
        choices=("npy", "face_alignment"),
        default="npy",
        help='npy: load landmark_dir/<stem>.npy. face_alignment: pip install face_alignment; use DataLoader num_workers=0.',
    )
    parser.add_argument(
        "--soft-sigma",
        type=float,
        default=0.0,
        help="If >0, Gaussian soft labels around score; if 0, one-hot to nearest class.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 with landmark_mode=face_alignment.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height, width = 224, 224

    use_manifest = args.manifest.strip() != ""

    if use_manifest:
        image_root = args.image_root.strip() or None
        landmark_dir = args.landmark_dir.strip() or None
        if args.landmark_mode == "npy" and not landmark_dir:
            raise SystemExit(
                "With --landmark-mode npy, you must set --landmark-dir to a folder of "
                "<image_stem>.npy arrays with shape (68, 2) in original image pixel coordinates."
            )
        dataset = FBPImageScoreManifestDataset(
            manifest_csv=args.manifest,
            image_root=image_root,
            num_classes=args.num_classes,
            one_based_score=True,
            soft_sigma=args.soft_sigma,
            image_size=(height, width),
            landmark_mode=args.landmark_mode,
            landmark_dir=landmark_dir,
        )
    else:
        dataset = _make_synthetic_batch(
            num_samples=args.num_samples,
            num_classes=args.num_classes,
            height=height,
            width=width,
            seed=0,
        )

    # Train/val split for selecting the best checkpoint.
    dataset_len = len(dataset)
    val_len = int(dataset_len * args.val_ratio)
    train_len = dataset_len - val_len
    if val_len < 1:
        train_len = dataset_len
        val_len = 0

    g = torch.Generator().manual_seed(args.seed)
    if val_len > 0:
        train_dataset, val_dataset = random_split(
            dataset, [train_len, val_len], generator=g
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
    else:
        val_loader = None

    model = DualBranchFBPModel(
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        backbone="mobilenet_v3_small",
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = FBPLoss(
        num_classes=args.num_classes,
        w_kl=args.w_kl,
        w_exp=args.w_exp,
        w_geo=args.w_geo,
        score_threshold=args.geo_threshold,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last.pth"
    best_path = ckpt_dir / "best.pth"

    best_val_loss_total = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        if val_loader is not None:
            val_metrics = eval_epoch(
                model, val_loader, criterion, device
            )
        else:
            # If no val split, track best on training metrics.
            val_metrics = train_metrics

        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"train_loss_total={train_metrics['loss_total']:.6f}  "
            f"val_loss_total={val_metrics['loss_total']:.6f}  "
            f"kl={train_metrics['loss_kl']:.6f}  "
            f"exp={train_metrics['loss_exp']:.6f}  "
            f"geo={train_metrics['loss_geo']:.6f}"
        )

        # Save last each epoch
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch + 1},
            str(last_path),
        )

        # Update best by validation loss_total (smaller is better).
        if val_metrics["loss_total"] < best_val_loss_total:
            best_val_loss_total = val_metrics["loss_total"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": best_epoch,
                },
                str(best_path),
            )
            print(
                f"  [BEST] epoch={best_epoch} val_loss_total={best_val_loss_total:.6f}"
            )

    # Load best checkpoint at the end for inference-ready weights.
    if best_path.is_file():
        best_ckpt = torch.load(str(best_path), map_location=device)
        state_dict = (
            best_ckpt["model_state_dict"]
            if isinstance(best_ckpt, dict) and "model_state_dict" in best_ckpt
            else best_ckpt
        )
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded best checkpoint from epoch={best_epoch}")
    else:
        print("Warning: best checkpoint not found; keeping last model weights.")


if __name__ == "__main__":
    main()
