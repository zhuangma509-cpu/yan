"""
Training utilities: train_epoch with standard optimization steps.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from models.dual_branch_fbp import DualBranchFBPModel

from .fbp_loss import FBPLoss, FBPLossBreakdown


def train_epoch(
    model: DualBranchFBPModel,
    loader: DataLoader,
    criterion: FBPLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """
    One full pass over the training loader.

    For each batch: zero_grad -> forward -> loss -> backward -> optimizer.step.

    Returns mean metrics over batches: loss_total, loss_kl, loss_exp, loss_geo.
    """
    model.train()
    sum_total = 0.0
    sum_kl = 0.0
    sum_exp = 0.0
    sum_geo = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        landmarks = batch["landmarks"].to(device, non_blocking=True)
        target_dist = batch["target_dist"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(images, landmarks)
        loss, breakdown = criterion(
            logits=out.logits,
            target_dist=target_dist,
            score_hat=out.score_hat,
            v_geo=out.v_geo,
        )
        loss.backward()
        optimizer.step()

        sum_total += breakdown.loss_total
        sum_kl += breakdown.loss_kl
        sum_exp += breakdown.loss_exp
        sum_geo += breakdown.loss_geo
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("train_epoch received an empty DataLoader")

    inv = 1.0 / float(num_batches)
    return {
        "loss_total": sum_total * inv,
        "loss_kl": sum_kl * inv,
        "loss_exp": sum_exp * inv,
        "loss_geo": sum_geo * inv,
    }


@torch.no_grad()
def eval_epoch(
    model: DualBranchFBPModel,
    loader: DataLoader,
    criterion: FBPLoss,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluation pass (no gradients) over an epoch-sized DataLoader.

    Returns mean metrics over batches: loss_total, loss_kl, loss_exp, loss_geo.
    """
    model.eval()
    sum_total = 0.0
    sum_kl = 0.0
    sum_exp = 0.0
    sum_geo = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        landmarks = batch["landmarks"].to(device, non_blocking=True)
        target_dist = batch["target_dist"].to(device, non_blocking=True)

        out = model(images, landmarks)
        _, breakdown = criterion(
            logits=out.logits,
            target_dist=target_dist,
            score_hat=out.score_hat,
            v_geo=out.v_geo,
        )

        sum_total += breakdown.loss_total
        sum_kl += breakdown.loss_kl
        sum_exp += breakdown.loss_exp
        sum_geo += breakdown.loss_geo
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("eval_epoch received an empty DataLoader")

    inv = 1.0 / float(num_batches)
    return {
        "loss_total": sum_total * inv,
        "loss_kl": sum_kl * inv,
        "loss_exp": sum_exp * inv,
        "loss_geo": sum_geo * inv,
    }
