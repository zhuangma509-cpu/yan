"""
Label Distribution Learning (LDL) composite loss for Facial Beauty Prediction:
KL divergence, expectation MSE, and geometric regularization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometric_prior import V_GEO_IDX_S_COURTS, V_GEO_IDX_S_EYES


@dataclass
class FBPLossBreakdown:
    """Per-term losses (scalar values after reduction over the batch)."""

    loss_kl: float
    loss_exp: float
    loss_geo: float
    loss_total: float


class FBPLoss(nn.Module):
    """
    Total loss: w_kl * L_KL + w_exp * L_exp + w_geo * L_geo.

    - L_KL: KL( y || softmax(logits) ) with soft target distribution y (batchmean).
    - L_exp: MSE between predicted expectation μ̂ and μ = sum_k y_k * k (levels 1..K).
    - L_geo: mean over batch of ReLU(μ̂ - τ) * (a * S_courts + b * S_eyes); τ defaults to 85 on a 1–100 scale.
    """

    def __init__(
        self,
        num_classes: int = 100,
        w_kl: float = 1.0,
        w_exp: float = 1.0,
        w_geo: float = 0.1,
        score_threshold: float = 85.0,
        geo_w_courts: float = 1.0,
        geo_w_eyes: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = num_classes
        self.w_kl = w_kl
        self.w_exp = w_exp
        self.w_geo = w_geo
        self.score_threshold = score_threshold
        self.geo_w_courts = geo_w_courts
        self.geo_w_eyes = geo_w_eyes
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        target_dist: torch.Tensor,
        score_hat: torch.Tensor,
        v_geo: torch.Tensor,
    ) -> tuple[torch.Tensor, FBPLossBreakdown]:
        """
        Args:
            logits: (B, K) raw classifier logits.
            target_dist: (B, K) ground-truth label distribution (non-negative, sums to 1 per row).
            score_hat: (B,) predicted expectation score from the model (levels 1..K).
            v_geo: (B, D_geo) geometric vector; S_courts and S_eyes are read at fixed indices.
        """
        if logits.dim() != 2 or logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"logits must be (B, {self.num_classes}), got {tuple(logits.shape)}"
            )
        if target_dist.shape != logits.shape:
            raise ValueError(
                f"target_dist shape {tuple(target_dist.shape)} must match logits"
            )
        if score_hat.dim() != 1 or score_hat.shape[0] != logits.shape[0]:
            raise ValueError(
                f"score_hat must be (B,) with B={logits.shape[0]}, got {tuple(score_hat.shape)}"
            )
        if v_geo.dim() != 2 or v_geo.shape[0] != logits.shape[0]:
            raise ValueError(
                f"v_geo must be (B, D_geo), got {tuple(v_geo.shape)}"
            )

        device = logits.device
        dtype = logits.dtype

        target_safe = target_dist.clamp(min=self.eps)
        target_safe = target_safe / target_safe.sum(dim=-1, keepdim=True)

        log_p = F.log_softmax(logits, dim=-1)
        loss_kl = F.kl_div(
            log_p,
            target_safe,
            reduction="batchmean",
            log_target=False,
        )

        levels = torch.arange(
            1,
            self.num_classes + 1,
            device=device,
            dtype=dtype,
        )
        mu_true = torch.sum(target_safe * levels, dim=-1)
        loss_exp = F.mse_loss(score_hat, mu_true)

        s_courts = v_geo[:, V_GEO_IDX_S_COURTS]
        s_eyes = v_geo[:, V_GEO_IDX_S_EYES]
        gate = F.relu(score_hat - self.score_threshold)
        geo_linear = self.geo_w_courts * s_courts + self.geo_w_eyes * s_eyes
        loss_geo = torch.mean(gate * geo_linear)

        loss_total = (
            self.w_kl * loss_kl + self.w_exp * loss_exp + self.w_geo * loss_geo
        )

        breakdown = FBPLossBreakdown(
            loss_kl=float(loss_kl.detach().item()),
            loss_exp=float(loss_exp.detach().item()),
            loss_geo=float(loss_geo.detach().item()),
            loss_total=float(loss_total.detach().item()),
        )

        return loss_total, breakdown
