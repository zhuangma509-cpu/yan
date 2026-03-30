"""
Convert scalar beauty scores to LDL soft label distributions (K-way).
"""

from __future__ import annotations

import torch


def score_to_distribution(
    score: float,
    num_classes: int,
    *,
    one_based: bool = True,
    sigma: float = 0.0,
) -> torch.Tensor:
    """
    Map a scalar score to a probability vector of length K.

    - one_based=True: valid scores are in [1, num_classes] (e.g. 1..100).
    - one_based=False: valid scores are in [0, num_classes - 1].

    If sigma <= 0: one-hot at the nearest class index (after clamping).
    If sigma > 0: Gaussian weights over class indices (1..K or 0..K-1 per one_based),
      then normalized to sum to 1.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    if sigma <= 0:
        y = torch.zeros(num_classes, dtype=torch.float32)
        if one_based:
            idx = int(round(score)) - 1
        else:
            idx = int(round(score))
        idx = max(0, min(num_classes - 1, idx))
        y[idx] = 1.0
        return y

    if one_based:
        centers = torch.arange(1, num_classes + 1, dtype=torch.float32)
        s = torch.tensor(float(score), dtype=torch.float32)
    else:
        centers = torch.arange(0, num_classes, dtype=torch.float32)
        s = torch.tensor(float(score), dtype=torch.float32)

    w = torch.exp(-0.5 * ((centers - s) / float(sigma)) ** 2)
    w = w / (w.sum() + 1e-12)
    return w
