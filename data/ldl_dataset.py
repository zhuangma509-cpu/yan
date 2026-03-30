"""
Dataset for LDL-style facial beauty training: images, landmarks, soft label distributions.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class FBPLDLDataset(Dataset):
    """
    Tensor-backed dataset for FBP with label distributions.

    Each sample:
      - image: float tensor (3, H, W)
      - landmarks: float tensor (68, 2)
      - target_dist: float tensor (K,) with non-negative entries summing to 1
    """

    def __init__(
        self,
        images: torch.Tensor,
        landmarks: torch.Tensor,
        target_distributions: torch.Tensor,
    ) -> None:
        super().__init__()
        if images.shape[0] != landmarks.shape[0] or images.shape[0] != target_distributions.shape[0]:
            raise ValueError(
                "images, landmarks, and target_distributions must have the same batch dimension N"
            )
        if landmarks.dim() != 3 or landmarks.shape[1] != 68 or landmarks.shape[2] != 2:
            raise ValueError(
                f"landmarks must be (N, 68, 2), got {tuple(landmarks.shape)}"
            )
        if target_distributions.dim() != 2:
            raise ValueError(
                f"target_distributions must be (N, K), got {tuple(target_distributions.shape)}"
            )
        self._images = images
        self._landmarks = landmarks
        self._target_distributions = target_distributions

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": self._images[idx].clone(),
            "landmarks": self._landmarks[idx].clone(),
            "target_dist": self._target_distributions[idx].clone(),
        }

    @property
    def num_classes(self) -> int:
        return int(self._target_distributions.shape[1])
