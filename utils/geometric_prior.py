"""
Geometric prior extraction for FBP based on facial landmarks.

Uses the common dlib/OpenCV 68-point facial landmark layout unless noted.
106-point layouts can be supported by remapping indices upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# dlib 68-point layout (0-based indices)
# Jaw: 0–16 (8 = chin tip). Brows: 17–26. Nose: 27–35. Left eye: 36–41. Right: 42–47. Mouth: 48–67

IDX_CHIN_TIP = 8
IDX_BROW_LEFT_OUTER = 17
IDX_BROW_LEFT_INNER = 21
IDX_BROW_RIGHT_INNER = 22
IDX_BROW_RIGHT_OUTER = 26
IDX_NOSE_TIP = 33
IDX_LE_OUTER = 36
IDX_LE_INNER = 39
IDX_RE_INNER = 42
IDX_RE_OUTER = 45


@dataclass(frozen=True)
class GeometricPriorConfig:
    """Hyperparameters for hairline estimation and deviation penalties."""

    hairline_scale: float = 0.38
    eps: float = 1e-8
    ideal_court: tuple[float, float, float] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    ideal_fifth: float = 0.2


class GeometricPriorExtractor:
    """
    Extract geometry-based features from 2D facial landmarks (68-point dlib order by default).

    Outputs a fixed-length vector V_geo containing three-court features, S_courts,
    five-eye-related features, and S_eyes.
    """

    def __init__(
        self,
        config: GeometricPriorConfig | None = None,
        landmark_layout: Literal["dlib68"] = "dlib68",
    ) -> None:
        self.config = config or GeometricPriorConfig()
        self.landmark_layout = landmark_layout

    @staticmethod
    def _to_numpy(points: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        if hasattr(points, "detach"):
            return points.detach().cpu().numpy().astype(np.float64)
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Expected landmarks of shape (N, 2), got {arr.shape}"
            )
        return arr

    def _estimate_hairline_y(self, P: np.ndarray) -> float:
        """Estimate hairline vertical position (smaller y = higher on image)."""
        brow_ys = P[IDX_BROW_LEFT_OUTER : IDX_BROW_RIGHT_OUTER + 1, 1]
        y_brow_top = float(np.min(brow_ys))
        y_chin = float(P[IDX_CHIN_TIP, 1])
        face_h = max(y_chin - y_brow_top, self.config.eps)
        y_hairline = y_brow_top - self.config.hairline_scale * face_h
        return y_hairline

    def _glabella(self, P: np.ndarray) -> np.ndarray:
        return (P[IDX_BROW_LEFT_INNER] + P[IDX_BROW_RIGHT_INNER]) * 0.5

    def _nose_base_y(self, P: np.ndarray) -> float:
        """鼻底: use nose tip (landmark 33) as stable midline lower nose reference."""
        return float(P[IDX_NOSE_TIP, 1])

    def _three_courts(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        y_hair = self._estimate_hairline_y(P)
        gb = self._glabella(P)
        y_gb = float(gb[1])
        y_nb = self._nose_base_y(P)
        y_chin = float(P[IDX_CHIN_TIP, 1])

        H1 = y_gb - y_hair
        H2 = y_nb - y_gb
        H3 = y_chin - y_nb
        H_raw = np.array([H1, H2, H3], dtype=np.float64)
        H_sum = float(np.sum(np.maximum(H_raw, 0.0))) + self.config.eps
        r = np.maximum(H_raw, 0.0) / H_sum

        ideal = np.array(self.config.ideal_court, dtype=np.float64)
        S_courts = float(np.linalg.norm(r - ideal))

        return H_raw, r, S_courts

    def _face_width_and_five_segments(
        self, P: np.ndarray
    ) -> tuple[float, np.ndarray, float]:
        xs = P[0:17, 0]
        x_left = float(np.min(xs))
        x_right = float(np.max(xs))
        W = x_right - x_left
        W = max(W, self.config.eps)

        x36 = float(P[IDX_LE_OUTER, 0])
        x39 = float(P[IDX_LE_INNER, 0])
        x42 = float(P[IDX_RE_INNER, 0])
        x45 = float(P[IDX_RE_OUTER, 0])

        w_L = abs(x39 - x36)
        w_R = abs(x45 - x42)
        d = abs(x42 - x39)

        s1 = max(x36 - x_left, 0.0)
        s5 = max(x_right - x45, 0.0)

        segments = np.array([s1, w_L, d, w_R, s5], dtype=np.float64)
        fifth = self.config.ideal_fifth * W
        resid = segments - fifth
        S_eyes = float(np.linalg.norm(resid) / (W + self.config.eps))

        return W, segments, S_eyes

    def extract(
        self,
        points: Union[np.ndarray, "torch.Tensor"],
        return_tensor: bool = False,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        P = self._to_numpy(points)
        if P.shape[0] < 68:
            raise ValueError(
                "GeometricPriorExtractor (dlib68) requires at least 68 landmarks."
            )

        H_raw, r_court, S_courts = self._three_courts(P)
        W, segments, S_eyes = self._face_width_and_five_segments(P)

        w_L = segments[1]
        w_R = segments[3]
        d = segments[2]

        fifth = self.config.ideal_fifth * W
        ratio_L = w_L / (fifth + self.config.eps)
        ratio_R = w_R / (fifth + self.config.eps)
        ratio_d = d / (fifth + self.config.eps)
        eye_width_sym = abs(w_L - w_R) / (W + self.config.eps)

        V_list = [
            H_raw[0],
            H_raw[1],
            H_raw[2],
            r_court[0],
            r_court[1],
            r_court[2],
            S_courts,
            W,
            w_L,
            w_R,
            d,
            segments[0],
            segments[4],
            ratio_L,
            ratio_R,
            ratio_d,
            eye_width_sym,
            S_eyes,
        ]

        V_geo = np.array(V_list, dtype=np.float64)

        if return_tensor:
            if torch is None:
                raise RuntimeError("PyTorch is not installed; cannot return tensor.")
            return torch.from_numpy(V_geo).float()

        return V_geo

    @property
    def output_dim(self) -> int:
        return 18

    def extract_batch(
        self,
        points: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Batch version of extract for landmarks of shape (B, 68, 2).
        Returns (B, output_dim) float tensor on the same device as input.
        """
        if torch is None:
            raise RuntimeError("PyTorch is not installed; cannot use extract_batch.")
        if points.dim() != 3 or points.shape[1] != 68 or points.shape[2] != 2:
            raise ValueError(
                f"Expected landmarks of shape (B, 68, 2), got {tuple(points.shape)}"
            )
        device = points.device
        dtype = points.dtype
        rows: list[torch.Tensor] = []
        for i in range(points.shape[0]):
            row = self.extract(points[i], return_tensor=True)
            rows.append(row.to(device=device, dtype=dtype))
        return torch.stack(rows, dim=0)


# Fixed size of V_geo for downstream networks (dual-branch MLP input dim).
GEO_FEATURE_DIM = 18

# Indices into V_geo for geometric regularization (see extract() V_list order).
V_GEO_IDX_S_COURTS = 6
V_GEO_IDX_S_EYES = 17


def compute_geometric_vector(
    points: Union[np.ndarray, "torch.Tensor"],
    config: GeometricPriorConfig | None = None,
    return_tensor: bool = False,
) -> Union[np.ndarray, "torch.Tensor"]:
    ext = GeometricPriorExtractor(config=config)
    return ext.extract(points, return_tensor=return_tensor)
