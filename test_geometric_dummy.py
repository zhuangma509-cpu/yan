"""
Dummy test: random 68x2 landmarks in plausible face-like layout, run GeometricPriorExtractor.
"""

from __future__ import annotations

import numpy as np

from utils.geometric_prior import GeometricPriorExtractor


def make_dummy_landmarks_68(seed: int = 42) -> np.ndarray:
    """
    Synthetic 68 points in dlib order: rough frontal face, y downward.
    Not a real face; only for pipeline smoke test.
    """
    rng = np.random.default_rng(seed)
    P = np.zeros((68, 2), dtype=np.float64)
    cx = 128.0
    face_w = 100.0 + rng.uniform(-5.0, 5.0)
    x_left = cx - face_w * 0.5
    x_right = cx + face_w * 0.5

    # Jaw 0–16: U-shaped, chin at 8
    for i in range(17):
        t = i / 16.0
        ang = np.pi * (1.0 - t)
        px = cx + (face_w * 0.48) * np.cos(ang)
        py = 200.0 + (face_w * 0.55) * np.sin(ang) + rng.normal(0.0, 0.5)
        P[i, 0] = px
        P[i, 1] = py

    # Brows 17–26
    for i in range(17, 27):
        u = (i - 17) / 9.0
        P[i, 0] = x_left + 15.0 + u * (face_w - 30.0)
        P[i, 1] = 95.0 + rng.normal(0.0, 0.3)

    # Nose 27–35
    for i in range(27, 36):
        P[i, 0] = cx + rng.normal(0.0, 0.4)
        P[i, 1] = 100.0 + (i - 27) * 8.0 + rng.normal(0.0, 0.2)

    # Left eye 36–41 (image left = smaller x)
    eye_y = 110.0
    P[36, :] = [cx - 28.0, eye_y]
    P[37, :] = [cx - 24.0, eye_y - 2.0]
    P[38, :] = [cx - 22.0, eye_y]
    P[39, :] = [cx - 18.0, eye_y]
    P[40, :] = [cx - 24.0, eye_y + 2.0]
    P[41, :] = [cx - 26.0, eye_y + 1.0]

    # Right eye 42–47
    P[42, :] = [cx + 18.0, eye_y]
    P[43, :] = [cx + 22.0, eye_y]
    P[44, :] = [cx + 24.0, eye_y - 2.0]
    P[45, :] = [cx + 28.0, eye_y]
    P[46, :] = [cx + 24.0, eye_y + 2.0]
    P[47, :] = [cx + 22.0, eye_y + 1.0]

    # Mouth 48–67: simple arc
    for i in range(48, 68):
        u = (i - 48) / 19.0
        P[i, 0] = cx - 22.0 + u * 44.0
        P[i, 1] = 175.0 + 8.0 * np.sin(np.pi * u) + rng.normal(0.0, 0.2)

    P += rng.normal(0.0, 0.15, size=P.shape)
    return P


def main() -> None:
    ext = GeometricPriorExtractor()
    P = make_dummy_landmarks_68()
    v_np = ext.extract(P, return_tensor=False)
    assert v_np.shape == (ext.output_dim,), v_np.shape
    print("NumPy V_geo shape:", v_np.shape)
    print("V_geo:", v_np)

    v_t = ext.extract(P, return_tensor=True)
    assert tuple(v_t.shape) == (ext.output_dim,)
    print("Torch V_geo shape:", tuple(v_t.shape))
    print("dtype:", v_t.dtype)

    print("OK: GeometricPriorExtractor runs on dummy 68 landmarks.")


if __name__ == "__main__":
    main()
