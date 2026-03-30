"""
Helpers for SCUT-FBP5500 official split files (e.g. data/1/train_1.txt).

Official format (per line): "<image_filename> <mean_score>"
Mean score is the average of raters on a 1–5 scale (may be fractional).

See: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
Images themselves come from the dataset archive (Google Drive / Baidu), not this repo.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

_LINE_RE = re.compile(
    r"^\s*(?P<fname>\S+)\s+(?P<score>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def parse_split_file_line(line: str) -> tuple[str, float] | None:
    """
    Parse one non-empty line from train_*.txt / test_*.txt.
    Returns (filename, mean_score) or None if comment/blank.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    m = _LINE_RE.match(s)
    if m is None:
        raise ValueError(f"Unrecognized SCUT split line: {line!r}")
    return m.group("fname"), float(m.group("score"))


def map_scut_score_to_training_scale(
    mean_score_1_to_5: float,
    mode: Literal["round_1_to_5", "linear_1_to_100"],
) -> float:
    """
    Map dataset mean (typically in [1,5]) to the scale used by FBPLDLDataset.

    - round_1_to_5: round to nearest integer in [1, 5] (use --num-classes 5).
    - linear_1_to_100: linear map [1,5] -> [1,100], then round to integer in [1,100]
      (use --num-classes 100).
    """
    if mode == "round_1_to_5":
        v = round(mean_score_1_to_5)
        return float(max(1, min(5, int(v))))
    if mode == "linear_1_to_100":
        x = float(mean_score_1_to_5)
        x = max(1.0, min(5.0, x))
        m = (x - 1.0) / 4.0 * 99.0 + 1.0
        return float(max(1, min(100, int(round(m)))))
    raise ValueError(f"Unknown mode: {mode}")


def iter_scut_split_file(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        p = parse_split_file_line(line)
        if p is not None:
            rows.append(p)
    return rows


def resolve_image_path(dataset_root: Path, filename: str) -> Path:
    """Join dataset_root with filename; filename is usually flat (e.g. mty152.jpg)."""
    p = Path(filename)
    if p.is_absolute():
        return p
    return dataset_root / p
