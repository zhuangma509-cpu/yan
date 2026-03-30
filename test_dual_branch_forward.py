"""
Forward-pass smoke test for DualBranchFBPModel (random image + landmark tensors).
Run from project root: python test_dual_branch_forward.py
"""

from __future__ import annotations

from models.dual_branch_fbp import run_dual_branch_forward_test


def main() -> None:
    run_dual_branch_forward_test(
        batch_size=4,
        height=224,
        width=224,
        hidden_dim=256,
        num_classes=100,
        backbone="mobilenet_v3_small",
    )


if __name__ == "__main__":
    main()
