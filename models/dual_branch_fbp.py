"""
Dual-branch Facial Beauty Prediction: visual backbone + geometric MLP, fusion head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)

from utils.geometric_prior import GEO_FEATURE_DIM, GeometricPriorExtractor


@dataclass
class DualBranchFBPOutput:
    """Forward outputs of DualBranchFBPModel."""

    logits: torch.Tensor
    probs: torch.Tensor
    score_hat: torch.Tensor
    f_img: torch.Tensor
    f_geo: torch.Tensor
    v_geo: torch.Tensor


def _build_mobilenet_v3_small_backbone(
    pretrained: bool,
) -> tuple[nn.Module, nn.Module]:
    weights = (
        MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    )
    net = mobilenet_v3_small(weights=weights)
    return net.features, net.avgpool


def _build_resnet18_backbone(pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = resnet18(weights=weights)
    layers = list(net.children())[:-1]
    return nn.Sequential(*layers)


class DualBranchFBPModel(nn.Module):
    """
    Branch A: ImageNet-pretrained lightweight CNN -> global feature F_img in R^C.
    Branch B: MLP encodes V_geo -> F_geo in R^C.
    Fusion: concat(F_img, F_geo) -> Linear -> logits (K classes) -> softmax.
    Expected rating: sum_k p_k * k with levels 1..K (default K=100 for 1–100 scale).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 100,
        backbone: Literal["mobilenet_v3_small", "resnet18"] = "mobilenet_v3_small",
        pretrained: bool = True,
        geo_hidden_mult: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.geo_input_dim = GEO_FEATURE_DIM

        if backbone == "mobilenet_v3_small":
            self.features, self.avgpool = _build_mobilenet_v3_small_backbone(pretrained)
            self._is_resnet_style = False
        elif backbone == "resnet18":
            self.backbone = _build_resnet18_backbone(pretrained)
            self._is_resnet_style = True
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            if self._is_resnet_style:
                z = self.backbone(dummy)
                if z.dim() == 4:
                    z = torch.flatten(z, 1)
                visual_in_dim = int(z.shape[1])
            else:
                z = self.avgpool(self.features(dummy))
                visual_in_dim = int(torch.flatten(z, 1).shape[1])

        self.img_proj = nn.Linear(visual_in_dim, hidden_dim)

        mid = max(hidden_dim, geo_hidden_mult * self.geo_input_dim)
        self.geo_encoder = nn.Sequential(
            nn.Linear(self.geo_input_dim, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        fuse_in = hidden_dim * 2
        self.head = nn.Linear(fuse_in, num_classes)
        self._geo_extractor = GeometricPriorExtractor()

    def extract_visual(self, images: torch.Tensor) -> torch.Tensor:
        if self._is_resnet_style:
            x = self.backbone(images)
            if x.dim() == 4:
                x = torch.flatten(x, 1)
        else:
            x = self.features(images)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        return self.img_proj(x)

    def encode_geometry(self, v_geo: torch.Tensor) -> torch.Tensor:
        return self.geo_encoder(v_geo)

    def forward(
        self,
        images: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> DualBranchFBPOutput:
        """
        Args:
            images: (B, 3, H, W) — e.g. 224x224 RGB, same convention as torchvision.
            landmarks: (B, 68, 2) — dlib-order 2D points (same space as geometric_prior).
        """
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images (B, 3, H, W), got {tuple(images.shape)}")
        v_geo = self._geo_extractor.extract_batch(landmarks)
        f_img = self.extract_visual(images)
        f_geo = self.encode_geometry(v_geo)
        fused = torch.cat([f_img, f_geo], dim=-1)
        logits = self.head(fused)
        probs = F.softmax(logits, dim=-1)
        levels = torch.arange(
            1,
            self.num_classes + 1,
            device=logits.device,
            dtype=logits.dtype,
        )
        score_hat = torch.sum(probs * levels, dim=-1)
        return DualBranchFBPOutput(
            logits=logits,
            probs=probs,
            score_hat=score_hat,
            f_img=f_img,
            f_geo=f_geo,
            v_geo=v_geo,
        )


def run_dual_branch_forward_test(
    batch_size: int = 4,
    height: int = 224,
    width: int = 224,
    hidden_dim: int = 256,
    num_classes: int = 100,
    backbone: Literal["mobilenet_v3_small", "resnet18"] = "mobilenet_v3_small",
    device: str | torch.device | None = None,
    seed: int = 0,
) -> None:
    """
    Random image + landmark tensors; checks output shapes for DualBranchFBPModel.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    torch.manual_seed(seed)
    model = DualBranchFBPModel(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        backbone=backbone,
        pretrained=True,
    ).to(device)
    model.eval()

    images = torch.randn(batch_size, 3, height, width, device=device)
    landmarks = torch.zeros(batch_size, 68, 2, device=device)
    cx, cy = width * 0.5, height * 0.45
    t = torch.linspace(0, 1, 68, device=device)
    landmarks[:, :, 0] = cx + 40.0 * torch.sin(t * 6.28).unsqueeze(0).expand(batch_size, -1)
    landmarks[:, :, 1] = cy + 50.0 * torch.cos(t * 3.14).unsqueeze(0).expand(batch_size, -1)

    with torch.no_grad():
        out = model(images, landmarks)

    assert out.logits.shape == (batch_size, num_classes)
    assert out.probs.shape == (batch_size, num_classes)
    assert out.score_hat.shape == (batch_size,)
    assert out.f_img.shape == (batch_size, hidden_dim)
    assert out.f_geo.shape == (batch_size, hidden_dim)
    assert out.v_geo.shape == (batch_size, GEO_FEATURE_DIM)
    assert torch.allclose(out.probs.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-4)

    print("DualBranchFBPModel forward test passed.")
    print(f"  device: {device}")
    print(f"  logits:   {tuple(out.logits.shape)}")
    print(f"  probs:    {tuple(out.probs.shape)}")
    print(f"  score_hat: {tuple(out.score_hat.shape)}")
    print(f"  f_img:    {tuple(out.f_img.shape)}")
    print(f"  f_geo:    {tuple(out.f_geo.shape)}")
    print(f"  v_geo:    {tuple(out.v_geo.shape)}")
    print(f"  sample score_hat: {out.score_hat[0].item():.4f}")


if __name__ == "__main__":
    run_dual_branch_forward_test()
