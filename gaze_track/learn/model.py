"""Compact gaze network: timm backbone → [gaze pitch, gaze yaw, head pitch, head yaw] (ETH convention,
radians, in the normalized camera)."""

from __future__ import annotations

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GazeNet(nn.Module):
    def __init__(self, backbone: str = "mobileone_s0", pretrained: bool = True):
        super().__init__()
        import timm

        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        # timm's num_features is not always the pooled output size (e.g. MobileNetV4 adds a
        # 1280-d head after a 960-d stage), so measure it.
        with torch.no_grad():
            dim = self.backbone.eval()(torch.zeros(1, 3, 112, 112)).shape[-1]
        self.backbone.train()
        self.head = nn.Linear(dim, 4)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x_uint8_bgr: torch.Tensor) -> torch.Tensor:
        """x: (N, H, W, 3) uint8 BGR (as stored) → (N, 4)."""
        x = x_uint8_bgr.flip(-1).permute(0, 3, 1, 2).float().div(255.0)
        return self.forward_float((x - self.mean) / self.std)

    def forward_float(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) normalized RGB float."""
        return self.head(self.backbone(x))


def reparameterize(model: GazeNet) -> GazeNet:
    """Fold MobileOne/RepViT/FastViT train-time branches into single convs (faster inference).
    Verified against the original outputs; falls back to the unfused model if anything changes."""
    import copy

    try:
        from timm.utils.model import reparameterize_model

        model = model.eval()
        fused = reparameterize_model(copy.deepcopy(model))
        x = torch.randint(0, 255, (2, 112, 112, 3), dtype=torch.uint8, device=next(model.parameters()).device)
        with torch.no_grad():
            ref, out = model(x), fused(x)
        # Relative to the output scale: an untrained MobileOne outputs ~1e15, where fusion's float
        # rounding (~1e-7 relative) exceeds any fixed absolute tolerance.
        if float((ref - out).abs().max()) <= 1e-4 * max(1.0, float(ref.abs().max())):
            return fused
    except Exception:
        pass
    return model
