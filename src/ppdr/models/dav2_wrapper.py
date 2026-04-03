"""
Wrappers for Depth Anything v2 (standalone) and DAv2 + Cleaned variant.

Because the repo's ``DepthAnythingV2`` has its DPTHead and ``forward()``
commented out, this module reconstructs the full architecture in a new class
``_DepthAnythingV2Full`` so the original ``dpt.py`` is never modified.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppdr.utils.geometry import recover_metric_depth_from_disparity
from ppdr.utils.heuristic import clean_flying_pixels
from ppdr.vendor.ppd.models.depth_anything_v2.dpt import DepthAnythingV2, DPTHead

from .interface import DepthModel


class DAv2(DepthModel):
    """
    Standalone Depth Anything V2 (ViT-L) wrapper.
    """

    def __init__(
        self,
        model_path: str = "checkpoints/depth_anything_v2_vitl.pth",
        device: str | torch.device | None = None,
    ):
        super().__init__(device)

        self.model = _DepthAnythingV2Full(
            encoder="vitl",
            features=256,
            out_channels=[256, 512, 1024, 1024],
        )
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def _predict(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) RGB float tensor in [0, 1].

        Returns:
            disparity: (B, H, W) relative disparity.
        """
        return self.model.infer_tensor(rgb, input_size=518)

    def align_pred_on_metric_depth(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return recover_metric_depth_from_disparity(pred, gt, mask)


class DAv2Cleaned(DAv2):
    """
    DAv2 + cross-gradient flying-pixel cleaning filter.
    """

    @torch.no_grad()
    def _predict(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Predict depth and clean flying pixels.

        Args:
            rgb: (B, 3, H, W) RGB float tensor in [0, 1].

        Returns:
            disparity: (B, H, W) relative disparity.
        """
        disparity = super()._predict(rgb)
        return clean_flying_pixels(disparity, rgb)


class _DepthAnythingV2Full(nn.Module):
    """
    Full Depth Anything V2 with DPTHead + forward(), built from the existing
    ``DepthAnythingV2`` (which only has the DINOv2 backbone loaded).
    """

    def __init__(
        self,
        encoder: str = "vitl",
        features: int = 256,
        out_channels: list | None = None,
    ):
        if out_channels is None:
            out_channels = [256, 512, 1024, 1024]

        super().__init__()
        self.intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
            "vitg": [9, 19, 29, 39],
        }
        self.encoder = encoder
        # Re-use the DINOv2 backbone from the original class
        base = DepthAnythingV2(
            encoder=encoder,
            features=features,
            out_channels=out_channels,
        )
        self.pretrained = base.pretrained
        self.depth_head = DPTHead(
            self.pretrained.embed_dim,
            features,
            use_bn=False,
            out_channels=out_channels,
            use_clstoken=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder], return_class_token=True
        )
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)
        return depth.squeeze(1)

    @torch.no_grad()
    def infer_tensor(self, rgb: torch.Tensor, input_size: int = 518) -> torch.Tensor:
        """
        Infers depth from a RGB tensor, matching the DAv2 preprocessing.
        """
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            rgb = rgb.permute(2, 0, 1)
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0

        h, w = rgb.shape[-2:]
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)

        scale = input_size / min(h, w)
        new_h = int(round(h * scale / 14)) * 14
        if new_h < input_size:
            new_h = int(math.ceil(h * scale / 14)) * 14

        new_w = int(round(w * scale / 14)) * 14
        if new_w < input_size:
            new_w = int(math.ceil(w * scale / 14)) * 14

        image = F.interpolate(
            rgb, size=(new_h, new_w), mode="bicubic", align_corners=False
        )
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        image = (image - mean) / std

        depth = self.forward(image)
        depth = F.interpolate(
            depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True
        ).squeeze(1)

        return depth
