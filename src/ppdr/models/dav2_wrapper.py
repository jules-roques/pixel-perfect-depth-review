"""
Wrappers for Depth Anything v2 (standalone) and DAv2 + Cleaned variant.

Because the repo's ``DepthAnythingV2`` has its DPTHead and ``forward()``
commented out, this module reconstructs the full architecture in a new class
``_DepthAnythingV2Full`` so the original ``dpt.py`` is never modified.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from ppdr.vendor.ppd.models.depth_anything_v2.dpt import DepthAnythingV2, DPTHead
from ppdr.vendor.ppd.models.depth_anything_v2.util.transform import (
    NormalizeImage,
    PrepareForNet,
    Resize,
)

from .heuristic import clean_flying_pixels
from .interface import DepthModel


class DAv2Cleaned(DepthModel):
    """
    DAv2 + cross-gradient flying-pixel cleaning filter.
    """

    def __init__(self, dav2: "DAv2"):
        self.dav2 = dav2

    @torch.no_grad()
    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """
        Predict depth and clean flying pixels.

        Returns:
            cleaned_depth: (H, W) depth with flying pixels zeroed.
        """
        depth, _ = self.dav2.predict(bgr)
        cleaned = clean_flying_pixels(depth, bgr)
        return cleaned


class DAv2(DepthModel):
    """
    Standalone Depth Anything V2 (ViT-L) wrapper.
    """

    def __init__(
        self,
        model_path: str = "checkpoints/depth_anything_v2_vitl.pth",
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = _DepthAnythingV2Full(
            encoder="vitl",
            features=256,
            out_channels=[256, 512, 1024, 1024],
        )
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            bgr: (H, W, 3) BGR uint8.

        Returns:
            depth: (H, W) relative depth (float32 numpy).
        """
        depth, _ = self.model.infer_image(bgr, input_size=518)
        return depth


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
    def infer_image(self, raw_image: np.ndarray, input_size: int = 518):
        image, (h, w) = self._image2tensor(raw_image, input_size)
        depth = self.forward(image)
        depth = F.interpolate(
            depth[:, None], (h, w), mode="bilinear", align_corners=True
        )[0, 0]
        return depth.cpu().numpy()

    @staticmethod
    def _image2tensor(raw_image: np.ndarray, input_size: int = 518):
        transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )
        h, w = raw_image.shape[:2]
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return image.to(device), (h, w)
