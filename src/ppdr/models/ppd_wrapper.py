"""
Wrapper for Pixel-Perfect Depth (PPD) model.
"""

import numpy as np
import torch

from ppdr.models.interface import DepthModel
from ppdr.vendor.ppd.models.ppd import PixelPerfectDepth


class PPD(DepthModel):
    """
    Thin wrapper around ``PixelPerfectDepth`` with a unified
    ``predict(image_bgr) -> (depth, resized_image)`` API.
    """

    def __init__(
        self,
        semantics_path: str = "checkpoints/depth_anything_v2_vitl.pth",
        model_path: str = "checkpoints/ppd.pth",
        sampling_steps: int = 4,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = PixelPerfectDepth(
            semantics_model="DA2",
            semantics_pth=semantics_path,
            sampling_steps=sampling_steps,
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu"), strict=False
        )
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """
        Run PPD inference.

        Args:
            bgr: (H, W, 3) BGR uint8 image.

        Returns:
            depth: (rH, rW) relative depth map (float32, numpy).
        """
        depth_tensor, _ = self.model.infer_image(bgr)
        depth = depth_tensor.squeeze().cpu().numpy()
        return depth
