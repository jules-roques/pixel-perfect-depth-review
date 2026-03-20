"""
Wrapper for Pixel-Perfect Depth (PPD) model.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# Ensure the PPD submodule is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "pixel-perfect-depth"))

from ppd.models.ppd import PixelPerfectDepth


from typing import Optional


class PPDModel:
    """
    Thin wrapper around ``PixelPerfectDepth`` with a unified
    ``predict(image_bgr) -> (depth, resized_image)`` API.
    """

    def __init__(
        self,
        semantics_pth: str = "checkpoints/depth_anything_v2_vitl.pth",
        model_pth: str = "checkpoints/ppd.pth",
        sampling_steps: int = 4,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = PixelPerfectDepth(
            semantics_model="DA2",
            semantics_pth=semantics_pth,
            sampling_steps=sampling_steps,
        )
        self.model.load_state_dict(
            torch.load(model_pth, map_location="cpu"), strict=False
        )
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray):
        """
        Run PPD inference.

        Args:
            image_bgr: (H, W, 3) BGR uint8 image.

        Returns:
            depth: (rH, rW) relative depth map (float32, numpy).
            resized_image: (rH, rW, 3) BGR uint8 — the resized input image
                that was actually fed to the network.
        """
        depth_tensor, resized_image = self.model.infer_image(image_bgr)
        depth = depth_tensor.squeeze().cpu().numpy()
        return depth, resized_image
