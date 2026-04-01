"""
Wrapper for Pixel-Perfect Depth (PPD) model.
"""

import kornia as K
import torch

from ppdr.models.interface import DepthModel
from ppdr.vendor.ppd.models.ppd import PixelPerfectDepth


class PPD(DepthModel):
    """
    Thin wrapper around ``PixelPerfectDepth`` with a unified
    ``predict(image_rgb) -> (depth, resized_image)`` API.
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
    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Run PPD inference.

        Args:
            rgb: (3, H, W) RGB tensor in [0, 1] or (H, W, 3) uint8 tensor.

        Returns:
            depth: (H, W) relative depth map.
        """
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            rgb = rgb.permute(2, 0, 1)
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0

        ori_h, ori_w = rgb.shape[-2:]

        # Resize to PPD expected area
        tar_area = 1024 * 768
        ori_area = ori_h * ori_w
        scale = (tar_area / ori_area) ** 0.5
        resize_h = max(16, int(round(ori_h * scale / 16)) * 16)
        resize_w = max(16, int(round(ori_w * scale / 16)) * 16)

        mode = "area" if scale < 1 else "bicubic"
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)

        import torch.nn.functional as F

        image = F.interpolate(
            rgb,
            size=(resize_h, resize_w),
            mode=mode,
            align_corners=False if mode == "bicubic" else None,
        )

        autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            depth_tensor = self.model.forward_test(image)

        depth = depth_tensor.squeeze()

        return K.geometry.resize(
            depth.unsqueeze(0).unsqueeze(0),
            (ori_h, ori_w),
            interpolation="bilinear",
        ).squeeze()
