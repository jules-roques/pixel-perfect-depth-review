"""
Wrapper for Pixel-Perfect Depth (PPD) model.
"""

import torch
import torch.nn.functional as F

from ppdr.models.interface import DepthModel
from ppdr.utils.geometry import create_valid_depth_mask, recover_metric_depth_from_log
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
        device: str | torch.device | None = None,
    ):
        super().__init__(device)

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
    def _predict(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        ori_h, ori_w = rgb.shape[-2:]

        # Calculate target resize dimensions
        tar_area = 1024 * 768
        ori_area = ori_h * ori_w
        scale = (tar_area / ori_area) ** 0.5

        # Cascade DiT requires dimensions divisible by patch size (usually 16 or 8)
        resize_h = max(16, int(round(ori_h * scale / 16)) * 16)
        resize_w = max(16, int(round(ori_w * scale / 16)) * 16)

        # Interpolate
        mode = "area" if scale < 1 else "bicubic"
        image = F.interpolate(
            rgb,
            size=(resize_h, resize_w),
            mode=mode,
            align_corners=(False if mode == "bicubic" else None),
        )

        # 6. Inference with Mixed Precision
        autocast_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            depth_tensor = self.model.forward_test(image)

        # 7. Resize back to original resolution
        pred_depth = F.interpolate(
            depth_tensor,
            size=(ori_h, ori_w),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)

        pred_mask = create_valid_depth_mask(pred_depth)

        return pred_depth, pred_mask

    def align_pred_on_metric_depth(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return recover_metric_depth_from_log(pred, gt, mask)
