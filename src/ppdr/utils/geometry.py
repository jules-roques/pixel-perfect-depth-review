"""
Geometry utilities for depth map processing.

- Euclidean-to-planar depth conversion (Hypersim convention)
- Depth unprojection to 3D point clouds (NumPy and GPU variants)
"""

import cv2
import numpy as np
import torch

from ppdr.utils.types import Intrinsics


def edge_mask(
    bgr: np.ndarray,
    canny_low: int,
    canny_high: int,
    dilation_px: int,
) -> np.ndarray:
    """
    Compute a dilated edge mask from an BGR image using Canny.

    Returns:
        (H, W) boolean mask, True at edge pixels.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    edges = cv2.Canny(gray, canny_low, canny_high)
    kernel = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return dilated > 0


def euclidean_to_planar(
    d: np.ndarray,
    intrinsics: Intrinsics,
) -> np.ndarray:
    """
    Convert Hypersim *Euclidean range* depth to *planar* (z-buffer) depth.

    Hypersim stores depth as the Euclidean distance from the camera centre to
    the scene point.  This converts it to the standard pinhole-camera planar
    depth Z using:

        Z = d / sqrt( ((u - cx)/fx)^2 + ((v - cy)/fy)^2 + 1 )

    Args:
        d:  (H, W) Euclidean range map.
        intrinsics: Camera intrinsics.

    Returns:
        (H, W) planar depth Z.
    """
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    term = np.sqrt(
        ((u - intrinsics.cx) / intrinsics.fx) ** 2
        + ((v - intrinsics.cy) / intrinsics.fy) ** 2
        + 1.0
    )
    Z = d / term
    return Z


def unproject_depth(
    Z: np.ndarray,
    intrinsics: Intrinsics,
    valid_mask: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Unproject masked depth pixels to 3D on GPU.

    Only the pixels where ``valid_mask`` is True are unprojected,
    avoiding large intermediate tensors for the full (H, W) grid.

    Returns:
        (N, 3) float32 tensor on ``device``.
    """
    vs, us = np.where(valid_mask)
    zs = Z[vs, us]

    u_t = torch.tensor(us, dtype=torch.float32, device=device)
    v_t = torch.tensor(vs, dtype=torch.float32, device=device)
    z_t = torch.tensor(zs, dtype=torch.float32, device=device)

    x_t = (u_t - intrinsics.cx) * z_t / intrinsics.fx
    y_t = (v_t - intrinsics.cy) * z_t / intrinsics.fy

    return torch.stack([x_t, y_t, z_t], dim=-1)
