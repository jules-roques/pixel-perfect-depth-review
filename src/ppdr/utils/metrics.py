"""
Edge-Aware Chamfer Distance metric and Canny-based edge masking.
"""

import numpy as np
import torch

from ppdr.utils.geometry import edge_mask, unproject_depth
from ppdr.utils.types import Intrinsics

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------
CANNY_LOW = 50
CANNY_HIGH = 150
DILATION_PIXELS = 5
MAX_POINTS = 10_000


def edge_aware_chamfer(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    bgr: np.ndarray,
    intrinsics: Intrinsics,
    valid_mask: np.ndarray,
    device: torch.device | None = None,
    canny_low: int = CANNY_LOW,
    canny_high: int = CANNY_HIGH,
    dilation_px: int = DILATION_PIXELS,
) -> float:
    """
    Compute the Edge-Aware Chamfer Distance (GPU-accelerated).

    1. Find edges in ``bgr`` via Canny, dilate.
    2. Combine edge mask with ``valid_mask`` and depth > 0 for both pred/GT.
    3. Unproject masked pixels to 3D on ``device``.
    4. Subsample to ``MAX_POINTS`` for tractable GPU distance computation.
    5. Return bidirectional Chamfer Distance.

    Args:
        pred_depth: (H, W) predicted *planar* depth.
        gt_depth:   (H, W) ground-truth *planar* depth.
        bgr:        (H, W, 3) uint8 image (BGR or RGB).
        intrinsics: Camera intrinsics.
        valid_mask: (H, W) boolean mask indicating valid GT pixels.
        device:     Torch device for the distance computation (default: CPU).
        canny_low, canny_high: Canny edge thresholds.
        dilation_px: Dilation radius for the edge mask.

    Returns:
        Chamfer Distance (scalar float).
    """
    if device is None:
        device = torch.device("cpu")

    edges = edge_mask(bgr, canny_low, canny_high, dilation_px)
    mask = valid_mask & edges & (pred_depth > 0) & (gt_depth > 0)

    if mask.sum() < 10:
        return float("nan")

    pred_pts = unproject_depth(pred_depth, intrinsics, mask, device)
    gt_pts = unproject_depth(gt_depth, intrinsics, mask, device)

    if len(pred_pts) < 2 or len(gt_pts) < 2:
        return float("nan")

    pred_pts = _subsample(pred_pts, MAX_POINTS)
    gt_pts = _subsample(gt_pts, MAX_POINTS)

    return _chamfer_distance_gpu(pred_pts, gt_pts)


def _subsample(points: torch.Tensor, max_points: int) -> torch.Tensor:
    """Uniformly subsample a point cloud if it exceeds ``max_points``."""
    if len(points) <= max_points:
        return points
    indices = torch.randperm(len(points), device=points.device)[:max_points]
    return points[indices]


@torch.no_grad()
def _chamfer_distance_gpu(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Bidirectional Chamfer Distance on GPU using ``torch.cdist``.

    CD = mean(min_b ||a - b||  for a in A) + mean(min_a ||b - a||  for b in B)
    """
    dists = torch.cdist(A.unsqueeze(0), B.unsqueeze(0)).squeeze(0)
    mean_a2b = dists.min(dim=1).values.mean()
    mean_b2a = dists.min(dim=0).values.mean()
    return float(mean_a2b + mean_b2a)
