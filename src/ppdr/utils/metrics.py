"""
Edge-Aware Chamfer Distance metric and Canny-based edge masking.
"""

import torch

from ppdr.utils.geometry import depth_to_point_cloud, edge_mask

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------
CANNY_LOW = 50
CANNY_HIGH = 150
DILATION_PIXELS = 5
MAX_POINTS = 10_000


def edge_aware_chamfer(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    rgb: torch.Tensor,
    m_cam_from_uv: torch.Tensor,
    valid_mask: torch.Tensor,
    canny_low: int = CANNY_LOW,
    canny_high: int = CANNY_HIGH,
    dilation_px: int = DILATION_PIXELS,
) -> float:
    """
    Compute the Edge-Aware Chamfer Distance (GPU-accelerated).

    1. Find edges in ``rgb`` via Canny, dilate.
    2. Combine edge mask with ``valid_mask`` and depth > 0 for both pred/GT.
    3. Unproject masked pixels to 3D on ``device``.
    4. Subsample to ``MAX_POINTS`` for tractable GPU distance computation.
    5. Return bidirectional Chamfer Distance.

    Args:
        pred_depth: (H, W) predicted *planar* depth.
        gt_depth:   (H, W) ground-truth *planar* depth.
        rgb:        (H, W, 3) uint8 image (RGB).
        m_cam_from_uv: Camera intrinsics matrix mapped to NDC.
        valid_mask: (H, W) boolean mask indicating valid GT pixels.
        canny_low, canny_high: Canny edge thresholds.
        dilation_px: Dilation radius for the edge mask.

    Returns:
        Chamfer Distance (scalar float).
    """

    edges = edge_mask(rgb, canny_low, canny_high, dilation_px)
    mask = valid_mask & edges & (pred_depth > 0) & (gt_depth > 0)

    if mask.sum() < 10:
        return float("nan")

    pred_pts = depth_to_point_cloud(pred_depth, m_cam_from_uv, mask)
    gt_pts = depth_to_point_cloud(gt_depth, m_cam_from_uv, mask)

    return _chamfer_distance(pred_pts, gt_pts)


def _subsample(points: torch.Tensor, max_points: int) -> torch.Tensor:
    """Uniformly subsample a point cloud if it exceeds ``max_points``."""
    if len(points) <= max_points:
        return points
    indices = torch.randperm(len(points), device=points.device)[:max_points]
    return points[indices]


@torch.no_grad()
def _chamfer_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Bidirectional Chamfer Distance on GPU using ``torch.cdist``.

    CD = mean(min_b ||a - b||  for a in A) + mean(min_a ||b - a||  for b in B)
    """
    dists = torch.cdist(A.unsqueeze(0), B.unsqueeze(0)).squeeze(0)
    mean_a2b = dists.min(dim=1).values.mean()
    mean_b2a = dists.min(dim=0).values.mean()
    return float(mean_a2b + mean_b2a)
