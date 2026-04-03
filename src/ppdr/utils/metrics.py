"""
Edge-Aware Chamfer Distance metric and Canny-based edge masking.
"""

import numpy as np
import torch
from scipy.spatial import KDTree

from ppdr.utils.geometry import depth_to_point_cloud, edge_mask

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------
CANNY_LOW = 0.2
CANNY_HIGH = 0.8
DILATION_PIXELS = 5
MAX_POINTS = 10_000


def edge_aware_chamfer(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    rgb: torch.Tensor,
    m_cam_from_uv: torch.Tensor,
    valid_mask: torch.Tensor,
    canny_low: float = CANNY_LOW,
    canny_high: float = CANNY_HIGH,
    dilation_px: int = DILATION_PIXELS,
) -> list[float]:
    """
    Compute the Edge-Aware Chamfer Distance (GPU-accelerated).

    1. Find edges in ``rgb`` via Canny, dilate.
    2. Combine edge mask with ``valid_mask`` and depth > 0 for both pred/GT.
    3. Unproject masked pixels to 3D on ``device``.
    4. Subsample to ``MAX_POINTS`` for tractable GPU distance computation.
    5. Return bidirectional Chamfer Distance.

    Args:
        pred_depth: (B, H, W) predicted *planar* depth.
        gt_depth:   (B, H, W) ground-truth *planar* depth.
        rgb:        (B, H, W, 3) uint8 image (RGB).
        m_cam_from_uv: Camera intrinsics matrix mapped to NDC.
        valid_mask: (B, H, W) boolean mask indicating valid GT pixels.
        canny_low, canny_high: Canny edge thresholds.
        dilation_px: Dilation radius for the edge mask.

    Returns:
        Chamfer Distance (scalar float).
    """

    assert pred_depth.ndim == 3  # shape (B, H, W)
    assert gt_depth.ndim == 3  # shape (B, H, W)
    assert rgb.ndim == 4 and rgb.shape[1] == 3  # shape (B, 3, H, W)
    assert valid_mask.ndim == 3  # shape (B, H, W)
    assert m_cam_from_uv.ndim == 3  # shape (B, 3, 3)

    edges = edge_mask(rgb, canny_low, canny_high, dilation_px)
    mask = valid_mask & edges

    b = pred_depth.shape[0]
    chamfer_distances = []
    for i in range(b):  # vectorizing masked point clouds is not ideal
        pred_pts = depth_to_point_cloud(pred_depth[i], m_cam_from_uv[i], mask[i])
        gt_pts = depth_to_point_cloud(gt_depth[i], m_cam_from_uv[i], mask[i])
        chamfer_distances.append(_chamfer_distance(pred_pts, gt_pts))

    return chamfer_distances


# KDTrees on GPU are not easy to use, so we do it on CPU
@torch.no_grad()
def _chamfer_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Fast CPU KD-Tree implementation of Chamfer Distance.
    """
    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()

    tree_A = KDTree(A_np)
    tree_B = KDTree(B_np)

    dists_b2a, _ = tree_A.query(B_np, k=1, workers=-1)
    dists_a2b, _ = tree_B.query(A_np, k=1, workers=-1)

    return float(np.mean(dists_a2b) + np.mean(dists_b2a))
