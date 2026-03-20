"""
Edge-Aware Chamfer Distance metric and Canny-based edge masking.
"""

import numpy as np
import cv2
from scipy.spatial import KDTree

from utils.geometry import unproject_depth

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------
CANNY_LOW = 50
CANNY_HIGH = 150
DILATION_PIXELS = 5


def _edge_mask(rgb: np.ndarray, canny_low: int = CANNY_LOW,
               canny_high: int = CANNY_HIGH,
               dilation_px: int = DILATION_PIXELS) -> np.ndarray:
    """
    Compute a dilated edge mask from an RGB image using Canny.

    Returns:
        (H, W) boolean mask, True at edge pixels.
    """
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb
    edges = cv2.Canny(gray, canny_low, canny_high)
    kernel = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return dilated > 0


def _chamfer_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Bidirectional Chamfer Distance between point sets A and B.

    CD = mean(d(a, B) for a in A) + mean(d(b, A) for b in B)
    """
    tree_B = KDTree(B)
    tree_A = KDTree(A)
    dists_A2B, _ = tree_B.query(A)
    dists_B2A, _ = tree_A.query(B)
    return float(np.mean(dists_A2B) + np.mean(dists_B2A))


def edge_aware_chamfer(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    valid_mask: np.ndarray,
    canny_low: int = CANNY_LOW,
    canny_high: int = CANNY_HIGH,
    dilation_px: int = DILATION_PIXELS,
) -> float:
    """
    Compute the Edge-Aware Chamfer Distance.

    1. Find edges in ``rgb`` via Canny, dilate.
    2. Combine edge mask with ``valid_mask`` and depth > 0 for both pred/GT.
    3. Unproject masked pixels to 3D.
    4. Return bidirectional Chamfer Distance.

    Args:
        pred_depth: (H, W) predicted *planar* depth.
        gt_depth:   (H, W) ground-truth *planar* depth.
        rgb:        (H, W, 3) uint8 image (BGR or RGB).
        intrinsics: (3, 3) camera intrinsic matrix.
        valid_mask: (H, W) boolean mask indicating valid GT pixels.
        canny_low, canny_high: Canny edge thresholds.
        dilation_px: Dilation radius for the edge mask.

    Returns:
        Chamfer Distance (scalar float).
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    edges = _edge_mask(rgb, canny_low, canny_high, dilation_px)

    # Combined mask: valid GT, edge pixels, both depths > 0
    mask = valid_mask & edges & (pred_depth > 0) & (gt_depth > 0)

    if mask.sum() < 10:
        return float("nan")

    pred_pts = unproject_depth(pred_depth, fx, fy, cx, cy, valid_mask=mask)
    gt_pts = unproject_depth(gt_depth, fx, fy, cx, cy, valid_mask=mask)

    if len(pred_pts) < 2 or len(gt_pts) < 2:
        return float("nan")

    return _chamfer_distance(pred_pts, gt_pts)
