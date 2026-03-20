"""
Geometry utilities for depth map processing.

- Euclidean-to-planar depth conversion (Hypersim convention)
- Depth unprojection to 3D point clouds
"""

import numpy as np
from typing import Optional


def euclidean_to_planar(
    d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Convert Hypersim *Euclidean range* depth to *planar* (z-buffer) depth.

    Hypersim stores depth as the Euclidean distance from the camera centre to
    the scene point.  This converts it to the standard pinhole-camera planar
    depth Z using:

        Z = d / sqrt( ((u - cx)/fx)^2 + ((v - cy)/fy)^2 + 1 )

    Args:
        d:  (H, W) Euclidean range map.
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.

    Returns:
        (H, W) planar depth Z.
    """
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32))

    term = np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)
    Z = d / term
    return Z


def unproject_depth(
    Z: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Unproject a planar depth map to a 3D point cloud.

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    Args:
        Z:  (H, W) planar depth map.
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.
        valid_mask: Optional (H, W) boolean mask.  Only valid pixels are
            returned.

    Returns:
        (N, 3) point cloud as float32.
    """
    H, W = Z.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32))

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    if valid_mask is not None:
        points = points[valid_mask.reshape(-1)]

    return points.astype(np.float32)
