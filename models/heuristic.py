"""
Cross-Gradient Flying Pixel Correction Filter.

Flying pixels occur at depth discontinuities where the depth map has a high
spatial gradient NOT supported by a high RGB gradient (i.e. no semantic edge).
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
DEPTH_GRADIENT_THRESHOLD = 0.05   # Fraction of depth range
RGB_GRADIENT_THRESHOLD = 30       # Absolute gradient magnitude (0-255 scale)


def clean_flying_pixels(
    depth: np.ndarray,
    rgb: np.ndarray,
    depth_threshold: float = DEPTH_GRADIENT_THRESHOLD,
    rgb_threshold: float = RGB_GRADIENT_THRESHOLD,
) -> np.ndarray:
    """
    Remove flying pixels from a depth map using a cross-gradient filter.

    Args:
        depth: (H, W) depth map (float32, metric or relative).
        rgb: (H, W, 3) RGB image (uint8 BGR or RGB).
        depth_threshold: Fraction of the depth range used as gradient
            threshold.  Pixels with depth-gradient magnitude above
            ``depth_threshold * (depth_max - depth_min)`` are candidates.
        rgb_threshold: Absolute gradient threshold for the RGB image.
            Candidates are only rejected when the RGB gradient is *below*
            this value (i.e. no supporting edge).

    Returns:
        Cleaned depth map (same shape). Inconsistent pixels are set to 0.
    """
    cleaned = depth.copy()

    # ---- Depth gradient magnitude (Sobel) ----------------------------------
    depth_f32 = depth.astype(np.float32)
    grad_x = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=3)
    depth_grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # ---- RGB gradient magnitude (Sobel on grayscale) -----------------------
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    else:
        gray = rgb
    gray = gray.astype(np.float32)
    rgb_gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    rgb_gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    rgb_grad_mag = np.sqrt(rgb_gx ** 2 + rgb_gy ** 2)

    # ---- Identify inconsistent pixels --------------------------------------
    valid = depth > 0
    depth_range = depth_f32[valid].max() - depth_f32[valid].min() if valid.any() else 1.0
    abs_depth_thresh = depth_threshold * depth_range

    inconsistent = (depth_grad_mag > abs_depth_thresh) & (rgb_grad_mag < rgb_threshold)
    cleaned[inconsistent] = 0.0

    return cleaned
