"""
Cross-Gradient Flying Pixel Correction Filter.

Flying pixels occur at depth discontinuities where the depth map has a high
spatial gradient NOT supported by a high RGB gradient (i.e. no semantic edge).
"""

import kornia as K
import torch

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
DEPTH_GRADIENT_THRESHOLD = 0.05  # Fraction of depth range
RGB_GRADIENT_THRESHOLD = 30.0  # Absolute gradient magnitude (0-255 scale)


def clean_flying_pixels(
    depth: torch.Tensor,
    rgb: torch.Tensor,
    depth_threshold: float = DEPTH_GRADIENT_THRESHOLD,
    rgb_threshold: float = RGB_GRADIENT_THRESHOLD,
) -> torch.Tensor:
    """
    Remove flying pixels from a depth map using a cross-gradient filter.

    Args:
        depth: (H, W) depth map (float32).
        rgb: (3, H, W) RGB image in [0, 1].
        depth_threshold: Fraction of the depth range used as gradient
            threshold.
        rgb_threshold: Absolute gradient threshold for the RGB image.

    Returns:
        Cleaned depth map (same shape). Inconsistent pixels are set to 0.
    """
    cleaned = depth.clone()

    depth_input = depth.unsqueeze(0).unsqueeze(0).float()
    rgb_input = rgb.unsqueeze(0)

    # ---- Depth gradient magnitude (Sobel) ----------------------------------
    depth_grad_mag = K.filters.sobel(depth_input).squeeze()

    # ---- RGB gradient magnitude (Sobel on grayscale) -----------------------
    gray = K.color.rgb_to_grayscale(rgb_input)
    # Scale to 0-255 to match the original threshold
    if rgb.is_floating_point() and rgb.max() <= 1.001:
        gray = gray * 255.0

    rgb_grad_mag = K.filters.sobel(gray).squeeze()

    # ---- Identify inconsistent pixels --------------------------------------
    valid = depth > 0
    if valid.any():
        depth_range = depth[valid].max() - depth[valid].min()
    else:
        depth_range = torch.tensor(1.0, device=depth.device)

    abs_depth_thresh = depth_threshold * depth_range

    inconsistent = (depth_grad_mag > abs_depth_thresh) & (rgb_grad_mag < rgb_threshold)
    cleaned[inconsistent] = 0.0

    return cleaned
