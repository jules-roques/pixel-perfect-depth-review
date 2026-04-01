"""
Geometry utilities for depth map processing.

- Euclidean-to-planar depth conversion (Hypersim convention)
- Depth unprojection to 3D point clouds (NumPy and GPU variants)
"""

import kornia as K
import torch


def edge_mask(
    rgb: torch.Tensor,
    canny_low: float = 0.2,
    canny_high: float = 0.8,
    dilation_px: int = 4,
) -> torch.Tensor:
    """
    Compute a dilated edge mask from an RGB image using Canny.

    Returns:
        (H, W) boolean mask, True at edge pixels.
    """
    gray = K.color.rgb_to_grayscale(rgb)
    _, edges = K.filters.canny(gray, canny_low, canny_high)
    kernel_size = dilation_px * 2 + 1
    kernel = torch.ones((kernel_size, kernel_size), device=rgb.device)
    dilated = K.morphology.dilation(edges, kernel)
    return dilated > 0


def distances_from_camera_to_depth(
    distances_from_camera: torch.Tensor, m_cam_from_uv: torch.Tensor
) -> torch.Tensor:
    h, w = distances_from_camera.shape
    ndc_grid = create_ndc_grid(h, w, device=distances_from_camera.device).to(
        distances_from_camera.dtype
    )
    rays = compute_ray_directions(ndc_grid, m_cam_from_uv)

    ray_lengths = torch.linalg.norm(rays, dim=-1)

    return distances_from_camera / ray_lengths


def depth_to_point_cloud(
    depth: torch.Tensor, m_cam_from_uv: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    h, w = depth.shape
    ndc_grid = create_ndc_grid(h, w, device=depth.device).to(depth.dtype)
    rays = compute_ray_directions(ndc_grid, m_cam_from_uv)

    point_cloud_grid = rays * depth.unsqueeze(-1)
    valid_points = point_cloud_grid[valid_mask]

    return valid_points


def create_ndc_grid(
    height: int, width: int, device: torch.device | str | None = None
) -> torch.Tensor:
    u = torch.arange(width, device=device, dtype=torch.float32)
    v = torch.arange(height, device=device, dtype=torch.float32)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    u_ndc = (2.0 * uu) / width - 1.0
    v_ndc = (2.0 * vv) / height - 1.0

    return torch.stack([u_ndc, v_ndc, torch.ones_like(u_ndc)], dim=-1)


def compute_ray_directions(
    ndc_grid: torch.Tensor, m_cam_from_uv: torch.Tensor
) -> torch.Tensor:
    return ndc_grid @ m_cam_from_uv.T
