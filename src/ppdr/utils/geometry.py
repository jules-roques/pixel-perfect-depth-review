"""
Geometry utilities for depth map processing.

- Euclidean-to-planar depth conversion (Hypersim convention)
- Depth unprojection to 3D point clouds (NumPy and GPU variants)
"""

import kornia as K
import torch


def edge_mask(
    rgb: torch.Tensor,
    canny_low: float,
    canny_high: float,
    dilation_px: int,
) -> torch.Tensor:
    """
    Compute a dilated edge mask from an RGB image using Canny.

    Returns:
        (B, H, W) boolean mask, True at edge pixels.
    """
    assert rgb.ndim == 4 and rgb.shape[1] == 3  # shape (B, 3, H, W)
    gray = K.color.rgb_to_grayscale(rgb)
    _, edges = K.filters.canny(gray, canny_low, canny_high)
    kernel_size = dilation_px * 2 + 1
    kernel = torch.ones((kernel_size, kernel_size), device=rgb.device)
    dilated = K.morphology.dilation(edges, kernel).squeeze(1)
    return dilated > 0


def distances_from_camera_to_depth(
    distances_from_camera: torch.Tensor, m_cam_from_uv: torch.Tensor
) -> torch.Tensor:
    assert distances_from_camera.ndim == 2  # shape (H, W)
    assert m_cam_from_uv.shape == (3, 3)

    h, w = distances_from_camera.shape
    ndc_grid = create_ndc_grid(h, w, device=distances_from_camera.device).to(
        distances_from_camera.dtype
    )
    rays = compute_ray_directions(ndc_grid, m_cam_from_uv)

    ray_lengths = torch.linalg.norm(rays, dim=-1)

    return distances_from_camera / ray_lengths


def depth_to_point_cloud(
    depth: torch.Tensor, m_cam_from_uv: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    assert depth.ndim == 2  # shape (H, W)
    assert m_cam_from_uv.shape == (3, 3)
    assert mask.ndim == 2  # shape (H, W)

    h, w = depth.shape
    ndc_grid = create_ndc_grid(h, w, device=depth.device).to(depth.dtype)
    rays = compute_ray_directions(ndc_grid, m_cam_from_uv)

    point_cloud_grid = rays * depth.unsqueeze(-1)
    valid_points = point_cloud_grid[mask]

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


def recover_metric_depth_from_log(
    pred: torch.Tensor,  # PPD output: predicted log-depth
    gt: torch.Tensor,  # GT metric depth (Hypersim)
    mask: torch.Tensor,  # Valid pixel mask
    eps: float = 1.0,
) -> torch.Tensor:
    """
    Affine alignment in log-depth space (for PPD).
    Solves: a * pred + b = log(gt + eps)  via least squares.
    """
    valid_pred = pred[mask].view(-1)
    valid_gt_log = torch.log(gt[mask].view(-1) + eps)

    # Build system: [pred, 1] @ [a, b]^T = gt_log
    A = torch.stack([valid_pred, torch.ones_like(valid_pred)], dim=1)  # (N, 2)
    # Normal equations: (A^T A) [a,b] = A^T gt_log
    ATA = A.T @ A  # (2, 2)
    ATb = A.T @ valid_gt_log  # (2,)
    a, b = torch.linalg.solve(ATA, ATb)

    pred_metric = torch.exp(a * pred + b) - eps
    return torch.clamp(pred_metric, min=1e-3, max=gt[mask].max().item())


def recover_metric_depth_from_disparity(
    disp: torch.Tensor,  # DA2 output: relative disparity
    gt: torch.Tensor,  # GT metric depth (Hypersim)
    mask: torch.Tensor,  # Valid pixel mask
) -> torch.Tensor:
    """
    Affine alignment in disparity space (for DA2).
    Solves: a * disp + b = 1/gt_depth  via least squares.

    Note: gt must have strictly positive values in the valid mask.
    """
    valid_disp = disp[mask].view(-1)
    valid_gt_disp = 1.0 / gt[mask].view(-1)  # depth → disparity

    A = torch.stack([valid_disp, torch.ones_like(valid_disp)], dim=1)  # (N, 2)
    ATA = A.T @ A
    ATb = A.T @ valid_gt_disp
    a, b = torch.linalg.solve(ATA, ATb)

    aligned_disp = a * disp + b
    aligned_disp = torch.clamp(aligned_disp, min=1e-6)  # prevent /0 and neg depth
    pred_metric = 1.0 / aligned_disp
    return torch.clamp(pred_metric, min=1e-3, max=gt[mask].max().item())
