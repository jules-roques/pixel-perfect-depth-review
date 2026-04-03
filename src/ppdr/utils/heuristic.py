import kornia as K
import torch


def clean_flying_pixels(
    depth: torch.Tensor,  # (B, H, W)
    rgb: torch.Tensor,  # (B, 3, H, W) in [0, 1]
    depth_grad_thresh: float = 0.05,  # min normalised depth gradient to flag
    rgb_edge_thresh: float = 0.05,  # Canny low threshold for RGB edges
    dilate_kernel: int = 3,  # dilation kernel size (px)
) -> torch.Tensor:  # (B, H, W) cleaned depth, 0 = invalid
    B, H, W = depth.shape
    d = depth.unsqueeze(1).float()  # (B, 1, H, W)

    # ── Normalise depth per image ─────────────────────────────────────────
    d_min = d.flatten(2).min(-1).values.view(B, 1, 1, 1)
    d_max = d.flatten(2).max(-1).values.view(B, 1, 1, 1)
    d_n = (d - d_min) / (d_max - d_min + 1e-8)

    # ── Soft depth gradient (Kornia Sobel) ────────────────────────────────
    depth_grad = K.filters.sobel(d_n)
    depth_grad_n = depth_grad / (
        depth_grad.flatten(2).max(-1).values.view(B, 1, 1, 1) + 1e-8
    )

    # ── Binary RGB edge map (Kornia Canny) ────────────────────────────────
    luma = K.color.rgb_to_grayscale(rgb)  # (B, 1, H, W)
    _, rgb_edges = K.filters.canny(
        luma,
        low_threshold=rgb_edge_thresh,
        high_threshold=rgb_edge_thresh * 2,
        kernel_size=(5, 5),
        sigma=(1.0, 1.0),
    )  # rgb_edges: (B, 1, H, W) binary

    # ── Flying pixel mask ─────────────────────────────────────────────────
    # Strong depth gradient with no corresponding RGB edge
    flying = (depth_grad_n > depth_grad_thresh) & (rgb_edges == 0)

    # ── Dilate to cover full halo around edges ────────────────────────────
    kernel = torch.ones(dilate_kernel, dilate_kernel, device=depth.device)
    flying = K.morphology.dilation(flying.float(), kernel) > 0  # (B, 1, H, W)

    # ── Zero out flying pixels ────────────────────────────────────────────
    return depth * ~flying.squeeze(1)  # (B, H, W)
