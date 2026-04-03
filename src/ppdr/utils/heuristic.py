import kornia as K

# def clean_flying_pixels(
#     depth: torch.Tensor,  # (B, H, W)
#     rgb: torch.Tensor,  # (B, 3, H, W) in [0, 1]
#     disp_grad_thresh: float = 0.05,
#     rgb_grad_thresh: float = 0.02,
#     ratio_thresh: float = 5.0,
#     dilate_iters: int = 1,
# ) -> torch.Tensor:
#     """
#     Remove flying pixels from a depth map using joint bilateral filtering and cross-gradient.

#     Args:
#         depth: (B, H, W) depth map (float32).
#         rgb: (B, 3, H, W) RGB image in [0, 1].

#     Returns:
#         Cleaned depth map (same shape). Inconsistent pixels are set to 0.
#     """

#     B, H, W = depth.shape
#     d = depth.unsqueeze(1).float()  # (B, 1, H, W)

#     # ── 1. Kornia joint bilateral as replacement source ───────────────────
#     d_filtered = K.filters.joint_bilateral_blur(
#         input=d,
#         guidance=rgb,
#         kernel_size=(9, 9),
#         sigma_color=0.01,
#         sigma_space=(1.0, 1.0),
#     )  # (B, 1, H, W)

#     # ── 2. Cross-gradient flying-pixel detection ──────────────────────────
#     def sobel_mag(x):
#         # x: (B, 1, H, W)
#         kx = torch.tensor(
#             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device
#         ).view(1, 1, 3, 3)
#         gx = F.conv2d(x, kx, padding=1)
#         gy = F.conv2d(x, kx.transpose(-1, -2), padding=1)
#         return torch.sqrt(gx**2 + gy**2 + 1e-8)

#     # Normalise depth per image before gradient
#     d_min = d.flatten(2).min(-1).values.view(B, 1, 1, 1)
#     d_max = d.flatten(2).max(-1).values.view(B, 1, 1, 1)
#     d_norm = (d - d_min) / (d_max - d_min + 1e-8)

#     g_d = sobel_mag(d_norm)
#     g_rgb = sobel_mag(rgb.mean(dim=1, keepdim=True))

#     # Normalise per image for threshold stability
#     g_d_n = g_d / (g_d.flatten(2).max(-1).values.view(B, 1, 1, 1) + 1e-8)
#     g_rgb_n = g_rgb / (g_rgb.flatten(2).max(-1).values.view(B, 1, 1, 1) + 1e-8)

#     flying = (
#         (g_d_n > disp_grad_thresh)
#         & (g_rgb_n < rgb_grad_thresh)
#         & (g_d / (g_rgb + 1e-6) > ratio_thresh)
#     )  # (B, 1, H, W) bool

#     if dilate_iters > 0:
#         k = torch.ones(1, 1, 3, 3, device=d.device)
#         for _ in range(dilate_iters):
#             flying = F.conv2d(flying.float(), k, padding=1) > 0

#     # ── 3. Replace only flagged pixels ────────────────────────────────────
#     cleaned = torch.where(flying, d_filtered, d)

#     return cleaned.squeeze(1)  # (B, H, W)


def clean_flying_pixels(depth, rgb):
    # No detection — just a tight edge-preserving smooth everywhere.
    # sigma_color=0.01 means the filter is essentially a no-op at any
    # pixel where the RGB neighbourhood has colour variation > 0.01,
    # so it only smooths in genuinely flat-colour regions.
    return K.filters.joint_bilateral_blur(
        input=depth.unsqueeze(1),
        guidance=rgb,
        kernel_size=(9, 9),
        sigma_color=0.005,
        sigma_space=(2.0, 2.0),
    ).squeeze(1)
