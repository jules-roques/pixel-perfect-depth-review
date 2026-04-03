"""
Edge-Aware Chamfer Distance metric and Canny-based edge masking.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial import KDTree

from ppdr.utils.geometry import depth_to_point_cloud, edge_mask


@dataclass
class Metrics:
    """Holds per-image metric values for one batch."""

    chamfer_distances: list[float] = field(default_factory=list)
    inference_times: list[float] = field(default_factory=list)
    precisions: list[float] = field(default_factory=list)
    recalls: list[float] = field(default_factory=list)
    fscores: list[float] = field(default_factory=list)

    def extend(self, other: "Metrics") -> None:
        self.chamfer_distances.extend(other.chamfer_distances)
        self.inference_times.extend(other.inference_times)
        self.precisions.extend(other.precisions)
        self.recalls.extend(other.recalls)
        self.fscores.extend(other.fscores)

    def mean(self) -> dict[str, float]:
        return {
            "mean_chamfer_distance": float(np.mean(self.chamfer_distances)),
            "mean_inference_time": float(np.mean(self.inference_times)),
            "mean_precision": float(np.mean(self.precisions)),
            "mean_recall": float(np.mean(self.recalls)),
            "mean_fscore": float(np.mean(self.fscores)),
        }

    def print_summary(self) -> None:
        for k, v in self.mean().items():
            print(f"  {k}: {v:.4f}")


def depth_fscore(
    pred: torch.Tensor,  # (B, H, W), 0 = masked
    gt: torch.Tensor,  # (B, H, W)
    valid: torch.Tensor,  # (B, H, W) GT valid mask
    delta: float = 1.25,
) -> dict[str, list[float]]:
    """
    Computes F-score per image in the batch.
    Returns lists of length B — one scalar per image.
    """
    B = pred.shape[0]
    precisions, recalls, fscores = [], [], []

    for b in range(B):
        p = pred[b]  # (H, W)
        g = gt[b]  # (H, W)
        v = valid[b]  # (H, W)

        pred_valid = v & (p > 0)  # unmasked AND GT-valid

        if pred_valid.sum() == 0 or v.sum() == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            fscores.append(0.0)
            continue

        # Ratio only on valid pred pixels — no spurious zeros
        ratio = torch.maximum(
            p[pred_valid] / g[pred_valid],
            g[pred_valid] / p[pred_valid],
        )  # (N,)  N = #pred_valid pixels

        accurate = ratio < delta  # (N,) bool

        precision = accurate.float().mean()
        recall = accurate.float().sum() / (v.float().sum() + 1e-8)
        fscore = 2 * precision * recall / (precision + recall + 1e-8)

        precisions.append(precision.item())
        recalls.append(recall.item())
        fscores.append(fscore.item())

    return {"precisions": precisions, "recalls": recalls, "fscores": fscores}


CANNY_LOW = 0.8
CANNY_HIGH = 0.9
DILATION_PX = 0


def edge_aware_chamfer(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    rgb: torch.Tensor,
    m_cam_from_uv: torch.Tensor,
    valid_mask: torch.Tensor,
    canny_low: float = CANNY_LOW,
    canny_high: float = CANNY_HIGH,
    dilation_px: int = DILATION_PX,
) -> list[float]:
    """
    Compute the Edge-Aware Chamfer Distance.

    1. Find edges in ``rgb`` via Canny (must be very strict to find only the main edges).
    2. Combine edge mask with ``valid_mask`` and depth > 0 for both pred/GT.
    3. Unproject masked pixels to 3D.
    4. Return bidirectional Chamfer Distance.

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
