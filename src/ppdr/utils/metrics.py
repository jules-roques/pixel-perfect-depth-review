from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial import KDTree

from ppdr.utils.geometry import (
    depth_canny_edge_mask,
    depth_to_point_cloud,
    edge_mask,
)


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

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "chamfer_distances": self.chamfer_distances,
            "inference_times": self.inference_times,
            "precisions": self.precisions,
            "recalls": self.recalls,
            "fscores": self.fscores,
        }


def depth_fscore(
    pred: torch.Tensor,  # (B, H, W) metric depth
    gt: torch.Tensor,  # (B, H, W)
    valid: torch.Tensor,  # (B, H, W) GT valid mask
    pred_mask: torch.Tensor,  # (B, H, W) False = flying pixel
    delta: float,
) -> dict[str, list[float]]:
    B = pred.shape[0]
    precisions, recalls, fscores = [], [], []

    for b in range(B):
        p = pred[b]
        g = gt[b]
        v = valid[b]

        pred_valid = v & pred_mask[b]

        if pred_valid.sum() == 0 or v.sum() == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            fscores.append(0.0)
            continue

        ratio = torch.maximum(
            p[pred_valid] / g[pred_valid],
            g[pred_valid] / p[pred_valid],
        )
        accurate = ratio < delta
        precision = accurate.float().sum() / (pred_mask[b].float().sum() + 1e-8)
        recall = accurate.float().sum() / (v.float().sum() + 1e-8)
        fscore = 2 * precision * recall / (precision + recall + 1e-8)

        precisions.append(precision.item())
        recalls.append(recall.item())
        fscores.append(fscore.item())

    return {"precisions": precisions, "recalls": recalls, "fscores": fscores}


def edge_aware_chamfer(
    *,
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    rgb: torch.Tensor,
    m_cam_from_uv: torch.Tensor,
    valid_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    canny_low: float,
    canny_high: float,
    dilation_px: int,
    depth_canny_low: float = 0.05,
    depth_canny_high: float = 0.1,
) -> list[float]:
    """
    Compute the Edge-Aware Chamfer Distance.

    GT point cloud is sampled at **RGB edges** (where depth discontinuities
    *should* be), while the predicted point cloud is sampled at **depth-
    gradient edges** (where the model *actually* places discontinuities).
    This makes the metric sensitive to edge sharpness: a model that smooths
    edges will have depth discontinuities displaced from the true RGB
    boundaries, yielding a larger chamfer distance.

    Args:
        pred_depth: (B, H, W) predicted *planar* depth.
        gt_depth:   (B, H, W) ground-truth *planar* depth.
        rgb:        (B, 3, H, W) float image (RGB).
        m_cam_from_uv: Camera intrinsics matrix mapped to NDC.
        valid_mask: (B, H, W) boolean mask indicating valid GT pixels.
        pred_mask:  (B, H, W) boolean mask indicating valid predicted pixels.
        canny_low, canny_high: Canny edge thresholds for RGB edges.
        dilation_px: Dilation radius for the edge masks.
        depth_canny_low, depth_canny_high: Normalised Canny thresholds for
            detecting edges in the predicted depth map.

    Returns:
        Per-image Chamfer Distance values.
    """

    # GT edges: from RGB — where discontinuities SHOULD be
    rgb_edges = edge_mask(rgb, canny_low, canny_high, dilation_px)
    gt_combined = valid_mask & rgb_edges

    # Pred edges: from depth gradient — where discontinuities ARE
    pred_edges = depth_canny_edge_mask(
        pred_depth,
        canny_low=depth_canny_low,
        canny_high=depth_canny_high,
        dilation_px=dilation_px,
    )
    pred_combined = pred_mask & pred_edges

    B = pred_depth.shape[0]
    chamfer_distances = []
    for i in range(B):
        pred_pts = depth_to_point_cloud(
            pred_depth[i], m_cam_from_uv[i], pred_combined[i]
        )
        gt_pts = depth_to_point_cloud(gt_depth[i], m_cam_from_uv[i], gt_combined[i])

        # Normalise point clouds by the median valid ground-truth depth
        # to ensure the metric is scale-invariant across different scenes
        valid_z = gt_depth[i][valid_mask[i]]
        if valid_z.numel() > 0:
            scale = valid_z.median().item()
            if scale > 1e-4:
                pred_pts = pred_pts / scale
                gt_pts = gt_pts / scale

        if len(pred_pts) == 0 or len(gt_pts) == 0:
            chamfer_distances.append(float("nan"))
            continue

        chamfer_distances.append(chamfer_distance(pred_pts, gt_pts))

    return chamfer_distances


# KDTrees on GPU are not easy to use, so we do it on CPU
@torch.no_grad()
def chamfer_distance(A: torch.Tensor, B: torch.Tensor) -> float:
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
