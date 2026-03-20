"""
Depth Estimation Benchmark
==========================

Benchmark three depth estimation configurations on a Hypersim test subset:
  - PPD   (Pixel-Perfect Depth)
  - DAv2  (Depth Anything v2, ViT-L)
  - DAv2-Cleaned  (DAv2 + cross-gradient flying-pixel filter)

Measures:
  1. Inference Time (ms)
  2. Edge-Aware Chamfer Distance

Usage:
    python benchmark.py [--max_images N] [--output_dir DIR] [--warmup W]
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure external PPD code is importable
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "external", "pixel-perfect-depth")
)

from models.ppd_wrapper import PPDModel
from models.dav2_wrapper import DAv2Model, DAv2CleanedModel
from utils.data_loader import HypersimLoader
from utils.geometry import euclidean_to_planar
from utils.metrics import edge_aware_chamfer
from visualize import generate_plots

# MoGe for intrinsics + metric depth
from ppd.moge.model.v2 import MoGeModel
from ppd.utils.align_depth_func import recover_metric_depth_ransac
from ppd.utils.transform import resize_keep_aspect

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA_ROOT = "data/hypersim_test_set"
OUTPUT_DIR = "results"
WARMUP = 5
MOGE_CHECKPOINT = "checkpoints/model.pt"


def _load_moge(device: torch.device) -> MoGeModel:
    """Load the MoGe model (shared, for intrinsics + metric depth)."""
    moge = MoGeModel.from_pretrained(MOGE_CHECKPOINT).to(device).eval()
    return moge


@torch.no_grad()
def _get_moge_intrinsics_and_depth(
    moge: MoGeModel, resized_image_bgr: np.ndarray, device: torch.device
):
    """
    Run MoGe on the resized image to obtain metric depth, validity mask,
    and **denormalized** camera intrinsics.

    Returns:
        moge_depth:  (rH, rW) metric depth (numpy float32).
        moge_mask:   (rH, rW) boolean mask.
        intrinsics:  (3, 3) float32 numpy — denormalized.
    """
    rH, rW = resized_image_bgr.shape[:2]
    moge_rgb = cv2.cvtColor(resized_image_bgr, cv2.COLOR_BGR2RGB)
    moge_tensor = torch.tensor(
        moge_rgb / 255.0, dtype=torch.float32, device=device
    ).permute(2, 0, 1)
    moge_depth, moge_mask, intrinsics = moge.infer(moge_tensor)

    # Fill invalid regions with max valid depth (as in run_point_cloud.py)
    moge_depth[~moge_mask] = moge_depth[moge_mask].max()

    # Denormalize intrinsics (MoGe returns normalized wrt image size)
    intrinsics[0, 0] *= rW
    intrinsics[1, 1] *= rH
    intrinsics[0, 2] *= rW
    intrinsics[1, 2] *= rH

    return moge_depth, moge_mask, intrinsics


def _time_predict(model, image_bgr: np.ndarray):
    """Time a single prediction and return (depth, resized_image, elapsed_ms)."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    depth, resized = model.predict(image_bgr)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return depth, resized, elapsed_ms


def warmup(model, image_bgr: np.ndarray, n: int = WARMUP):
    """Run warm-up inferences to initialise CUDA kernels."""
    for _ in range(n):
        model.predict(image_bgr)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Depth Estimation Benchmark")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit number of images (default: all)",
    )
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load models -------------------------------------------------------
    print("Loading PPD …")
    ppd = PPDModel(device=str(device))
    print("Loading DAv2 …")
    dav2 = DAv2Model(device=str(device))
    dav2_cleaned = DAv2CleanedModel(dav2)

    print("Loading MoGe …")
    moge = _load_moge(device)

    models = {
        "PPD": ppd,
        "DAv2": dav2,
        "DAv2-Cleaned": dav2_cleaned,
    }

    # ---- Data loader -------------------------------------------------------
    loader = HypersimLoader(args.data_root, max_images=args.max_images)
    print(f"Benchmark: {len(loader)} images from {args.data_root}")

    # ---- Warm-up (use the first image) -------------------------------------
    print(f"Warm-up ({args.warmup} iterations per model) …")
    first_img = None
    for img, _, _, _ in loader:
        first_img = img
        break
    if first_img is not None:
        for name, model in models.items():
            warmup(model, first_img, args.warmup)
            print(f"  {name} warm-up done.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- Evaluation loop ---------------------------------------------------
    results = {
        "inference_time": {k: [] for k in models},
        "chamfer_distance": {k: [] for k in models},
        "per_image": [],
    }

    for idx, (image_bgr, gt_depth_euc, gt_valid, entry_name) in enumerate(loader):
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"Processing image {idx + 1}/{len(loader)} …")

        per_image = {"name": entry_name}

        for name, model in models.items():
            # ---------- Inference + timing ----------------------------------
            depth_pred, resized_img, elapsed = _time_predict(model, image_bgr)
            results["inference_time"][name].append(elapsed)
            per_image[f"{name}_time_ms"] = elapsed

            # ---------- MoGe intrinsics + metric alignment ------------------
            # Resize GT image the same way as the model to get the right
            # intrinsics for this resolution.
            resized_for_moge = resize_keep_aspect(image_bgr)
            rH, rW = resized_for_moge.shape[:2]

            moge_depth, moge_mask, intrinsics = _get_moge_intrinsics_and_depth(
                moge, resized_for_moge, device
            )

            # Resize predicted depth to match MoGe resolution
            depth_pred_resized = cv2.resize(
                depth_pred, (rW, rH), interpolation=cv2.INTER_LINEAR
            )

            # Align predicted (relative) depth → metric depth
            metric_pred = recover_metric_depth_ransac(
                depth_pred_resized, moge_depth, moge_mask
            )

            # ---------- GT: Euclidean → Planar Z ----------------------------
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            # Resize GT depth + mask to the same resolution
            gt_depth_resized = cv2.resize(
                gt_depth_euc, (rW, rH), interpolation=cv2.INTER_NEAREST
            )
            gt_valid_resized = cv2.resize(
                gt_valid.astype(np.uint8), (rW, rH), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            gt_planar = euclidean_to_planar(gt_depth_resized, fx, fy, cx, cy)

            # ---------- Edge-Aware Chamfer Distance -------------------------
            cd = edge_aware_chamfer(
                metric_pred,
                gt_planar,
                resized_for_moge,
                intrinsics,
                gt_valid_resized,
            )
            results["chamfer_distance"][name].append(cd if np.isfinite(cd) else None)
            per_image[f"{name}_chamfer"] = cd if np.isfinite(cd) else None

            # Clear GPU cache between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["per_image"].append(per_image)

    # ---- Save results ------------------------------------------------------
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved raw results to {json_path}")

    # ---- Summary -----------------------------------------------------------
    print("\n=== Summary ===")
    for name in models:
        avg_t = np.mean(results["inference_time"][name])
        valid_cd = [c for c in results["chamfer_distance"][name] if c is not None]
        avg_cd = np.mean(valid_cd) if valid_cd else float("nan")
        print(f"  {name:16s}  Avg time: {avg_t:8.1f} ms  |  Avg Chamfer: {avg_cd:.6f}")

    # ---- Plots -------------------------------------------------------------
    plot_path = os.path.join(args.output_dir, "results.png")
    generate_plots(json_path, plot_path)


if __name__ == "__main__":
    main()
