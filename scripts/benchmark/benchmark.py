import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from ppdr.models import PPD, DAv2, DAv2Cleaned, MoGe
from ppdr.models.interface import DepthModel
from ppdr.utils.data_loader import HypersimLoader
from ppdr.utils.geometry import euclidean_to_planar
from ppdr.utils.metrics import edge_aware_chamfer
from ppdr.utils.types import EvalMetrics, ImageContext, SafeDepth
from ppdr.vendor.ppd.utils.align_depth_func import recover_metric_depth_ransac
from ppdr.vendor.ppd.utils.transform import resize_keep_aspect
from ppdr.visualization import generate_plots

DATA_ROOT: str = "data/hypersim_test_set"
OUTPUT_DIR: str = "results"
WARMUP: int = 5
MOGE_CHECKPOINT: str = "checkpoints/moge2.pt"
MAX_ITERATIONS: int = 10000


@dataclass
class ModelResults:
    metrics: dict[str, EvalMetrics]

    def get_avg_inference_time(self) -> float:
        return float(np.mean([m.inference_time for m in self.metrics.values()]))

    def get_avg_chamfer_distance(self) -> float:
        valid_cd = [
            m.chamfer_distance
            for m in self.metrics.values()
            if m.chamfer_distance is not None
        ]
        return float(np.mean(valid_cd)) if valid_cd else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    return parser.parse_args()


def main() -> None:
    print("Starting benchmark...")
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    loader = HypersimLoader(args.data_root, max_images=args.max_images)

    # We store all contexts in the RAM, which is not ideal for large datasets
    print("Preparing image contexts (MoGe extraction)...")
    moge = MoGe(MOGE_CHECKPOINT, device)
    contexts = prepare_all_contexts(loader, moge, device)

    print("Loading and evaluating models...")
    models = load_all_inference_models(device)
    first_img = next(iter(loader), (None,))[0]
    results = evaluate_all(models, contexts, first_img, args.warmup)

    print("Saving and plotting...")
    save_and_plot(results, args.output_dir)


def load_all_inference_models(device: torch.device) -> dict[str, DepthModel]:
    device_str = str(device)
    ppd = PPD(device=device_str)
    dav2 = DAv2(device=device_str)
    models: dict[str, DepthModel] = {
        "PPD": ppd,
        "DAv2": dav2,
        "DAv2-Cleaned": DAv2Cleaned(dav2),
    }
    return models


def prepare_all_contexts(
    loader: HypersimLoader, moge: MoGe, device: torch.device
) -> list[ImageContext]:
    contexts = []
    for idx, (bgr, gt_depth, name) in enumerate(loader):
        if idx >= MAX_ITERATIONS:
            break
        ctx = prepare_image_context(name, bgr, gt_depth, moge, device)
        contexts.append(ctx)
    return contexts


def warmup_model(
    model: DepthModel, warmup_img: np.ndarray | None, warmup_iters: int
) -> None:
    if warmup_img is not None:
        execute_warmup(model, warmup_img, warmup_iters)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def execute_warmup(model: Any, bgr: np.ndarray, iters: int) -> None:
    iters = min(iters, MAX_ITERATIONS)
    for _ in range(iters):
        model.predict(bgr)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_all(
    models: dict[str, DepthModel],
    contexts: list[ImageContext],
    warmup_img: np.ndarray | None,
    warmup_iters: int,
) -> dict[str, ModelResults]:
    results: dict[str, ModelResults] = {}

    for model_name, model in models.items():
        print(f"Warming up model: {model_name}...")
        warmup_model(model, warmup_img, warmup_iters)

        print(f"Evaluating model: {model_name}...")
        metrics = {}
        for ctx in contexts:
            metrics[ctx.name] = evaluate_model_on_image(model, ctx)
        results[model_name] = ModelResults(metrics=metrics)

    return results


def save_and_plot(results: dict[str, ModelResults], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nSummary")
    for name, model_results in results.items():
        avg_t = model_results.get_avg_inference_time()
        avg_cd = model_results.get_avg_chamfer_distance()
        print(f"{name:16s} Avg time: {avg_t:8.1f} ms | Avg Chamfer: {avg_cd:.6f}")

    generate_plots(json_path, os.path.join(output_dir, "results.png"))


def prepare_image_context(
    name: str,
    bgr: np.ndarray,
    gt_euclidean: SafeDepth,
    moge: MoGe,
    device: torch.device,
) -> ImageContext:
    resized_bgr = resize_keep_aspect(bgr)
    h, w = resized_bgr.shape[:2]

    resized_gt_euc = cv2.resize(
        gt_euclidean.depth, (w, h), interpolation=cv2.INTER_NEAREST
    )
    resized_valid = cv2.resize(
        gt_euclidean.valid_mask.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    intrinsics = moge.infer_intrinsics(resized_bgr)

    gt_planar = euclidean_to_planar(resized_gt_euc, intrinsics)

    return ImageContext(
        name=name,
        original_bgr=bgr,
        resized_bgr=resized_bgr,
        gt_depth=SafeDepth(depth=gt_planar, valid_mask=resized_valid),
        intrinsics=intrinsics,
        device=device,
    )


def evaluate_model_on_image(
    model: Any,
    ctx: ImageContext,
) -> EvalMetrics:
    t0 = time.time()
    pred_depth, elapsed = time_prediction(model, ctx.original_bgr)
    t1 = time.time()
    print(f"Inference time: {t1 - t0}")

    t2 = time.time()
    h, w = ctx.resized_bgr.shape[:2]
    res_pred = cv2.resize(pred_depth, (w, h), interpolation=cv2.INTER_LINEAR)

    metric_pred = recover_metric_depth_ransac(
        res_pred, ctx.moge_depth.depth, ctx.moge_depth.valid_mask
    )
    cd = edge_aware_chamfer(
        metric_pred,
        ctx.gt_depth.depth,
        ctx.resized_bgr,
        ctx.intrinsics,
        ctx.gt_depth.valid_mask,
        device=ctx.device,
    )
    t3 = time.time()
    print(f"Measure Time: {t3 - t2}")

    valid_cd = cd if np.isfinite(cd) else None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return EvalMetrics(inference_time=elapsed, chamfer_distance=valid_cd)


def time_prediction(model: DepthModel, bgr: np.ndarray) -> tuple[np.ndarray, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    depth = model.predict(bgr)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return depth, (time.perf_counter() - start) * 1000.0


if __name__ == "__main__":
    main()
