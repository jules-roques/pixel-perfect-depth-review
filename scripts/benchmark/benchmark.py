import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from ppdr.models import PPD, DAv2, DAv2Cleaned
from ppdr.models.interface import DepthModel
from ppdr.utils.dataset import HypersimDataset
from ppdr.utils.metrics import edge_aware_chamfer
from ppdr.utils.reader import HypersimReader
from ppdr.vendor.ppd.utils.align_depth_func import recover_metric_depth_ransac
from ppdr.visualization import generate_plots

DATA_ROOT: str = "data/hypersim_test_set"
OUTPUT_DIR: str = "results"
WARMUP: int = 5
MAX_ITERATIONS: int = 10000


@dataclass
class ModelResults:
    inference_times: list[float]
    chamfer_distances: list[float]

    def get_avg_inference_time(self) -> float:
        return float(np.mean(self.inference_times))

    def get_avg_chamfer_distance(self) -> float:
        return float(np.mean(self.chamfer_distances))


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
    reader = HypersimReader(args.data_root)
    dataset = HypersimDataset(reader)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Loading and evaluating models...")
    models = load_all_inference_models(device)
    first_img, _, _, _ = dataset[0]
    results = evaluate_all_models(models, loader, first_img, args.warmup)

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


def warmup_model(
    model: DepthModel, warmup_img: torch.Tensor, warmup_iters: int
) -> None:
    iters = min(warmup_iters, MAX_ITERATIONS)
    for _ in range(iters):
        model.predict(warmup_img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_all_models(
    models: dict[str, DepthModel],
    loader: DataLoader,
    warmup_img: torch.Tensor,
    warmup_iters: int,
) -> dict[str, ModelResults]:

    results: dict[str, ModelResults] = {}
    for model_name, model in models.items():
        print(f"Evaluating model: {model_name}...")
        warmup_model(model, warmup_img, warmup_iters)
        results[model_name] = evaluate_one_model(model, loader)

    return results


def evaluate_one_model(
    model: DepthModel,
    loader: DataLoader,
) -> ModelResults:
    chamfer_distances: list[float] = []
    inference_times: list[float] = []
    for entry in loader:
        chamfer_distance, inference_time = evaluate_model_on_entry(model, entry)
        chamfer_distances.append(chamfer_distance)
        inference_times.append(inference_time)
    return ModelResults(
        chamfer_distances=chamfer_distances, inference_times=inference_times
    )


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


def evaluate_model_on_entry(
    model: DepthModel,
    entry: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[float, float]:
    image, true_depth, valid_mask, ndc_to_cam = entry
    pred_depth, elapsed = time_prediction(model, image)

    metric_pred = recover_metric_depth_ransac(pred_depth, true_depth, valid_mask)
    cd = edge_aware_chamfer(
        metric_pred,
        true_depth,
        image,
        ndc_to_cam,
        valid_mask,
    )

    return cd, elapsed


def time_prediction(model: DepthModel, rgb: torch.Tensor) -> tuple[torch.Tensor, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    depth = model.predict(rgb)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return depth, (time.perf_counter() - start) * 1000.0


if __name__ == "__main__":
    main()
