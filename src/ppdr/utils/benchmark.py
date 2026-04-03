import time
from itertools import islice

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppdr.models.interface import DepthModel
from ppdr.utils.dataset import HypersimDataset
from ppdr.utils.metrics import edge_aware_chamfer


class Benchmark:
    """Runs depth-model benchmarks on a given dataset."""

    def __init__(self, dataset: HypersimDataset) -> None:
        self.dataset = dataset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        models: dict[str, DepthModel],
        batch_size: int,
        max_batches: int | None = None,
        warmup_batches: int = 1,
    ) -> dict[str, dict[str, list[float]]]:
        """Evaluate *models* and return per-model results.

        Args:
            models: mapping of model name → DepthModel instance.
            max_images: optional cap on the number of dataset images to
                evaluate. When ``None``, every image in the dataset is used.
            warmup_images: number of warm-up forward passes per model.

        Returns:
            A dict mapping model names to their measured metrics.
        """
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        results = Benchmark._evaluate_all_models(
            models, loader, max_batches, warmup_batches
        )

        return results

    @staticmethod
    def _evaluate_all_models(
        models: dict[str, DepthModel],
        loader: DataLoader,
        max_batches: int | None,
        warmup_batches: int,
    ) -> dict[str, dict[str, list[float]]]:
        results = {}
        for model_name, model in models.items():
            print(f"Evaluating model: {model_name}...")
            Benchmark._warmup_model(model, loader, warmup_batches)
            results[model_name] = Benchmark._evaluate_one_model(
                model, loader, max_batches
            )
        return results

    @staticmethod
    def _warmup_model(
        model: DepthModel, loader: DataLoader, warmup_batches: int
    ) -> None:
        for entry in islice(loader, warmup_batches):
            image = entry["image"].to(model.device)
            model.predict(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _evaluate_one_model(
        model: DepthModel,
        loader: DataLoader,
        max_batches: int | None,
    ) -> dict[str, list[float]]:

        metrics = {
            "chamfer_distances": [],
            "inference_times": [],
        }

        total_batches = len(loader) if max_batches is None else max_batches
        for entry in tqdm(islice(loader, max_batches), total=total_batches):
            chamfer_distance, inference_time = Benchmark._evaluate_model_on_entry(
                model, entry
            )
            metrics["chamfer_distances"].extend(chamfer_distance)
            metrics["inference_times"].append(inference_time)
        return metrics

    @staticmethod
    def _evaluate_model_on_entry(
        model: DepthModel,
        entry: dict[str, torch.Tensor],
    ) -> tuple[list[float], float]:
        image = entry["image"].to(model.device)
        true_depth = entry["depth"].to(model.device)
        valid_mask = entry["valid_mask"].to(model.device)
        ndc_to_cam = entry["ndc_to_cam"].to(model.device)

        pred, time_elapsed = Benchmark._time_prediction(model, image)
        metric_depth = model.align_pred_on_metric_depth(pred, true_depth, valid_mask)
        chamfer_distances = edge_aware_chamfer(
            metric_depth,
            true_depth,
            image,
            ndc_to_cam,
            valid_mask,
        )

        return chamfer_distances, time_elapsed

    @staticmethod
    def _time_prediction(
        model: DepthModel, rgb: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        depth = model.predict(rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return depth, (time.perf_counter() - start) * 1000.0
