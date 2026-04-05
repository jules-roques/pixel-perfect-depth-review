import time
from itertools import islice

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ppdr.models.interface import DepthModel
from ppdr.utils.dataset import HypersimDataset
from ppdr.utils.metrics import Metrics, depth_fscore, edge_aware_chamfer


class Benchmark:
    """Runs depth-model benchmarks on a given dataset."""

    # Canny edge detection parameters (strict to find main edges)
    CANNY_LOW = 0.2
    CANNY_HIGH = 0.5
    DILATION_PX = 1

    # Normalised Canny thresholds for depth edge detection
    DEPTH_CANNY_LOW = 0.2
    DEPTH_CANNY_HIGH = 0.5

    # Ratio threshold for fscore computation
    DELTA = 1.05

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
    ) -> dict:
        """Evaluate *models* and return per-model results.

        Args:
            models: mapping of model name → DepthModel instance.
            max_images: optional cap on the number of dataset images to
                evaluate. When ``None``, every image in the dataset is used.
            warmup_images: number of warm-up forward passes per model.

        Returns:
            A dict mapping model names to their measured metrics and the delta value.
        """
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        results = {}
        for model_name, model in models.items():
            print(f"Evaluating model: {model_name}...")
            Benchmark._warmup_model(model, loader, warmup_batches)
            results[model_name] = Benchmark._evaluate_one_model(
                model, loader, max_batches
            )

        results = {name: res.to_dict() for name, res in results.items()}
        results["delta"] = Benchmark.DELTA

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
    ) -> Metrics:

        metrics = Metrics()
        total_batches = len(loader) if max_batches is None else max_batches
        for batch in tqdm(islice(loader, max_batches), total=total_batches):
            batch_metrics = Benchmark._evaluate_model_on_batch(model, batch)
            metrics.extend(batch_metrics)

        return metrics

    @staticmethod
    def _evaluate_model_on_batch(
        model: DepthModel,
        batch: dict[str, torch.Tensor],
    ) -> Metrics:
        images = batch["image"].to(model.device)
        true_depths = batch["depth"].to(model.device)
        gt_valid_masks = batch["valid_mask"].to(model.device)
        ndc_to_cams = batch["ndc_to_cam"].to(model.device)
        B = images.shape[0]

        pred_depths, pred_valid_masks, time_elapsed = Benchmark._time_prediction(
            model, images
        )

        metric_depth = model.align_pred_on_metric_depth(
            pred_depths, true_depths, gt_valid_masks & pred_valid_masks
        )

        chamfer_distances = edge_aware_chamfer(
            pred_depth=metric_depth,
            gt_depth=true_depths,
            rgb=images,
            m_cam_from_uv=ndc_to_cams,
            valid_mask=gt_valid_masks,
            pred_mask=pred_valid_masks,
            canny_low=Benchmark.CANNY_LOW,
            canny_high=Benchmark.CANNY_HIGH,
            dilation_px=Benchmark.DILATION_PX,
            depth_canny_low=Benchmark.DEPTH_CANNY_LOW,
            depth_canny_high=Benchmark.DEPTH_CANNY_HIGH,
        )

        fscores_dict = depth_fscore(
            metric_depth,
            true_depths,
            gt_valid_masks,
            pred_valid_masks,
            delta=Benchmark.DELTA,
        )

        return Metrics(
            chamfer_distances=chamfer_distances,
            inference_times=[time_elapsed / B] * B,
            precisions=fscores_dict["precisions"],
            recalls=fscores_dict["recalls"],
            fscores=fscores_dict["fscores"],
        )

    @staticmethod
    def _time_prediction(
        model: DepthModel, rgb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        depth, mask = model.predict(rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return depth, mask, (time.perf_counter() - start) * 1000.0
