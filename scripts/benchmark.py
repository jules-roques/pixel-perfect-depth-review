import argparse
import json
import os

import torch

from ppdr.models import PPD, DAv2, DAv2Cleaned
from ppdr.models.interface import DepthModel
from ppdr.utils.benchmark import Benchmark
from ppdr.utils.dataset import HypersimDataset
from ppdr.utils.metrics import Metrics
from ppdr.utils.reader import HypersimReader

DATA_ROOT: str = "data/hypersim_test_set"
OUTPUT_DIR: str = "results"
BATCH_SIZE: int = 2
WARMUP_BATCHES: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--warmup_batches", type=int, default=WARMUP_BATCHES)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


def load_all_inference_models(device: torch.device) -> dict[str, DepthModel]:
    ppd = PPD(device=device)
    dav2 = DAv2(device=device)
    dav2_cleaned = DAv2Cleaned(device=device)
    models: dict[str, DepthModel] = {
        "PPD": ppd,
        "DAv2": dav2,
        "DAv2-Cleaned": dav2_cleaned,
    }
    return models


def main() -> None:
    print("Starting benchmark...")
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading models...")
    models = load_all_inference_models(device)

    print("Running benchmark on dataset...")
    dataset = HypersimDataset(HypersimReader(args.data_root))
    benchmark = Benchmark(dataset=dataset)
    results = benchmark.run(
        models=models,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        warmup_batches=args.warmup_batches,
    )

    print_summary(results)
    save_results(results, args.output_dir)


def print_summary(results: dict[str, Metrics]) -> None:
    for name, model_results in results.items():
        print(f"\n{name}")
        model_results.print_summary()


def save_results(results: dict[str, Metrics], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "results.json")
    data = {name: res.to_dict() for name, res in results.items()}
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
