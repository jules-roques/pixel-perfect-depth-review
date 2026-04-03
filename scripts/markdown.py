import argparse
import json
import os

import numpy as np
import prettytable as pt

from ppdr.utils.metrics import Metrics


def load_results(json_path: str) -> dict[str, Metrics]:
    with open(json_path) as f:
        raw = json.load(f)
    return {name: Metrics(**fields) for name, fields in raw.items()}


def remove_nan_values(results: dict[str, Metrics]) -> None:
    for metrics in results.values():
        metrics.chamfer_distances = [
            d for d in metrics.chamfer_distances if not np.isnan(d)
        ]
        metrics.precisions = [p for p in metrics.precisions if not np.isnan(p)]
        metrics.recalls = [r for r in metrics.recalls if not np.isnan(r)]
        metrics.fscores = [f for f in metrics.fscores if not np.isnan(f)]


def format_statistics(data: list[float], decimals: int = 4) -> str:
    if not data:
        return "N/A"

    mean_value = np.mean(data)
    std_value = np.std(data)
    return f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"


def build_depth_scores_table(results: dict[str, Metrics]) -> str:
    table = pt.PrettyTable()
    table.set_style(pt.TableStyle.MARKDOWN)
    table.field_names = ["Model", "Precision", "Recall", "F-score"]

    for model_name, metrics in results.items():
        table.add_row(
            [
                model_name,
                format_statistics(metrics.precisions),
                format_statistics(metrics.recalls),
                format_statistics(metrics.fscores),
            ]
        )

    return table.get_formatted_string()


def build_inference_time_table(results: dict[str, Metrics]) -> str:
    table = pt.PrettyTable()
    table.set_style(pt.TableStyle.MARKDOWN)
    table.field_names = ["Model", "Inference Time (ms)"]

    for model_name, metrics in results.items():
        table.add_row(
            [model_name, format_statistics(metrics.inference_times, decimals=2)]
        )

    return table.get_formatted_string()


def build_chamfer_distance_table(results: dict[str, Metrics]) -> str:
    table = pt.PrettyTable()
    table.set_style(pt.TableStyle.MARKDOWN)
    table.field_names = ["Model", "Chamfer Distance"]

    for model_name, metrics in results.items():
        table.add_row([model_name, format_statistics(metrics.chamfer_distances)])

    return table.get_formatted_string()


def save_markdown_report(results: dict[str, Metrics], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "metrics_report.md")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("# Model Evaluation Metrics\n\n")

        file.write("## Depth Score Metrics (δ = 1.25)\n\n")
        file.write(build_depth_scores_table(results))
        file.write("\n\n")

        file.write("## Per-Image Inference Time\n\n")
        file.write(build_inference_time_table(results))
        file.write("\n\n")

        file.write("## Edge-Aware Chamfer Distance\n\n")
        file.write(build_chamfer_distance_table(results))
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", type=str, default="results/results.json")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    results = load_results(args.results_json)
    remove_nan_values(results)

    save_markdown_report(results, args.output_dir)
    print(f"Markdown report generated in the '{args.output_dir}' folder.")


if __name__ == "__main__":
    main()
