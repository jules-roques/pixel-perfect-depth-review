import argparse
import json
import os

import numpy as np
import prettytable as pt

from ppdr.utils.metrics import Metrics


def load_results(json_path: str) -> tuple[float, dict[str, Metrics]]:
    with open(json_path) as file:
        raw_data = json.load(file)

    delta = raw_data.pop("delta")
    metrics_data = {name: Metrics(**fields) for name, fields in raw_data.items()}

    return delta, metrics_data


def remove_nan_values(metrics_data: dict[str, Metrics]) -> None:
    for metrics in metrics_data.values():
        metrics.chamfer_distances = [
            distance for distance in metrics.chamfer_distances if not np.isnan(distance)
        ]
        metrics.precisions = [
            precision for precision in metrics.precisions if not np.isnan(precision)
        ]
        metrics.recalls = [recall for recall in metrics.recalls if not np.isnan(recall)]
        metrics.fscores = [fscore for fscore in metrics.fscores if not np.isnan(fscore)]


def format_statistics(data: list[float], decimals: int = 4) -> str:
    if not data:
        return "N/A"

    mean_value = np.mean(data)
    std_value = np.std(data)
    return f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"


def build_depth_scores_table(metrics_data: dict[str, Metrics]) -> str:
    table = pt.PrettyTable()
    table.set_style(pt.TableStyle.MARKDOWN)
    table.field_names = ["Model", "Precision", "Recall", "F-score"]

    for model_name, metrics in metrics_data.items():
        table.add_row([
            model_name,
            format_statistics(metrics.precisions),
            format_statistics(metrics.recalls),
            format_statistics(metrics.fscores),
        ])

    return table.get_formatted_string()


def build_inference_time_table(metrics_data: dict[str, Metrics]) -> str:
    table = pt.PrettyTable()
    table.set_style(pt.TableStyle.MARKDOWN)
    table.field_names = ["Model", "Inference Time (ms)"]

    for model_name, metrics in metrics_data.items():
        table.add_row([
            model_name,
            format_statistics(metrics.inference_times, decimals=2),
        ])

    return table.get_formatted_string()


def build_chamfer_distance_table(metrics_data: dict[str, Metrics]) -> str:
    table = pt.PrettyTable()
    table.set_style(pt.TableStyle.MARKDOWN)
    table.field_names = ["Model", "Chamfer Distance"]

    for model_name, metrics in metrics_data.items():
        table.add_row([model_name, format_statistics(metrics.chamfer_distances)])

    return table.get_formatted_string()


def save_markdown_report(
    metrics_data: dict[str, Metrics], delta: float, output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "metrics_report.md")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("# Model Evaluation Metrics\n\n")

        file.write(f"## Depth Score Metrics (δ = {delta})\n\n")
        file.write(build_depth_scores_table(metrics_data))
        file.write("\n\n")

        file.write("## Per-Image Inference Time\n\n")
        file.write(build_inference_time_table(metrics_data))
        file.write("\n\n")

        file.write("## Edge-Aware Chamfer Distance\n\n")
        file.write(build_chamfer_distance_table(metrics_data))
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", type=str, default="results/results.json")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    delta, metrics_data = load_results(args.results_json)
    remove_nan_values(metrics_data)

    save_markdown_report(metrics_data, delta, args.output_dir)
    print(f"Markdown report generated in the '{args.output_dir}' folder.")


if __name__ == "__main__":
    main()
