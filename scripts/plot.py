import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from ppdr.utils.metrics import Metrics

# ── I/O ────────────────────────────────────────────────────────────────────────


def load_results(json_path: str) -> dict[str, Metrics]:
    """Deserialise a results JSON produced by save_results() into Metrics objects."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", type=str, default="results/results.json")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    results = load_results(args.results_json)
    remove_nan_values(results)
    generate_plots(results, args.output_dir)
    print(f"Plots generated in '{args.output_dir}' folder")


# ── Palette: one colour per model, consistent across all plots ─────────────────
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _model_colors(model_names: list[str]) -> dict[str, str]:
    return {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(model_names)}


def _boxplot(
    ax: plt.Axes,
    data: list[list[float]],  # one list per model
    model_names: list[str],
    colors: dict[str, str],
    title: str,
    ylabel: str,
    formatter: ticker.Formatter | None = None,
    use_log: bool = False,
) -> None:
    positions = np.arange(len(model_names))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        notch=False,
        showfliers=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, linestyle="none", alpha=0.4),
    )
    for patch, name in zip(bp["boxes"], model_names, strict=True):
        c = colors[name]
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    for flier, name in zip(bp["fliers"], model_names, strict=True):
        flier.set_markerfacecolor(colors[name])
        flier.set_markeredgecolor(colors[name])

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    if use_log:
        ax.set_yscale("symlog", linthresh=1.0)
        

    if formatter is not None:
        ax.yaxis.set_major_formatter(formatter)


def generate_plots(
    results: dict[str, "Metrics"],
    output_dir: str,
) -> None:
    """
    Generate and save three box-plot figures for model comparison.

    Figures produced
    ----------------
    1. depth_scores.png    — precision / recall / F-score (δ=1.05)
    2. inference_time.png  — per-image inference time (ms)
    3. chamfer_distance.png — edge-aware Chamfer distance
    """
    os.makedirs(output_dir, exist_ok=True)
    model_names = list(results.keys())
    colors = _model_colors(model_names)

    # ── 1. Depth scores ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    fig.suptitle(
        "Depth Score Metrics (δ = 1.05)", fontsize=13, fontweight="bold", y=1.01
    )

    score_cfg = [
        ("precisions", "Precision", axes[0]),
        ("recalls", "Recall", axes[1]),
        ("fscores", "F-score", axes[2]),
    ]
    pct_fmt = ticker.FuncFormatter(lambda x, _: f"{x:.2f}")

    for attr, title, ax in score_cfg:
        _boxplot(
            ax,
            data=[getattr(results[m], attr) for m in model_names],
            model_names=model_names,
            colors=colors,
            title=title,
            ylabel="Score",
            formatter=pct_fmt,
        )
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "depth_scores.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ── 2. Inference time ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("Per-Image Inference Time", fontsize=13, fontweight="bold")

    _boxplot(
        ax,
        data=[results[m].inference_times for m in model_names],
        model_names=model_names,
        colors=colors,
        title="Inference Time",
        ylabel="Time (ms)",
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "inference_time.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # ── 3. Chamfer distance ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("Edge-Aware Chamfer Distance", fontsize=13, fontweight="bold")

    _boxplot(
        ax,
        data=[results[m].chamfer_distances for m in model_names],
        model_names=model_names,
        colors=colors,
        title="Chamfer Distance",
        ylabel="Distance",
        use_log=True,
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "chamfer_distance.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
