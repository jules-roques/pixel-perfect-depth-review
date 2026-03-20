"""
Visualization: bar plots for benchmark results.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Colour palette (harmonious, modern)
# ---------------------------------------------------------------------------
COLORS = {
    "PPD": "#6C5CE7",          # purple
    "DAv2": "#00B894",         # teal
    "DAv2-Cleaned": "#E17055", # coral
}


def generate_plots(results_json_path: str, output_path: str = "results.png"):
    """
    Read ``results.json`` and produce two side-by-side bar plots:

    - Average Inference Time (ms)
    - Average Edge-Aware Chamfer Distance

    Saves as ``output_path``.
    """
    with open(results_json_path, "r") as f:
        results = json.load(f)

    model_names = list(results["inference_time"].keys())
    avg_times = [np.mean(results["inference_time"][m]) for m in model_names]
    avg_chamfer = [np.nanmean(results["chamfer_distance"][m]) for m in model_names]
    colors = [COLORS.get(m, "#636e72") for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Bar Plot 1: Inference Time --------------------------------------
    ax = axes[0]
    bars = ax.bar(model_names, avg_times, color=colors, edgecolor="white",
                  linewidth=1.2, width=0.5)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Average Inference Time", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    # ---- Bar Plot 2: Chamfer Distance ------------------------------------
    ax = axes[1]
    bars = ax.bar(model_names, avg_chamfer, color=colors, edgecolor="white",
                  linewidth=1.2, width=0.5)
    ax.set_ylabel("Chamfer Distance", fontsize=12)
    ax.set_title("Average Edge-Aware Chamfer Distance", fontsize=14,
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, avg_chamfer):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plots to {output_path}")
