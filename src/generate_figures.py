#!/usr/bin/env python3
"""
Manuscript Figure Generation Script
=================================
This script generates manuscript figures from the executable submission
code path.

Important:
    - Manuscript Figure 3 (training behavior) can be reproduced with the ``--paper-figure3`` option.
    - Manuscript Figure 4 (boxplot) can be reproduced with the ``--paper-figure4`` option.
    - Manuscript Figure 5 (paired differences) can be reproduced with the ``--paper-figure5`` option.
    - Manuscript Figure 6 (confusion matrices) can be reproduced with the ``--paper-figure6`` option.
    - All paper figures use the same core logic as the reference figures, ensuring alignment between the submission code and manuscript visuals.

Usage:
    python -m src.generate_figures
    python -m src.generate_figures --paper-figure3
    python -m src.generate_figures --paper-figure4
    python -m src.generate_figures --paper-figure5
    python -m src.generate_figures --paper-figure6
    python -m src.generate_figures --paper-figure3 --paper-figure4 --paper-figure5 --paper-figure6

Required:
    - Run training first to generate results in results/experiment_results/
    - matplotlib and seaborn for visualization
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
import warnings

WORKSPACE = Path(".")
MPL_DIR = WORKSPACE / ".matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

RESULTS_DIR = WORKSPACE / "results" / "experiment_results"
FIGURES_DIR = WORKSPACE / "figures"
PAPER_FIGURE3_OUTPUT = FIGURES_DIR / "Figure_3_training_behavior.png"
PAPER_FIGURE4_OUTPUT = FIGURES_DIR / "Figure_4_boxplot.png"
PAPER_FIGURE5_OUTPUT = FIGURES_DIR / "Figure_5_paired_differences.png"
PAPER_FIGURE6_OUTPUT = FIGURES_DIR / "Figure_6_confusion_matrices.png"
CLASS_LABELS = ["Sup-int", "Parabasal", "Koilocytes", "Dyskeratotic", "Metaplastic"]
PAPER_FIGURE3_SPECS = [
    ("cv_summary_baseline", "Baseline", "#4C72B0"),
    ("cv_summary_se", "Layer4 attention", "#55A868"),
    ("cv_summary_se_avgpool", "Avgpool attention", "#C44E52"),
]
PAPER_FIGURE6_SPECS = [
    ("cv_summary_baseline", "Baseline"),
    ("cv_summary_se", "Layer4 attention"),
    ("cv_summary_se_avgpool", "Avgpool attention"),
]

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate manuscript figures (Figure 3-6) from experiment results."
    )
    parser.add_argument(
        "--paper-figure3",
        action="store_true",
        help="Also generate the manuscript Figure 3 training behavior plot.",
    )
    parser.add_argument(
        "--paper-figure4",
        action="store_true",
        help="Also generate the manuscript Figure 4 boxplot.",
    )
    parser.add_argument(
        "--paper-figure5",
        action="store_true",
        help="Also generate the manuscript Figure 5 paired differences plot.",
    )
    parser.add_argument(
        "--paper-figure6",
        action="store_true",
        help="Also generate the manuscript Figure 6 confusion matrices.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing experiment result JSON files.",
    )
    parser.add_argument(
        "--paper-figure3-output",
        type=Path,
        default=PAPER_FIGURE3_OUTPUT,
        help="Output path for manuscript Figure 3.",
    )
    parser.add_argument(
        "--paper-figure4-output",
        type=Path,
        default=PAPER_FIGURE4_OUTPUT,
        help="Output path for manuscript Figure 4.",
    )
    parser.add_argument(
        "--paper-figure5-output",
        type=Path,
        default=PAPER_FIGURE5_OUTPUT,
        help="Output path for manuscript Figure 5.",
    )
    parser.add_argument(
        "--paper-output",
        type=Path,
        default=PAPER_FIGURE6_OUTPUT,
        help="Output path for manuscript Figure 6.",
    )
    return parser.parse_args()


def load_results(results_dir):
    """Load experiment results"""
    results = {}
    for summary_key, _title, *_rest in PAPER_FIGURE3_SPECS:
        path = results_dir / f"{summary_key}.json"
        with path.open("r", encoding="utf-8") as handle:
            results[summary_key] = json.load(handle)
    return results


def load_runs(path):
    """Load runs from a single model summary file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    runs = sorted(data["runs"], key=lambda item: (item["seed"], item["fold"]))
    return runs


def aggregate_row_normalized_confusion(summary):
    """Aggregate run-level confusion matrices and row-normalize them."""
    runs = summary.get("runs", [])
    if not runs:
        raise ValueError("Summary does not contain any cross-validation runs.")

    aggregate_cm = np.zeros((5, 5), dtype=float)
    for run in runs:
        aggregate_cm += np.array(run["val_metrics"]["confusion_matrix"], dtype=float)

    row_totals = aggregate_cm.sum(axis=1, keepdims=True)
    row_normalized = np.divide(
        aggregate_cm,
        row_totals,
        out=np.zeros_like(aggregate_cm),
        where=row_totals > 0,
    ) * 100.0
    aggregate_accuracy = float(np.trace(aggregate_cm) / aggregate_cm.sum() * 100.0)
    return row_normalized, aggregate_accuracy


def collect_history_arrays(summary):
    """Extract history arrays from 15 runs of a single model."""
    runs = summary.get("runs", [])
    if not runs:
        raise ValueError("Summary does not contain any cross-validation runs.")

    histories = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for run in runs:
        history = run.get("history", {})
        for key in histories:
            values = history.get(key, [])
            if not values:
                raise ValueError(f"Run history missing {key}")
            histories[key].append(values)

    return {key: np.array(values, dtype=float) for key, values in histories.items()}


def summarize_history(summary):
    """Compute mean ± SD across 15 runs for each epoch."""
    arrays = collect_history_arrays(summary)
    return {
        "train_loss_mean": arrays["train_loss"].mean(axis=0),
        "train_loss_std": arrays["train_loss"].std(axis=0),
        "val_loss_mean": arrays["val_loss"].mean(axis=0),
        "val_loss_std": arrays["val_loss"].std(axis=0),
        "train_acc_mean": arrays["train_acc"].mean(axis=0) * 100.0,
        "train_acc_std": arrays["train_acc"].std(axis=0) * 100.0,
        "val_acc_mean": arrays["val_acc"].mean(axis=0) * 100.0,
        "val_acc_std": arrays["val_acc"].std(axis=0) * 100.0,
    }


def generate_paper_figure3_training_behavior(results, output_path):
    """Generate manuscript Figure 3: training behavior with 3x2 subplot layout."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 500
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10

    fig, axes = plt.subplots(3, 2, figsize=(8.69, 9.6), sharex="col")
    epochs = np.arange(1, 16)

    for row_index, (summary_key, row_title, color) in enumerate(PAPER_FIGURE3_SPECS):
        if summary_key not in results:
            raise KeyError(f"Required summary missing for Figure 3: {summary_key}")
        stats = summarize_history(results[summary_key])
        loss_ax = axes[row_index, 0]
        acc_ax = axes[row_index, 1]

        loss_ax.plot(epochs, stats["train_loss_mean"], color=color, linewidth=1.8, label="Train")
        loss_ax.fill_between(
            epochs,
            stats["train_loss_mean"] - stats["train_loss_std"],
            stats["train_loss_mean"] + stats["train_loss_std"],
            color=color,
            alpha=0.10,
        )
        loss_ax.plot(
            epochs,
            stats["val_loss_mean"],
            color=color,
            linewidth=1.6,
            linestyle="--",
            label="Validation",
        )
        loss_ax.fill_between(
            epochs,
            stats["val_loss_mean"] - stats["val_loss_std"],
            stats["val_loss_mean"] + stats["val_loss_std"],
            color=color,
            alpha=0.06,
        )
        loss_ax.set_ylabel(f"{row_title}\nLoss")
        loss_ax.set_xlim(1, 15)
        loss_ax.set_ylim(bottom=0)
        loss_ax.set_xticks(np.arange(2, 16, 2))
        loss_ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

        acc_ax.plot(epochs, stats["train_acc_mean"], color=color, linewidth=1.8, label="Train")
        acc_ax.fill_between(
            epochs,
            stats["train_acc_mean"] - stats["train_acc_std"],
            stats["train_acc_mean"] + stats["train_acc_std"],
            color=color,
            alpha=0.10,
        )
        acc_ax.plot(
            epochs,
            stats["val_acc_mean"],
            color=color,
            linewidth=1.6,
            linestyle="--",
            label="Validation",
        )
        acc_ax.fill_between(
            epochs,
            stats["val_acc_mean"] - stats["val_acc_std"],
            stats["val_acc_mean"] + stats["val_acc_std"],
            color=color,
            alpha=0.06,
        )
        acc_ax.set_ylabel(f"{row_title}\nAccuracy (%)")
        acc_ax.set_xlim(1, 15)
        acc_ax.set_ylim(81, 101)
        acc_ax.set_xticks(np.arange(2, 16, 2))
        acc_ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    axes[0, 0].set_title("Loss")
    axes[0, 1].set_title("Accuracy")
    for row_index in range(2):
        axes[row_index, 0].tick_params(labelbottom=False)
        axes[row_index, 1].tick_params(labelbottom=False)

    axes[2, 0].set_xlabel("Epoch")
    axes[2, 1].set_xlabel("Epoch")
    axes[0, 1].legend(loc="lower right", frameon=True, fontsize=9)

    fig.suptitle("Mean Training Behavior Across 15 Matched Development Runs", fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.06, top=0.92, hspace=0.10, wspace=0.28)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Manuscript Figure 3 saved to {output_path}")


def generate_paper_figure4_boxplot(results, output_path):
    """Generate manuscript Figure 4: boxplot of cross-validation accuracies."""
    print("Generating manuscript Figure 4: boxplot...")

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 500
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 12

    accuracies = []
    model_names = ["Baseline", "Layer4 attention", "Avgpool attention"]

    summary_keys = ["cv_summary_baseline", "cv_summary_se", "cv_summary_se_avgpool"]
    for summary_key in summary_keys:
        if summary_key not in results:
            raise KeyError(f"Required summary missing for Figure 4: {summary_key}")
        summary = results[summary_key]
        runs = summary.get("runs", [])
        run_accuracies = [run["val_metrics"]["accuracy"] * 100.0 for run in runs]
        accuracies.append(run_accuracies)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Create boxplot
    bp = ax.boxplot(
        accuracies,
        labels=model_names,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markersize": 8},
        boxprops={"facecolor": "lightblue", "color": "darkblue"},
        whiskerprops={"color": "darkblue"},
        capprops={"color": "darkblue"},
        flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 6},
    )

    # Set colors
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Plot data points
    for i, data in enumerate(accuracies):
        y = data
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.plot(x, y, 'o', markersize=6, alpha=0.6, color=colors[i])

    ax.set_ylabel("Cross-Validation Accuracy (%)", fontsize=12)
    ax.set_xlabel("Model Configuration", fontsize=12)
    ax.set_ylim(97.0, 99.5)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_title("Distribution of matched seed-fold cross-validation accuracies\nacross the 15 development runs", fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Manuscript Figure 4 saved to {output_path}")


def generate_paper_figure5_paired_differences(results, output_path, results_dir):
    """Generate manuscript Figure 5: paired differences plot."""
    print("Generating manuscript Figure 5: paired differences...")

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 500
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11

    baseline_runs = load_runs(results_dir / "cv_summary_baseline.json")
    layer4_runs = load_runs(results_dir / "cv_summary_se.json")
    avgpool_runs = load_runs(results_dir / "cv_summary_se_avgpool.json")

    layer4_diffs = []
    avgpool_diffs = []
    for ref, layer4, avgpool in zip(baseline_runs, layer4_runs, avgpool_runs):
        ref_acc = ref["val_metrics"]["accuracy"]
        layer4_acc = layer4["val_metrics"]["accuracy"]
        avgpool_acc = avgpool["val_metrics"]["accuracy"]
        layer4_diffs.append((layer4_acc - ref_acc) * 100.0)
        avgpool_diffs.append((avgpool_acc - ref_acc) * 100.0)

    layer4_diffs = np.array(layer4_diffs, dtype=float)
    avgpool_diffs = np.array(avgpool_diffs, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True, constrained_layout=True)

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.08, 0.08, size=len(layer4_diffs))

    axes[0].axhline(0.0, color="#666666", linestyle="--", linewidth=1)
    axes[0].scatter(jitter, layer4_diffs, s=42, color="#1f77b4", edgecolor="white", linewidth=0.5, alpha=0.9)
    axes[0].scatter([0], [layer4_diffs.mean()], s=48, color="black", zorder=5)
    axes[0].set_title("Layer4 attention - Baseline", fontsize=11, pad=10)
    axes[0].set_xlim(-0.22, 0.22)
    axes[0].set_xticks([])
    axes[0].set_ylabel("Accuracy difference (pp)")
    axes[0].grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[0].set_ylim(-0.65, 0.85)
    axes[0].text(
        0.02,
        0.04,
        f"Mean: {layer4_diffs.mean():+.2f} pp",
        transform=axes[0].transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    jitter = rng.uniform(-0.08, 0.08, size=len(avgpool_diffs))

    axes[1].axhline(0.0, color="#666666", linestyle="--", linewidth=1)
    axes[1].scatter(jitter, avgpool_diffs, s=42, color="#d62728", edgecolor="white", linewidth=0.5, alpha=0.9)
    axes[1].scatter([0], [avgpool_diffs.mean()], s=48, color="black", zorder=5)
    axes[1].set_title("Avgpool attention - Baseline", fontsize=11, pad=10)
    axes[1].set_xlim(-0.22, 0.22)
    axes[1].set_xticks([])
    axes[1].grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)
    axes[1].text(
        0.02,
        0.04,
        f"Mean: {avgpool_diffs.mean():+.2f} pp",
        transform=axes[1].transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    fig.suptitle("Paired Seed-Fold Accuracy Differences Across 15 Matched Development Runs", fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Manuscript Figure 5 saved to {output_path}")


def generate_paper_figure6_confusion_matrices(results, output_path):
    """Generate manuscript Figure 6 from archived cross-validation summaries."""
    print("Generating manuscript Figure 6: confusion matrices...")

    # Keep the manuscript Figure 6 rendering path aligned with the local
    # workspace helper so the submission repo is a stripped-down counterpart
    # rather than a numerically different implementation.
    sns.set_style("white")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 500
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 13

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))
    cbar_ax = fig.add_axes([0.92, 0.14, 0.015, 0.74])

    for index, (summary_key, panel_title) in enumerate(PAPER_FIGURE6_SPECS):
        if summary_key not in results:
            raise KeyError(f"Required summary missing for manuscript Figure 6: {summary_key}")

        row_normalized, aggregate_accuracy = aggregate_row_normalized_confusion(results[summary_key])
        sns.heatmap(
            row_normalized,
            ax=axes[index],
            cmap="Blues",
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 10},
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            cbar=index == len(PAPER_FIGURE6_SPECS) - 1,
            cbar_ax=cbar_ax if index == len(PAPER_FIGURE6_SPECS) - 1 else None,
            cbar_kws={"label": "Row-normalized percentage (%)"},
            square=True,
        )
        axes[index].set_title(
            f"{panel_title}\nAggregate accuracy: {aggregate_accuracy:.2f}%"
        )
        axes[index].set_xlabel("Predicted")
        axes[index].set_xticklabels(CLASS_LABELS, rotation=45, ha="right")
        axes[index].set_yticklabels(CLASS_LABELS, rotation=0)
        axes[index].tick_params(axis="x", labelsize=10)
        axes[index].tick_params(axis="y", labelsize=10)

    axes[0].set_ylabel("True")
    for axis in axes[1:]:
        axis.set_ylabel("")

    fig.subplots_adjust(left=0.055, right=0.88, bottom=0.21, top=0.84, wspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Manuscript Figure 6 saved to {output_path}")





def main():
    """Main function"""
    args = parse_args()

    print("\n" + "=" * 70)
    print("Manuscript Figure Generation Script")
    print("=" * 70)

    MPL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nFigures will be saved to: {FIGURES_DIR}")
    print(f"Results will be loaded from: {args.results_dir}")
    if args.paper_figure3:
        print(f"Manuscript Figure 3 will also be generated at: {args.paper_figure3_output}")
    if args.paper_figure4:
        print(f"Manuscript Figure 4 will also be generated at: {args.paper_figure4_output}")
    if args.paper_figure5:
        print(f"Manuscript Figure 5 will also be generated at: {args.paper_figure5_output}")
    if args.paper_figure6:
        print(f"Manuscript Figure 6 will also be generated at: {args.paper_output}")
    if not args.paper_figure3 and not args.paper_figure4 and not args.paper_figure5 and not args.paper_figure6:
        print("To generate manuscript Figure 3, 4, 5, and/or 6, run:")
        print("  python -m src.generate_figures --paper-figure3 --paper-figure4 --paper-figure5 --paper-figure6")

    if not args.results_dir.exists():
        print(f"\n[ERROR] Results directory not found: {args.results_dir}")
        print("Please run training first:")
        print("  python -m src.train")
        return

    results = load_results(args.results_dir)

    if not results:
        print(f"\n[ERROR] No result files found in: {args.results_dir}")
        print("Please run training first:")
        print("  python -m src.train")
        return

    print(f"\n[OK] Loaded {len(results)} result files")

    try:
        if args.paper_figure3:
            generate_paper_figure3_training_behavior(results, args.paper_figure3_output)
        if args.paper_figure4:
            generate_paper_figure4_boxplot(results, args.paper_figure4_output)
        if args.paper_figure5:
            generate_paper_figure5_paired_differences(results, args.paper_figure5_output, args.results_dir)
        if args.paper_figure6 or (not args.paper_figure3 and not args.paper_figure4 and not args.paper_figure5):
            generate_paper_figure6_confusion_matrices(results, args.paper_output)

        print("\n" + "=" * 70)
        print("All figures generated successfully!")
        print(f"Output directory: {FIGURES_DIR}")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Figure generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
