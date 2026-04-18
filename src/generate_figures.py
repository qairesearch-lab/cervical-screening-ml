#!/usr/bin/env python3
"""
Figure Generation Script
=======================
This script generates all figures for the paper from experiment results.

Usage:
    python -m src.generate_figures

Required:
    - Run training first to generate results in results/experiment_results/
    - matplotlib and seaborn for visualization
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = Path(".")
RESULTS_DIR = WORKSPACE / "results" / "experiment_results"
FIGURES_DIR = WORKSPACE / "figures"

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


def load_results():
    """Load experiment results"""
    results = {}

    summary_files = [
        "cv_summary_baseline.json",
        "cv_summary_se.json",
        "cv_summary_se_avgpool.json",
        "test_summary.json",
        "statistical_analysis.json"
    ]

    for fname in summary_files:
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            with open(fpath, 'r') as f:
                key = fname.replace(".json", "")
                results[key] = json.load(f)

    return results


def generate_figure1_model_comparison(results):
    """Generate Figure 1: Model Comparison Bar Chart (CV Results)"""
    print("Generating Figure 1: Model Comparison...")

    models = ["Baseline", "+dual-pooling (layer4)", "+dual-pooling (avgpool)"]
    cv_keys = ["cv_summary_baseline", "cv_summary_se", "cv_summary_se_avgpool"]

    accuracy_means = []
    accuracy_stds = []

    for key in cv_keys:
        if key in results:
            acc = results[key]["aggregate_metrics"]["accuracy"]
            accuracy_means.append(acc["mean"] * 100)
            accuracy_stds.append(acc["std"] * 100)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(models))
    bars = ax.bar(x, accuracy_means, yerr=accuracy_stds, capsize=5,
                  color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Figure 1. Cross-Validation Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([min(accuracy_means) - 5, 100])

    for bar, mean, std in zip(bars, accuracy_means, accuracy_stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.5,
                f'{mean:.2f}±{std:.2f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure1_model_comparison.png", bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 1 saved to {FIGURES_DIR / 'figure1_model_comparison.png'}")


def generate_figure2_confusion_matrix(results):
    """Generate Figure 2: Confusion Matrix for Best Model (dual-pooling layer4)"""
    print("Generating Figure 2: Confusion Matrix...")

    if "cv_summary_se" not in results:
        print("[WARNING] dual-pooling results not found, skipping confusion matrix")
        return

    class_names = ["superficial-intermediate", "parabasal", "koilocytes", "dyskeratotic", "metaplastic"]

    avg_cm = np.zeros((5, 5), dtype=float)
    count = 0

    for run in results["cv_summary_se"]["runs"]:
        cm = np.array(run["val_metrics"]["confusion_matrix"])
        avg_cm += cm
        count += 1

    avg_cm = avg_cm / count if count > 0 else avg_cm

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Figure 2. Average Confusion Matrix (+dual-pooling layer4)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure2_confusion_matrix.png", bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 2 saved to {FIGURES_DIR / 'figure2_confusion_matrix.png'}")


def generate_figure3_training_curves(results):
    """Generate Figure 3: Training Curves"""
    print("Generating Figure 3: Training Curves...")

    if "cv_summary_se" not in results or not results["cv_summary_se"]["runs"]:
        print("[WARNING] dual-pooling results not found, skipping training curves")
        return

    first_run = results["cv_summary_se"]["runs"][0]
    history = first_run.get("history", {})

    if not history:
        print("[WARNING] History not found, skipping training curves")
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history["val_loss"], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Figure 3a. Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [x * 100 for x in history["train_acc"]], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, [x * 100 for x in history["val_acc"]], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Figure 3b. Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure3_training_curves.png", bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 3 saved to {FIGURES_DIR / 'figure3_training_curves.png'}")


def generate_figure4_class_performance(results):
    """Generate Figure 4: Class-wise Performance"""
    print("Generating Figure 4: Class-wise Performance...")

    if "cv_summary_se" not in results:
        print("[WARNING] dual-pooling results not found, skipping class performance")
        return

    class_names = ["superficial-intermediate", "parabasal", "koilocytes", "dyskeratotic", "metaplastic"]
    short_names = ["Super-Inter", "Parabasal", "Koilocytes", "Dyskeratotic", "Metaplastic"]

    sensitivities = []
    specificities = []

    for class_name in class_names:
        sens = results["cv_summary_se"]["class_metrics"][class_name]["sensitivity"]
        spec = results["cv_summary_se"]["class_metrics"][class_name]["specificity"]
        sensitivities.append(sens["mean"] * 100)
        specificities.append(spec["mean"] * 100)

    x = np.arange(len(short_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, sensitivities, width, label='Sensitivity', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, specificities, width, label='Specificity', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Cell Class')
    ax.set_ylabel('Score (%)')
    ax.set_title('Figure 4. Class-wise Sensitivity and Specificity (+dual-pooling layer4)')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0, 110])

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure4_class_performance.png", bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 4 saved to {FIGURES_DIR / 'figure4_class_performance.png'}")


def generate_figure5_statistical_analysis(results):
    """Generate Figure 5: Statistical Analysis"""
    print("Generating Figure 5: Statistical Analysis...")

    if "statistical_analysis" not in results:
        print("[WARNING] Statistical analysis not found, skipping")
        return

    analysis = results["statistical_analysis"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    baseline_acc = [x * 100 for x in analysis["baseline_acc"]]
    se_acc = [x * 100 for x in analysis["se_acc"]]
    se_avgpool_acc = [x * 100 for x in analysis["se_avgpool_acc"]]

    seeds = range(1, len(baseline_acc) + 1)

    ax1.bar(seeds, baseline_acc, color='#3498db', alpha=0.8, label='Baseline')
    ax1.bar(seeds, se_acc, bottom=baseline_acc, color='#e74c3c', alpha=0.8, label='+dual-pooling (layer4)')
    ax1.set_xlabel('Seed')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Seed-level Accuracy Comparison')
    ax1.legend()

    comparison_data = {
        'Baseline': baseline_acc,
        '+dual-pooling (layer4)': se_acc,
        '+dual-pooling (avgpool)': se_avgpool_acc
    }
    bp = ax2.boxplot(comparison_data.values(), labels=comparison_data.keys(), patch_artist=True)
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('(b) Accuracy Distribution by Model')

    effect_sizes = [analysis["cohens_d"], analysis["cohens_d_avgpool"]]
    model_names = ['+dual-pooling (layer4)', '+dual-pooling (avgpool)']
    colors = ['#e74c3c', '#2ecc71']
    bars = ax3.bar(model_names, effect_sizes, color=colors, alpha=0.8)
    ax3.axhline(y=0.8, color='gray', linestyle='--', label='Large effect (0.8)')
    ax3.axhline(y=0.5, color='gray', linestyle=':', label='Medium effect (0.5)')
    ax3.set_ylabel("Cohen's d")
    ax3.set_title('(c) Effect Sizes vs Baseline')
    ax3.legend(loc='upper right')

    ax4.axis('off')
    stats_text = f"""
    Statistical Analysis Summary
    ============================

    Baseline vs +dual-pooling (layer4):
      t-statistic: {analysis['t_statistic']:.4f}
      p-value: {analysis['p_value']:.4f}
      Cohen's d: {analysis['cohens_d']:.4f}
      Significant: {'Yes' if analysis['significant'] else 'No'}

    Baseline vs +dual-pooling (avgpool):
      t-statistic: {analysis['t_statistic_avgpool']:.4f}
      p-value: {analysis['p_value_avgpool']:.4f}
      Cohen's d: {analysis['cohens_d_avgpool']:.4f}
      Significant: {'Yes' if analysis['significant_avgpool'] else 'No'}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    ax4.set_title('(d) Statistical Test Results')

    plt.suptitle('Figure 5. Statistical Analysis Results', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure5_statistical_analysis.png", bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 5 saved to {FIGURES_DIR / 'figure5_statistical_analysis.png'}")


def generate_figure6_architecture():
    """Generate Figure 6: Model Architecture Diagram"""
    print("Generating Figure 6: Architecture Diagram...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = ['(a) Baseline ResNet-50', '(b) +dual-pooling (layer4)', '(c) +dual-pooling (avgpool)']
    descriptions = [
        "Input (224x224)\n→ ResNet-50\n→ GAP\n→ FC(2048→512)\n→ FC(512→5)\n→ Output",
        "Input (224x224)\n→ ResNet-50\n→ dual-pooling(layers[1-4])\n→ GAP\n→ FC(2048→512)\n→ FC(512→5)\n→ Output",
        "Input (224x224)\n→ ResNet-50\n→ GAP\n→ dual-pooling(channel)\n→ FC(2048→512)\n→ FC(512→5)\n→ Output"
    ]

    for ax, title, desc in zip(axes, titles, descriptions):
        ax.text(0.5, 0.5, desc, ha='center', va='center', fontsize=10,
                family='monospace', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title(title)
        ax.axis('off')

    plt.suptitle('Figure 6. Model Architecture Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure6_architecture.png", bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 6 saved to {FIGURES_DIR / 'figure6_architecture.png'}")


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("Figure Generation Script")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nFigures will be saved to: {FIGURES_DIR}")

    if not RESULTS_DIR.exists():
        print(f"\n[ERROR] Results directory not found: {RESULTS_DIR}")
        print("Please run training first:")
        print("  python -m src.train")
        return

    results = load_results()

    if not results:
        print(f"\n[ERROR] No result files found in: {RESULTS_DIR}")
        print("Please run training first:")
        print("  python -m src.train")
        return

    print(f"\n[OK] Loaded {len(results)} result files")

    try:
        generate_figure1_model_comparison(results)
        generate_figure2_confusion_matrix(results)
        generate_figure3_training_curves(results)
        generate_figure4_class_performance(results)
        generate_figure5_statistical_analysis(results)
        generate_figure6_architecture()

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