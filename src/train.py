#!/usr/bin/env python3
"""
Unified Cervical Cancer Classification Experiment
==================================================
Main training script for ResNet-50 models with SE (Squeeze-and-Excitation) attention.

This script implements:
    1. ResNet-50 Baseline
    2. ResNet-50 + SE Attention (layer4)
    3. ResNet-50 + SE Attention (avgpool)

Training strategy:
    - Step 1: 3 seeds x 5-fold CV (train/val only, primary results)
    - Step 2: Final model on full data (fixed 15 epochs) + test evaluation
    - Step 3: Report CV (main) + Test (final validation)

Usage:
    python -m src.train

Output:
    - results/cv_summary_*.json
    - results/test_summary.json
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')

# Use relative paths based on project root
WORKSPACE = Path(".")
RESULTS_DIR = WORKSPACE / "results" / "experiment_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "image_size": 224,
    "batch_size": 32,
    "epochs": 15,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "reduction": 16,
    "num_workers": 0,
    "device": "mps"
}

SEEDS = [42, 52, 62]
N_FOLDS = 5
CLASS_NAMES = ["superficial-intermediate", "parabasal", "koilocytes", "dyskeratotic", "metaplastic"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SIPaKMeDDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


def create_model(model_type: str) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    if model_type == "se_layer4":
        model.layer4.se = ChannelAttention(2048, reduction=CONFIG["reduction"])
    elif model_type == "se_avgpool":
        model.avgpool = nn.Sequential(
            model.avgpool,
            ChannelAttention(2048, reduction=CONFIG["reduction"]),
        )

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(CONFIG["dropout"]),
        nn.Linear(512, 5),
    )

    return model


def load_data():
    splits_dir = WORKSPACE / "data" / "splits"

    train_paths = np.load(splits_dir / "train_paths.npy", allow_pickle=True)
    train_labels = np.load(splits_dir / "train_labels.npy", allow_pickle=True)
    val_paths = np.load(splits_dir / "val_paths.npy", allow_pickle=True)
    val_labels = np.load(splits_dir / "val_labels.npy", allow_pickle=True)

    all_paths = np.concatenate([train_paths, val_paths])
    all_labels = np.concatenate([train_labels, val_labels])

    test_paths = np.load(splits_dir / "test_paths.npy", allow_pickle=True)
    test_labels = np.load(splits_dir / "test_labels.npy", allow_pickle=True)
    # Ensure paths are relative to current working directory
    all_paths = [str(WORKSPACE / p) if not Path(p).is_absolute() else str(p) for p in all_paths]
    test_paths = [str(WORKSPACE / p) if not Path(p).is_absolute() else str(p) for p in test_paths]

    return all_paths, all_labels, test_paths, test_labels


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if train_mode:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train_mode:
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / total
    acc = total_correct / total

    return avg_loss, acc, all_labels, all_preds


def evaluate_metrics(labels, preds):
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "confusion_matrix": cm.tolist(),
    }

    metrics["class_metrics"] = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics["class_metrics"][name] = {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    return metrics


def train_fold(model_type, seed, fold_idx, train_idx, val_idx, all_paths, all_labels):
    set_seed(seed)

    fold_name = f"{model_type}_seed{seed}_fold{fold_idx}"
    fold_dir = RESULTS_DIR / "cv" / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    result_file = fold_dir / "metrics.json"
    if result_file.exists():
        print(f"  [Skip] {fold_name} already exists")
        return json.loads(result_file.read_text())

    print(f"\n  [CV] Training: {fold_name}")

    train_transform, eval_transform = get_transforms()

    train_dataset = SIPaKMeDDataset([all_paths[i] for i in train_idx], [all_labels[i] for i in train_idx], train_transform)
    val_dataset = SIPaKMeDDataset([all_paths[i] for i in val_idx], [all_labels[i] for i in val_idx], eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if DEVICE.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )

    model = create_model(model_type).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:2d}/{CONFIG['epochs']} | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                  f"Best: {best_val_acc:.4f}")

    model.load_state_dict(best_state)
    _, _, val_labels, val_preds = run_epoch(model, val_loader, criterion)
    val_metrics = evaluate_metrics(val_labels, val_preds)

    duration = time.time() - start_time

    result = {
        "model_type": model_type,
        "seed": seed,
        "fold": fold_idx,
        "config": CONFIG,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "history": history,
        "val_metrics": val_metrics,
        "duration_seconds": duration,
    }

    result_file.write_text(json.dumps(result, indent=2))
    torch.save(best_state, fold_dir / "best_model.pth")

    print(f"    [OK] Completed: Val Acc = {best_val_acc:.4f}, Time = {duration/60:.1f}min")

    return result


def run_cross_validation(model_type, seed, all_paths, all_labels):
    set_seed(seed)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):
        result = train_fold(
            model_type, seed, fold_idx,
            train_idx, val_idx,
            all_paths, all_labels
        )
        fold_results.append(result)

    return fold_results


def train_final_model(model_type, seed, all_paths, all_labels, test_paths, test_labels):
    set_seed(seed)

    final_name = f"{model_type}_seed{seed}_final"
    final_dir = RESULTS_DIR / "final" / final_name
    final_dir.mkdir(parents=True, exist_ok=True)

    result_file = final_dir / "metrics.json"
    if result_file.exists():
        print(f"  [Skip] {final_name} already exists")
        return json.loads(result_file.read_text())

    print(f"\n[Final Model] {model_type} | Seed {seed}")

    train_transform, eval_transform = get_transforms()

    train_dataset = SIPaKMeDDataset(all_paths, all_labels, train_transform)
    test_dataset = SIPaKMeDDataset(test_paths, test_labels, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    model = create_model(model_type).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    for epoch in range(CONFIG["epochs"]):
        train_loss, _, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        if epoch % 5 == 0 or epoch == CONFIG["epochs"] - 1:
            print(f"    Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss:.4f}")

    final_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(final_state)
    _, test_acc, labels, preds = run_epoch(model, test_loader, criterion)
    test_metrics = evaluate_metrics(labels, preds)

    result = {
        "model_type": model_type,
        "seed": seed,
        "test_acc": test_acc,
        "test_metrics": test_metrics
    }

    result_file.write_text(json.dumps(result, indent=2))
    torch.save(final_state, final_dir / "final_model.pth")

    print(f"    [OK] Test Acc = {test_acc:.4f}")

    return result


def summarize_cv_results(all_cv_results, model_type):
    model_results = [r for r in all_cv_results if r["model_type"] == model_type]

    metrics = {}
    for key in ["accuracy", "precision", "recall", "f1"]:
        vals = [r["val_metrics"][key] for r in model_results]
        metrics[key] = {
            "values": vals,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    class_summary = {}
    for class_name in CLASS_NAMES:
        for metric_name in ["sensitivity", "specificity"]:
            vals = [r["val_metrics"]["class_metrics"][class_name][metric_name]
                    for r in model_results]
            class_summary.setdefault(class_name, {})[metric_name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
            }

    return {
        "model_type": model_type,
        "total_runs": len(model_results),
        "aggregate_metrics": metrics,
        "class_metrics": class_summary,
        "runs": model_results,
    }


def summarize_test_results(final_results, model_type):
    model_results = [r for r in final_results if r["model_type"] == model_type]
    vals = [r["test_acc"] for r in model_results]

    return {
        "values": vals,
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)),
    }


def get_seed_level_means(cv_results, model_type):
    seed_dict = {}
    for r in cv_results:
        if r["model_type"] != model_type:
            continue
        seed = r["seed"]
        seed_dict.setdefault(seed, []).append(r["val_metrics"]["accuracy"])

    return [np.mean(v) for v in seed_dict.values()]


def write_report(cv_summary_baseline, cv_summary_se, cv_summary_se_avgpool, test_baseline, test_se, test_se_avgpool):
    def fmt(mean, std):
        return f"{mean*100:.2f}% ± {std*100:.2f}%"

    md = []
    md.append("# Unified Experiment Report")
    md.append("")
    md.append("## Configuration")
    md.append("")
    md.append(f"- Device: `{DEVICE}`")
    md.append(f"- Seeds: `{SEEDS}`")
    md.append(f"- CV Folds: `{N_FOLDS}`")
    md.append(f"- Total CV runs per model: `{len(SEEDS) * N_FOLDS}`")
    md.append(f"- Final test evaluations per model: `{len(SEEDS)}`")
    md.append("")
    md.append("### Training Parameters")
    md.append("")
    md.append(f"- Epochs: `{CONFIG['epochs']}`")
    md.append(f"- Batch size: `{CONFIG['batch_size']}`")
    md.append(f"- Learning rate: `{CONFIG['learning_rate']}`")
    md.append(f"- Optimizer: `AdamW`")
    md.append(f"- Scheduler: `CosineAnnealingLR`")
    md.append("")

    md.append("## Main Results: Cross-Validation (15 runs per model)")
    md.append("")
    md.append("| Model | Accuracy | Precision | Recall | F1-Score |")
    md.append("|-------|----------|-----------|--------|----------|")

    b_cv = cv_summary_baseline["aggregate_metrics"]
    s_cv = cv_summary_se["aggregate_metrics"]
    s_avgpool_cv = cv_summary_se_avgpool["aggregate_metrics"]

    md.append(f"| Baseline | {fmt(b_cv['accuracy']['mean'], b_cv['accuracy']['std'])} | "
              f"{fmt(b_cv['precision']['mean'], b_cv['precision']['std'])} | "
              f"{fmt(b_cv['recall']['mean'], b_cv['recall']['std'])} | "
              f"{fmt(b_cv['f1']['mean'], b_cv['f1']['std'])} |")

    md.append(f"| +SE (layer4) | {fmt(s_cv['accuracy']['mean'], s_cv['accuracy']['std'])} | "
              f"{fmt(s_cv['precision']['mean'], s_cv['precision']['std'])} | "
              f"{fmt(s_cv['recall']['mean'], s_cv['recall']['std'])} | "
              f"{fmt(s_cv['f1']['mean'], s_cv['f1']['std'])} |")

    md.append(f"| +SE (avgpool) | {fmt(s_avgpool_cv['accuracy']['mean'], s_avgpool_cv['accuracy']['std'])} | "
              f"{fmt(s_avgpool_cv['precision']['mean'], s_avgpool_cv['precision']['std'])} | "
              f"{fmt(s_avgpool_cv['recall']['mean'], s_avgpool_cv['recall']['std'])} | "
              f"{fmt(s_avgpool_cv['f1']['mean'], s_avgpool_cv['f1']['std'])} |")

    acc_diff_cv = s_cv['accuracy']['mean'] - b_cv['accuracy']['mean']
    acc_diff_cv_avgpool = s_avgpool_cv['accuracy']['mean'] - b_cv['accuracy']['mean']
    md.append("")
    md.append(f"**CV Accuracy Improvement (+SE layer4):** {acc_diff_cv*100:+.2f}%")
    md.append(f"**CV Accuracy Improvement (+SE avgpool):** {acc_diff_cv_avgpool*100:+.2f}%")
    md.append("")

    md.append("## Final Test Evaluation (3 runs per model)")
    md.append("")
    md.append("| Model | Test Accuracy |")
    md.append("|-------|---------------|")
    md.append(f"| Baseline | {fmt(test_baseline['mean'], test_baseline['std'])} |")
    md.append(f"| +SE (layer4) | {fmt(test_se['mean'], test_se['std'])} |")
    md.append(f"| +SE (avgpool) | {fmt(test_se_avgpool['mean'], test_se_avgpool['std'])} |")

    acc_diff_test = test_se['mean'] - test_baseline['mean']
    acc_diff_test_avgpool = test_se_avgpool['mean'] - test_baseline['mean']
    md.append("")
    md.append(f"**Test Accuracy Improvement (+SE layer4):** {acc_diff_test*100:+.2f}%")
    md.append(f"**Test Accuracy Improvement (+SE avgpool):** {acc_diff_test_avgpool*100:+.2f}%")
    md.append("")

    md.append("## Class-wise Metrics (from CV)")
    md.append("")
    md.append("| Class | Sensitivity | Specificity |")
    md.append("|-------|-------------|-------------|")

    for class_name in CLASS_NAMES:
        sens = cv_summary_se["class_metrics"][class_name]["sensitivity"]
        spec = cv_summary_se["class_metrics"][class_name]["specificity"]
        md.append(f"| {class_name} | {fmt(sens['mean'], sens['std'])} | {fmt(spec['mean'], spec['std'])} |")

    md.append("")

    report_path = RESULTS_DIR / "REPORT.md"
    report_path.write_text("\n".join(md))
    print(f"\n[OK] Report saved to: {report_path}")

    return "\n".join(md)


def statistical_analysis(cv_results, cv_summary_baseline, cv_summary_se, cv_summary_se_avgpool):
    from scipy import stats

    print("\n" + "="*70)
    print("Statistical Analysis (CV Results - Seed Level)")
    print("="*70)
    print("Note: Statistical tests were conducted at the seed level to avoid dependency between cross-validation folds.")

    baseline_acc = get_seed_level_means(cv_results, "baseline")
    se_acc = get_seed_level_means(cv_results, "se_layer4")
    se_avgpool_acc = get_seed_level_means(cv_results, "se_avgpool")

    print("\n--- Baseline vs +SE (layer4) ---")
    t_stat, p_value = stats.ttest_rel(baseline_acc, se_acc)

    print(f"\nPaired t-test (n={len(baseline_acc)}):")
    print(f"  Baseline: {np.mean(baseline_acc)*100:.2f}% ± {np.std(baseline_acc, ddof=1)*100:.2f}%")
    print(f"  SE (layer4): {np.mean(se_acc)*100:.2f}% ± {np.std(se_acc, ddof=1)*100:.2f}%")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.001:
        print(f"  *** Significant at p < 0.001")
    elif p_value < 0.01:
        print(f"  ** Significant at p < 0.01")
    elif p_value < 0.05:
        print(f"  * Significant at p < 0.05")
    else:
        print(f"  Not significant (p >= 0.05)")

    print("\n--- Baseline vs +SE (avgpool) ---")
    t_stat_avgpool, p_value_avgpool = stats.ttest_rel(baseline_acc, se_avgpool_acc)

    print(f"\nPaired t-test (n={len(baseline_acc)}):")
    print(f"  Baseline: {np.mean(baseline_acc)*100:.2f}% ± {np.std(baseline_acc, ddof=1)*100:.2f}%")
    print(f"  SE (avgpool): {np.mean(se_avgpool_acc)*100:.2f}% ± {np.std(se_avgpool_acc, ddof=1)*100:.2f}%")
    print(f"  t-statistic: {t_stat_avgpool:.4f}")
    print(f"  p-value: {p_value_avgpool:.4f}")

    if p_value_avgpool < 0.001:
        print(f"  *** Significant at p < 0.001")
    elif p_value_avgpool < 0.01:
        print(f"  ** Significant at p < 0.01")
    elif p_value_avgpool < 0.05:
        print(f"  * Significant at p < 0.05")
    else:
        print(f"  Not significant (p >= 0.05)")

    mean_diff = np.mean(se_acc) - np.mean(baseline_acc)
    pooled_std = np.sqrt((np.var(baseline_acc, ddof=1) + np.var(se_acc, ddof=1)) / 2)
    cohens_d = mean_diff / pooled_std

    print(f"\n--- Effect Sizes ---")
    print(f"\nEffect size (Cohen's d) - SE (layer4): {cohens_d:.4f}")

    mean_diff_avgpool = np.mean(se_avgpool_acc) - np.mean(baseline_acc)
    pooled_std_avgpool = np.sqrt((np.var(baseline_acc, ddof=1) + np.var(se_avgpool_acc, ddof=1)) / 2)
    cohens_d_avgpool = mean_diff_avgpool / pooled_std_avgpool

    print(f"Effect size (Cohen's d) - SE (avgpool): {cohens_d_avgpool:.4f}")

    for name, d in [("SE (layer4)", cohens_d), ("SE (avgpool)", cohens_d_avgpool)]:
        if abs(d) < 0.2:
            print(f"  {name}: Negligible effect")
        elif abs(d) < 0.5:
            print(f"  {name}: Small effect")
        elif abs(d) < 0.8:
            print(f"  {name}: Medium effect")
        else:
            print(f"  {name}: Large effect")

    analysis = {
        "baseline_acc": baseline_acc,
        "se_acc": se_acc,
        "se_avgpool_acc": se_avgpool_acc,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "t_statistic_avgpool": float(t_stat_avgpool),
        "p_value_avgpool": float(p_value_avgpool),
        "cohens_d": float(cohens_d),
        "cohens_d_avgpool": float(cohens_d_avgpool),
        "significant": p_value < 0.05,
        "significant_avgpool": p_value_avgpool < 0.05,
    }

    analysis_path = RESULTS_DIR / "statistical_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2))

    return analysis


def main():
    print("="*70)
    print("Unified Cervical Cancer Classification Experiment")
    print("Optimized Experiment Flow:")
    print("  Step 1: 3 seeds x 5-fold CV (train/val only)")
    print("  Step 2: Final model on full data (fixed 15 epochs) + test evaluation")
    print("  Step 3: Report CV (main) + Test (final validation)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Seeds: {SEEDS}")
    print(f"  CV Folds: {N_FOLDS}")
    print(f"  Epochs: {CONFIG['epochs']}")

    print("\nLoading data...")
    all_paths, all_labels, test_paths, test_labels = load_data()
    print(f"  Training/Val: {len(all_paths)} images")
    print(f"  Test: {len(test_paths)} images (held out)")

    print("\n" + "="*70)
    print("STEP 1: Cross-Validation (3 seeds x 5 folds)")
    print("="*70)

    cv_results = []

    for model_type in ["baseline", "se_layer4", "se_avgpool"]:
        print(f"\n{'='*70}")
        print(f"Model: {model_type}")
        print(f"{'='*70}")

        for seed in SEEDS:
            print(f"\nSeed {seed}:")
            fold_results = run_cross_validation(model_type, seed, all_paths, all_labels)
            cv_results.extend(fold_results)

    print("\n" + "="*70)
    print("Summarizing CV Results")
    print("="*70)

    cv_summary_baseline = summarize_cv_results(cv_results, "baseline")
    cv_summary_se = summarize_cv_results(cv_results, "se_layer4")
    cv_summary_se_avgpool = summarize_cv_results(cv_results, "se_avgpool")

    with open(RESULTS_DIR / "cv_summary_baseline.json", "w") as f:
        json.dump(cv_summary_baseline, f, indent=2)
    with open(RESULTS_DIR / "cv_summary_se.json", "w") as f:
        json.dump(cv_summary_se, f, indent=2)
    with open(RESULTS_DIR / "cv_summary_se_avgpool.json", "w") as f:
        json.dump(cv_summary_se_avgpool, f, indent=2)

    print("\n" + "="*70)
    print("STEP 2: Final Test Evaluation (Held-out Test Set)")
    print("="*70)

    final_results = []

    for model_type in ["baseline", "se_layer4", "se_avgpool"]:
        print(f"\nModel: {model_type}")
        for seed in SEEDS:
            res = train_final_model(
                model_type, seed,
                all_paths, all_labels,
                test_paths, test_labels
            )
            final_results.append(res)

    test_baseline = summarize_test_results(final_results, "baseline")
    test_se = summarize_test_results(final_results, "se_layer4")
    test_se_avgpool = summarize_test_results(final_results, "se_avgpool")

    with open(RESULTS_DIR / "test_summary.json", "w") as f:
        json.dump({"baseline": test_baseline, "se": test_se, "se_avgpool": test_se_avgpool}, f, indent=2)

    print("\n" + "="*70)
    print("STEP 3: Reporting and Statistical Analysis")
    print("="*70)

    write_report(cv_summary_baseline, cv_summary_se, cv_summary_se_avgpool, test_baseline, test_se, test_se_avgpool)
    statistical_analysis(cv_results, cv_summary_baseline, cv_summary_se, cv_summary_se_avgpool)

    print("\n" + "="*70)
    print("Experiment Complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()