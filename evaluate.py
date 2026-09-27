"""Evaluation script for binary MI vs. non-MI classifier."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import config
from train import predict, save_checkpoint


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray):
    """Find threshold that maximizes F1 on the validation set."""
    thresholds = np.arange(0.05, 0.95, 0.05)
    best_t = 0.5
    best_f1 = 0.0
    for t in thresholds:
        preds = (y_pred >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)


def compute_test_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    """Compute comprehensive binary metrics on the test set."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    metrics = {}
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int(len(y_true) - y_true.sum())
    metrics["threshold"] = float(threshold)

    if len(np.unique(y_true)) < 2:
        metrics["auroc"] = np.nan
        metrics["auprc"] = np.nan
        metrics["f1"] = np.nan
        metrics["sensitivity"] = np.nan
        metrics["specificity"] = np.nan
        metrics["ppv"] = np.nan
        metrics["npv"] = np.nan
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, y_pred))
        metrics["auprc"] = float(average_precision_score(y_true, y_pred))

        binary_preds = (y_pred >= threshold).astype(int)
        metrics["f1"] = float(f1_score(y_true, binary_preds, zero_division=0))

        cm = confusion_matrix(y_true, binary_preds)
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["ppv"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return metrics


def plot_training_curves(history: dict, output_path: Path):
    """Plot train/val loss and validation AUROC over epochs."""
    import matplotlib.pyplot as plt
    history = {key: [row[key] for row in history] for key in history[0]}
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_auroc"], label="Val AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Validation AUROC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    """Plot ROC curve."""
    import matplotlib.pyplot as plt
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    fig, ax = plt.subplots(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"MI (AUC={roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    """Plot precision-recall curve."""
    import matplotlib.pyplot as plt
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    fig, ax = plt.subplots(figsize=(8, 8))
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    ax.plot(recall, precision, label=f"MI (AUPRC={auprc:.3f})", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test set)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float, output_path: Path
):
    """Plot confusion matrix using the chosen threshold."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    binary_preds = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, binary_preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Pred non-MI", "Pred MI"],
        yticklabels=["True non-MI", "True MI"],
    )
    ax.set_title(f"Confusion matrix (thr={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_predictions(ecg_ids, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path):
    """Save a CSV with ecg_id, true label, and predicted probability."""
    df = pd.DataFrame(
        {
            "ecg_id": ecg_ids,
            "true_mi": y_true.ravel().astype(int),
            "pred_proba_mi": y_pred.ravel(),
        }
    )
    df.to_csv(output_path, index=False)


def run_evaluation(model, loaders, checkpoint, output_dir, device):
    """Tune only on validation, persist threshold, then evaluate held-out test."""
    if checkpoint['threshold'] is None:
        y_val, p_val, _, _ = predict(model, loaders['val'], device)
        checkpoint['threshold'] = find_optimal_threshold(y_val, p_val)
        save_checkpoint(checkpoint, output_dir / 'best_model.pt')
    threshold = checkpoint['threshold']
    y, p, ids, _ = predict(model, loaders['test'], device)
    metrics = compute_test_metrics(y, p, threshold)
    metrics['at_0_5'] = compute_test_metrics(y, p, 0.5)
    metrics['smoke_test'] = checkpoint['smoke_test']
    metrics['selected_epoch'] = checkpoint['epoch']
    metrics['confusion_matrix'] = confusion_matrix(y, p >= threshold, labels=[0, 1]).tolist()
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    save_predictions(ids, y, p, output_dir / 'predictions.csv')
    if not checkpoint['smoke_test']:
        plot_roc_curve(y, p, output_dir / 'roc_curve.png')
        plot_pr_curve(y, p, output_dir / 'pr_curve.png')
        plot_confusion_matrix(y, p, threshold, output_dir / 'confusion_matrix.png')
        history_path = output_dir / 'history.json'
        if history_path.exists():
            plot_training_curves(json.loads(history_path.read_text()), output_dir / 'training_curves.png')
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics
