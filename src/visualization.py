"""Plotting utilities reproducing the paper's result figures
(Figures 6-10). All plots are saved at 600 DPI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 600


def _save(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_training_curves(history, out="outputs/figures/training_curves.png"):
    """Figure 6: loss and accuracy convergence."""
    epochs = range(1, len(history.train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history.train_loss, label="Training Loss")
    ax1.plot(epochs, history.val_loss, "--", label="Validation Loss")
    ax1.axvline(history.best_epoch, color="gray", ls=":",
                label=f"Best (epoch {history.best_epoch})")
    ax1.set(title="(a) Loss Curves", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history.train_acc, label="Training Accuracy")
    ax2.plot(epochs, history.val_acc, "--", label="Validation Accuracy")
    ax2.set(title="(b) Accuracy Curves", xlabel="Epoch", ylabel="Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)
    return _save(fig, out)


def plot_confusion_matrix(cm, class_names, out="outputs/figures/confusion_matrix.png"):
    """Figure 8: raw-count and row-normalized confusion matrices."""
    cm = np.asarray(cm)
    cm_norm = cm / (cm.sum(1, keepdims=True) + 1e-12) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, mat, title, fmt in (
        (ax1, cm, "(a) Raw Counts", "d"),
        (ax2, cm_norm, "(b) Row-Normalized (%)", ".1f"),
    ):
        im = ax.imshow(mat, cmap="Reds")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set(title=title, xlabel="Predicted", ylabel="True")
        thresh = mat.max() / 2
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out)


def plot_roc_curves(roc: Dict, class_names, out="outputs/figures/roc_curves.png"):
    """Figure 9: one-vs-rest ROC curves."""
    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = []
    for c, name in enumerate(class_names):
        d = roc[c]
        aucs.append(d["auc"])
        ax.plot(d["fpr"], d["tpr"], label=f"{name} (AUC = {d['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random (AUC = 0.500)")
    ax.set(title="One-vs-Rest ROC Curves", xlabel="False Positive Rate",
           ylabel="True Positive Rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.55, 0.10, f"Macro-Average AUC = {np.nanmean(aucs):.3f}",
            bbox=dict(boxstyle="round", fc="lightblue"))
    return _save(fig, out)


def plot_ablation(configs: List[str], accs: List[float],
                  out="outputs/figures/ablation.png"):
    """Figure 10(a): ablation accuracy bar chart."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(configs, accs, color="steelblue")
    bars[-1].set_color("navy")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.1, f"{a:.1f}%",
                ha="center", fontsize=8)
    ax.set(title="Ablation Study", ylabel="Classification Accuracy (%)",
           ylim=(min(accs) - 3, max(accs) + 2))
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def plot_lead_time(models: Dict[str, np.ndarray],
                   out="outputs/figures/lead_time.png", threshold_days=14):
    """Figure 12(b): lead-time distribution box plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(models.keys())
    data = [models[n] for n in names]
    ax.boxplot(data, labels=names)
    ax.axhline(threshold_days, color="green", ls="--",
               label=f"Maintenance threshold ({threshold_days} d)")
    for i, d in enumerate(data, 1):
        ax.text(i, np.median(d), f"Med: {np.median(d):.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.set(title="Fault Detection Lead Time", ylabel="Lead Time (days)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)
