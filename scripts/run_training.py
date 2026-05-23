"""End-to-end training entry point.

Usage:
    python -m scripts.run_training            # synthetic demo data
    python -m scripts.run_training --epochs 50

Builds the feature pipeline, trains the Transformer-LSTM model, evaluates on
the held-out test set, and writes result figures to ``outputs/figures``.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from configs.config import get_config
from src.dataset import make_dataloaders
from src.model import build_model
from src.train import Trainer
from src.evaluate import classification_report, print_report, roc_curves
from src import visualization as viz
from data.synthetic_generator import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Spectral-Temporal model.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-per-class", type=int, default=40)
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    args = parser.parse_args()

    cfg = get_config()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.no_smote:
        cfg.train.use_smote = False

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    print("=" * 60)
    print("Generating synthetic dataset (replace with real data loaders)")
    print("=" * 60)
    X, y = generate_dataset(cfg, n_per_class=args.n_per_class)
    print(f"Dataset: X={X.shape}  classes={np.bincount(y)}")

    train_loader, val_loader, test_loader = make_dataloaders(X, y, cfg)

    model = build_model(cfg)
    print(f"Model parameters: {model.count_parameters():,}")

    trainer = Trainer(model, cfg)
    print("\nTraining...")
    history = trainer.fit(train_loader, val_loader)
    trainer.save(args.checkpoint)

    print("\nEvaluating on held-out test set...")
    proba, labels = trainer.predict_proba(test_loader)
    report = classification_report(labels, proba, cfg.class_names)
    print_report(report, cfg.class_names)

    print("\nWriting figures...")
    viz.plot_training_curves(history)
    viz.plot_confusion_matrix(report["confusion_matrix"], cfg.class_names)
    viz.plot_roc_curves(
        roc_curves(labels, proba, cfg.model.n_classes), cfg.class_names
    )
    # Ablation accuracies as reported in Table 9 (illustrative reference values).
    viz.plot_ablation(
        ["LSTM only", "FFT+MLP", "Transformer only",
         "Time+FFT->LSTM", "Transformer+FFT", "Proposed"],
        [88.2, 89.1, 90.5, 91.5, 93.0, 95.8],
    )
    print("Done. See outputs/figures/")


if __name__ == "__main__":
    main()
