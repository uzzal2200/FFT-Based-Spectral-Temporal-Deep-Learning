"""Training pipeline (Section 4.3, Algorithm 1, Table 6).

Implements cross-entropy optimization with the Adam optimizer, gradient-norm
clipping, ReduceLROnPlateau scheduling, early stopping, and best-checkpoint
saving on lowest validation loss.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import Config
from src.model import TransformerLSTM


@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


class Trainer:
    def __init__(self, model: TransformerLSTM, cfg: Config, device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.train.lr,
            betas=cfg.train.adam_betas,
            eps=cfg.train.adam_eps,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min",
            factor=cfg.train.scheduler_factor,
            patience=cfg.train.scheduler_patience,
        )
        self.history = TrainingHistory()
        self._best_state = None

    # ------------------------------------------------------------------ #
    def _run_epoch(self, loader: DataLoader, train: bool):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        torch.set_grad_enabled(train)

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            if train:
                self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.grad_clip_norm
                )
                self.optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

        torch.set_grad_enabled(True)
        return total_loss / total, correct / total

    # ------------------------------------------------------------------ #
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        patience = 0
        for epoch in range(1, self.cfg.train.epochs + 1):
            tr_loss, tr_acc = self._run_epoch(train_loader, train=True)
            va_loss, va_acc = self._run_epoch(val_loader, train=False)
            self.scheduler.step(va_loss)

            self.history.train_loss.append(tr_loss)
            self.history.val_loss.append(va_loss)
            self.history.train_acc.append(tr_acc)
            self.history.val_acc.append(va_acc)
            self.history.lr.append(self.optimizer.param_groups[0]["lr"])

            improved = va_loss < self.history.best_val_loss
            if improved:
                self.history.best_val_loss = va_loss
                self.history.best_epoch = epoch
                self._best_state = copy.deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1

            print(
                f"Epoch {epoch:3d}/{self.cfg.train.epochs} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.2e}"
                + ("  *best*" if improved else "")
            )

            if patience >= self.cfg.train.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(best epoch {self.history.best_epoch}).")
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return self.history

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": self.model.state_dict(),
             "history": self.history.__dict__},
            path,
        )
        print(f"Saved checkpoint -> {path}")

    @torch.no_grad()
    def predict_proba(self, loader: DataLoader):
        self.model.eval()
        probs, labels = [], []
        for X, y in loader:
            X = X.to(self.device)
            p = torch.softmax(self.model(X), dim=1).cpu().numpy()
            probs.append(p)
            labels.append(y.numpy())
        return np.concatenate(probs), np.concatenate(labels)
