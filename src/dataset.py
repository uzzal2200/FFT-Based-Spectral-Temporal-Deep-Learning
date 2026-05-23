"""Dataset assembly, splitting, SMOTE balancing, and torch wrappers
(Sections 5.2, 5.3, Algorithm 1).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from configs.config import Config


class SequenceDataset(Dataset):
    """Wraps fused feature sequences and integer labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def stratified_split(
    X: np.ndarray, y: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, ...]:
    """Stratified train / val / test split with a fixed seed."""
    rng = np.random.default_rng(cfg.train.seed)
    train_idx, val_idx, test_idx = [], [], []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(n * cfg.train.test_split)
        n_val = int(n * cfg.train.val_split)
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])

    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)
    test_idx = rng.permutation(test_idx)
    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


def apply_smote(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """SMOTE oversampling on the *training* partition only (Sec. 5.3).

    Sequences are flattened to 2-D for SMOTE, then reshaped back. Falls
    back to a no-op (with a warning) if imbalanced-learn is unavailable.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:  # pragma: no cover
        print("[warn] imbalanced-learn not installed; skipping SMOTE.")
        return X, y

    n, T, d = X.shape
    X_flat = X.reshape(n, T * d)

    # k_neighbors must be < smallest class count.
    min_count = np.bincount(y).min()
    k = max(1, min(5, min_count - 1))
    sm = SMOTE(random_state=seed, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_flat, y)
    return X_res.reshape(-1, T, d), y_res


def make_dataloaders(
    X: np.ndarray, y: np.ndarray, cfg: Config
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test loaders with optional SMOTE on train."""
    Xtr, ytr, Xva, yva, Xte, yte = stratified_split(X, y, cfg)

    if cfg.train.use_smote:
        Xtr, ytr = apply_smote(Xtr, ytr, cfg.train.seed)

    g = torch.Generator().manual_seed(cfg.train.seed)
    train = DataLoader(
        SequenceDataset(Xtr, ytr), batch_size=cfg.train.batch_size,
        shuffle=True, num_workers=cfg.train.num_workers, generator=g,
    )
    val = DataLoader(
        SequenceDataset(Xva, yva), batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.train.num_workers,
    )
    test = DataLoader(
        SequenceDataset(Xte, yte), batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.train.num_workers,
    )
    return train, val, test
