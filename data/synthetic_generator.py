"""Synthetic vibration-signal generator.

The SCADA data used in the paper is proprietary, and the NREL GRC / CWRU
benchmark datasets must be downloaded separately (see README). This module
generates *physically motivated* synthetic vibration signals so the full
pipeline can be exercised end-to-end out of the box.

Each fault class injects characteristic spectral signatures:
  - Normal:              gear-mesh harmonics + broadband noise
  - Gearbox Tooth Fault: strong gear-mesh sidebands
  - Inner Race Bearing:  BPIF impulse train + harmonics
  - Outer Race Bearing:  BPOF impulse train + harmonics
  - Rotor Imbalance:     dominant 1x shaft component
"""
from __future__ import annotations

import numpy as np

from configs.config import Config


def _impulse_train(t: np.ndarray, freq: float, decay: float = 800.0) -> np.ndarray:
    """Exponentially decaying periodic impulses at ``freq`` Hz."""
    period = 1.0 / freq
    phase = np.mod(t, period)
    return np.exp(-decay * phase) * np.sin(2 * np.pi * 4000 * t)


def generate_signal(fault_class: int, cfg: Config, duration_s: float,
                     rng: np.random.Generator) -> np.ndarray:
    fs = cfg.signal.fs
    t = np.arange(int(duration_s * fs)) / fs
    ff = cfg.fault_freq

    sig = 0.3 * np.sin(2 * np.pi * ff.shaft_hz * t)           # baseline shaft
    sig += 0.5 * np.sin(2 * np.pi * ff.gear_mesh_hz * t)      # gear mesh
    sig += 0.2 * rng.standard_normal(len(t))                  # broadband noise

    if fault_class == 1:        # Gearbox tooth fault: mesh sidebands
        for sb in (-ff.shaft_hz, ff.shaft_hz):
            sig += 0.6 * np.sin(2 * np.pi * (ff.gear_mesh_hz + sb) * t)
    elif fault_class == 2:      # Inner race bearing
        sig += 1.2 * _impulse_train(t, ff.bpif_hz)
    elif fault_class == 3:      # Outer race bearing
        sig += 1.4 * _impulse_train(t, ff.bpof_hz)
    elif fault_class == 4:      # Rotor imbalance
        sig += 1.5 * np.sin(2 * np.pi * ff.shaft_hz * t)

    return sig.astype(np.float64)


def generate_dataset(cfg: Config, n_per_class: int = 40,
                     duration_s: float = 0.5):
    """Build a synthetic dataset of fused feature sequences.

    Returns ``(X, y)`` with X of shape (n_sequences, T, 38).
    To mimic the paper's ~3:1 normal:fault imbalance, the Normal class is
    generated with 3x the per-class count before SMOTE.
    """
    from src.feature_fusion import FeaturePipeline

    rng = np.random.default_rng(cfg.train.seed)

    # Fit the preprocessor on a pooled normal sample.
    fit_sig = generate_signal(0, cfg, duration_s * 3, rng)
    pipeline = FeaturePipeline(cfg).fit(fit_sig)

    X_list, y_list = [], []
    counts = {0: n_per_class * 3}
    for c in range(1, cfg.model.n_classes):
        counts[c] = n_per_class

    for c, n in counts.items():
        for _ in range(n):
            sig = generate_signal(c, cfg, duration_s, rng)
            power = rng.uniform(0.3, 1.0)
            rpm = rng.uniform(0.4, 1.0)
            fused = pipeline.fuse_signal(
                sig,
                norm_power=np.full(2000, power),
                norm_rpm=np.full(2000, rpm),
            )
            seqs = pipeline.make_sequences(fused)
            # Keep one representative sequence per generated signal.
            X_list.append(seqs[0])
            y_list.append(c)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y
