"""Dual-domain feature fusion and sequence assembly (Sections 4.1.4, 3.6).

Concatenates the 15-D time-domain and 23-D frequency-domain feature
vectors into the 38-D fused representation (Eq. 23) and assembles
sequences of T = 64 consecutive vectors (the model input tensor).
"""
from __future__ import annotations

import numpy as np

from configs.config import Config
from src.preprocessing import SignalPreprocessor
from src.fft_features import FFTFeatureExtractor
from src.time_features import TimeFeatureExtractor


class FeaturePipeline:
    """End-to-end raw-signal -> fused-feature-sequence pipeline."""

    def __init__(self, cfg: Config, preprocessor: SignalPreprocessor | None = None):
        self.cfg = cfg
        self.pre = preprocessor or SignalPreprocessor(cfg.signal)
        self.fft = FFTFeatureExtractor(cfg.signal, cfg.fault_freq)
        self.tfe = TimeFeatureExtractor()

    def fit(self, x_train: np.ndarray) -> "FeaturePipeline":
        self.pre.fit(x_train)
        return self

    # ------------------------------------------------------------------ #
    def fuse_signal(
        self,
        x: np.ndarray,
        norm_power: np.ndarray | None = None,
        norm_rpm: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return fused features ``(n_windows, 38)`` for a 1-D signal."""
        windowed = self.pre.transform(x)                       # Hann-tapered
        raw_frames = self.pre.preprocess_raw_for_time_features(x)  # un-tapered

        f_freq = self.fft.extract_batch(windowed)
        f_time = self.tfe.extract_batch(raw_frames, norm_power, norm_rpm)

        fused = np.concatenate([f_time, f_freq], axis=1)       # Eq. 23
        assert fused.shape[1] == self.cfg.features.feature_dim
        return fused

    # ------------------------------------------------------------------ #
    def make_sequences(self, fused: np.ndarray) -> np.ndarray:
        """Slide a window of length T over fused vectors.

        Returns ``(n_sequences, T, 38)``.
        """
        T = self.cfg.features.sequence_length
        if len(fused) < T:
            pad = np.repeat(fused[-1:], T - len(fused), axis=0)
            fused = np.concatenate([fused, pad], axis=0)
        n_seq = len(fused) - T + 1
        idx = np.arange(T)[None, :] + np.arange(n_seq)[:, None]
        return fused[idx]
