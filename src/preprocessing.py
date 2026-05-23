"""Signal preprocessing pipeline (Section 4.1.2).

Implements the four sequential operations described in the paper:
  1. Eighth-order Butterworth low-pass anti-aliasing filter
  2. High-pass filter (DC / mounting-resonance removal)
  3. Comb / notch filter at the power-line frequency
  4. Z-score standardization (using training statistics)

followed by overlapping segmentation and Hann windowing prior to FFT.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, get_window

from configs.config import SignalConfig


@dataclass
class ChannelStats:
    """Per-channel mean / std fitted on the training partition (Eq. 22)."""
    mean: float
    std: float


class SignalPreprocessor:
    """Stateful preprocessor: ``fit`` on training data, then ``transform``."""

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg
        self._stats: Optional[ChannelStats] = None
        self._window = get_window(cfg.window_type, cfg.window_length, fftbins=True)

        # Pre-design the cascaded anti-aliasing / band-shaping filters.
        nyq = cfg.fs / 2.0
        self._sos_lp = butter(
            cfg.butter_order, cfg.lowpass_cutoff_hz / nyq, btype="low", output="sos"
        )
        self._sos_hp = butter(
            4, cfg.highpass_hz / nyq, btype="high", output="sos"
        )
        b_notch, a_notch = iirnotch(cfg.notch_hz, cfg.notch_q, fs=cfg.fs)
        self._notch_ba = (b_notch, a_notch)

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #
    def _apply_filters(self, x: np.ndarray) -> np.ndarray:
        from scipy.signal import filtfilt

        x = sosfiltfilt(self._sos_lp, x)
        x = sosfiltfilt(self._sos_hp, x)
        b, a = self._notch_ba
        x = filtfilt(b, a, x)
        return x.astype(np.float64)

    # ------------------------------------------------------------------ #
    # Standardization (Eq. 22)
    # ------------------------------------------------------------------ #
    def fit(self, x_train: np.ndarray) -> "SignalPreprocessor":
        filtered = self._apply_filters(np.asarray(x_train, dtype=np.float64))
        self._stats = ChannelStats(
            mean=float(np.mean(filtered)),
            std=float(np.std(filtered) + 1e-12),
        )
        return self

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self._stats is None:
            # Fall back to per-signal standardization if not fitted.
            return (x - x.mean()) / (x.std() + 1e-12)
        return (x - self._stats.mean) / self._stats.std

    # ------------------------------------------------------------------ #
    # Segmentation + windowing
    # ------------------------------------------------------------------ #
    def segment(self, x: np.ndarray) -> np.ndarray:
        """Split a 1-D signal into overlapping Hann-windowed frames.

        Returns an array of shape ``(n_windows, window_length)``.
        """
        N, H = self.cfg.window_length, self.cfg.hop_size
        if len(x) < N:
            x = np.pad(x, (0, N - len(x)))
        n_windows = 1 + (len(x) - N) // H
        idx = np.arange(N)[None, :] + H * np.arange(n_windows)[:, None]
        frames = x[idx]
        return frames * self._window[None, :]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Full pipeline: filter -> standardize -> segment + window."""
        x = np.asarray(x, dtype=np.float64).ravel()
        x = self._apply_filters(x)
        x = self._standardize(x)
        return self.segment(x)

    def preprocess_raw_for_time_features(self, x: np.ndarray) -> np.ndarray:
        """Filtered, standardized, segmented signal WITHOUT the Hann taper.

        Time-domain statistics (Sec. 4.1.4) are computed on the un-tapered
        frames so amplitude-based features are not biased by the window.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        x = self._apply_filters(x)
        x = self._standardize(x)
        N, H = self.cfg.window_length, self.cfg.hop_size
        if len(x) < N:
            x = np.pad(x, (0, N - len(x)))
        n_windows = 1 + (len(x) - N) // H
        idx = np.arange(N)[None, :] + H * np.arange(n_windows)[:, None]
        return x[idx]
