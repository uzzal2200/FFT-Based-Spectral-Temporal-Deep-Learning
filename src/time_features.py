"""Time-domain feature extraction (Section 4.1.4).

Produces the 15-dimensional time-domain feature vector F_t per frame:

   1   RMS amplitude
   2   Peak amplitude
   3   Peak-to-peak amplitude
   4   Crest factor
   5   Kurtosis
   6   Skewness
   7   Variance
   8   Mean absolute value
   9   Shape factor
   10  Impulse factor
   11  Margin factor
   12  Signal energy entropy
   13  Zero-crossing rate
   14  Normalized power      (turbine operational state)
   15  Normalized RPM        (turbine operational state)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as _kurtosis, skew as _skew

EPS = 1e-12


class TimeFeatureExtractor:
    """15-D time-domain feature extractor."""

    N_FEATURES = 15

    def extract(
        self,
        frame: np.ndarray,
        norm_power: float = 0.0,
        norm_rpm: float = 0.0,
    ) -> np.ndarray:
        """Return the 15-D time-domain feature vector for one frame.

        ``norm_power`` and ``norm_rpm`` are the synchronized, normalized
        SCADA operational-state values for the window (Sec. 4.1.4).
        """
        x = np.asarray(frame, dtype=np.float64)
        abs_x = np.abs(x)

        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(abs_x.max())
        p2p = float(x.max() - x.min())
        crest = peak / (rms + EPS)
        kurt = float(_kurtosis(x, fisher=True, bias=False))
        skewness = float(_skew(x, bias=False))
        variance = float(np.var(x))
        mav = float(abs_x.mean())
        shape = rms / (mav + EPS)
        impulse = peak / (mav + EPS)
        margin = peak / (np.mean(np.sqrt(abs_x)) ** 2 + EPS)

        # Signal energy entropy over a normalized energy distribution.
        energy = x ** 2
        p = energy / (energy.sum() + EPS)
        energy_entropy = float(-(p * np.log2(p + EPS)).sum())

        # Zero-crossing rate.
        zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))

        feats = np.array([
            rms, peak, p2p, crest, kurt, skewness, variance, mav,
            shape, impulse, margin, energy_entropy, zcr,
            float(norm_power), float(norm_rpm),
        ], dtype=np.float64)

        assert feats.shape[0] == self.N_FEATURES, feats.shape
        return feats

    def extract_batch(
        self,
        frames: np.ndarray,
        norm_power: np.ndarray | None = None,
        norm_rpm: np.ndarray | None = None,
    ) -> np.ndarray:
        n = len(frames)
        power = norm_power if norm_power is not None else np.zeros(n)
        rpm = norm_rpm if norm_rpm is not None else np.zeros(n)
        return np.stack(
            [self.extract(fr, p, r) for fr, p, r in zip(frames, power, rpm)],
            axis=0,
        )
