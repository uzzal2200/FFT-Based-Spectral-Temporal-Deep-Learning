"""FFT-based spectral feature extraction (Sections 3.5, 4.1.3, Table 3).

Produces the 23-dimensional frequency-domain feature vector F_f for each
Hann-windowed signal frame:

   1      Spectral energy            (Eq. 12)
   2      Spectral centroid          (Eq. 13)
   3      Spectral spread            (Eq. 14)
   4      Spectral entropy           (Eq. 15)
   5-6    Peak frequency + amplitude (Eq. 17-18)
   7-9    HAR_2, HAR_3, HAR_4        (Eq. 16)
   10-12  Sub-band energies (3 bands)
   13-15  BPOF / BPIF / BSF amplitudes
   16-18  Gear-mesh harmonic amplitudes (1x, 2x, 3x)
   19-20  Fundamental + first sideband (shaft modulation)
   21-23  RMS / kurtosis / crest factor of the PSD
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as _kurtosis

from configs.config import SignalConfig, FaultFrequencyConfig

EPS = 1e-12


class FFTFeatureExtractor:
    """Vectorized 23-D spectral feature extractor."""

    N_FEATURES = 23

    def __init__(self, sig_cfg: SignalConfig, fault_cfg: FaultFrequencyConfig):
        self.sig = sig_cfg
        self.fault = fault_cfg
        self.N = sig_cfg.window_length
        self.fs = sig_cfg.fs
        # One-sided frequency axis (Eq. 4).
        self.freqs = np.fft.rfftfreq(self.N, d=1.0 / self.fs)

    # ------------------------------------------------------------------ #
    def _bin(self, freq_hz: float) -> int:
        """Nearest one-sided FFT bin index for a physical frequency."""
        return int(np.clip(round(freq_hz * self.N / self.fs), 0, len(self.freqs) - 1))

    def _psd(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One-sided magnitude and PSD (Eq. 10-11)."""
        X = np.fft.rfft(frame)
        mag = np.abs(X)
        psd = (2.0 / self.N ** 2) * mag ** 2
        psd[0] = (1.0 / self.N ** 2) * mag[0] ** 2          # DC
        if self.N % 2 == 0:
            psd[-1] = (1.0 / self.N ** 2) * mag[-1] ** 2     # Nyquist
        return mag, psd

    # ------------------------------------------------------------------ #
    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Return the 23-D spectral feature vector for one frame."""
        mag, psd = self._psd(np.asarray(frame, dtype=np.float64))
        f = self.freqs
        psd_sum = psd.sum() + EPS

        # 1. Spectral energy (Eq. 12)
        e_spec = psd.sum()

        # 2. Spectral centroid (Eq. 13)
        f_c = float((f * psd).sum() / psd_sum)

        # 3. Spectral spread (Eq. 14)
        spread = float((((f - f_c) ** 2) * psd).sum() / psd_sum)

        # 4. Spectral entropy (Eq. 15)
        p = psd / psd_sum
        entropy = float(-(p * np.log2(p + EPS)).sum())

        # 5-6. Peak frequency + amplitude (Eq. 17-18), excluding DC
        k_star = int(np.argmax(mag[1:]) + 1)
        f_peak = float(f[k_star])
        a_peak = float(mag[k_star] / self.N)

        # 7-9. Harmonic amplitude ratios HAR_2..HAR_4 (Eq. 16)
        k1 = self._bin(self.fault.shaft_hz)
        a1 = mag[k1] + EPS
        hars = [
            float(mag[self._bin(m * self.fault.shaft_hz)] / a1)
            for m in (2, 3, 4)
        ]

        # 10-12. Sub-band energies
        band_energies = []
        for lo, hi in self.fault.sub_bands_hz:
            mask = (f >= lo) & (f < hi)
            band_energies.append(float(psd[mask].sum()))

        # 13-15. Bearing defect-frequency amplitudes
        bpof = float(mag[self._bin(self.fault.bpof_hz)])
        bpif = float(mag[self._bin(self.fault.bpif_hz)])
        bsf = float(mag[self._bin(self.fault.bsf_hz)])

        # 16-18. Gear-mesh harmonic amplitudes (1x, 2x, 3x)
        gmf = self.fault.gear_mesh_hz
        gear_amps = [float(mag[self._bin(h * gmf)]) for h in (1, 2, 3)]

        # 19-20. Fundamental + first shaft sideband on BPOF
        fundamental = float(mag[self._bin(self.fault.shaft_hz)])
        sideband = float(mag[self._bin(self.fault.bpof_hz + self.fault.shaft_hz)])

        # 21-23. Distribution shape of the PSD (spectral-domain statistics)
        psd_rms = float(np.sqrt(np.mean(psd ** 2)))
        psd_kurt = float(_kurtosis(psd, fisher=True, bias=False))
        psd_crest = float(psd.max() / (psd_rms + EPS))

        feats = np.array([
            e_spec, f_c, spread, entropy,
            f_peak, a_peak,
            *hars,
            *band_energies,
            bpof, bpif, bsf,
            *gear_amps,
            fundamental, sideband,
            psd_rms, psd_kurt, psd_crest,
        ], dtype=np.float64)

        assert feats.shape[0] == self.N_FEATURES, feats.shape
        return feats

    def extract_batch(self, frames: np.ndarray) -> np.ndarray:
        """Extract features for an array of frames ``(n_windows, N)``."""
        return np.stack([self.extract(fr) for fr in frames], axis=0)
