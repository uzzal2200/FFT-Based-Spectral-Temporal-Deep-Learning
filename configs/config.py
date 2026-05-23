"""Central configuration for the FFT-based Spectral-Temporal framework.

All values reflect the hyperparameters reported in the paper
(Tables 4 and 6, Sections 3-5). Edit this single file to reproduce
or sweep experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------- #
# Signal acquisition and preprocessing (Sections 3.1, 4.1.1, 4.1.2)
# --------------------------------------------------------------------------- #
@dataclass
class SignalConfig:
    fs: int = 25_600                 # unified analysis sampling rate (Hz)
    window_length: int = 1_024       # N: FFT analysis window (samples)
    hop_size: int = 256              # H: 75% overlap
    window_type: str = "hann"        # Hann window (Sec. 3.4)

    # Butterworth filtering (Sec. 4.1.2)
    butter_order: int = 8
    lowpass_margin_hz: float = 500.0  # cutoff = fs/2 - margin
    highpass_hz: float = 5.0          # remove DC / mounting resonance
    notch_hz: float = 50.0            # power-line interference
    notch_q: float = 30.0

    @property
    def lowpass_cutoff_hz(self) -> float:
        return self.fs / 2.0 - self.lowpass_margin_hz

    @property
    def freq_resolution_hz(self) -> float:
        return self.fs / self.window_length


# --------------------------------------------------------------------------- #
# Fault-frequency physics (Sec. 4.1.3, 6.1)
# --------------------------------------------------------------------------- #
@dataclass
class FaultFrequencyConfig:
    shaft_hz: float = 30.0           # fundamental shaft frequency
    bpof_hz: float = 287.0           # ball pass outer frequency
    bpif_hz: float = 412.0           # ball pass inner frequency
    bsf_hz: float = 153.0            # ball spin frequency
    gear_mesh_hz: float = 127.3      # NREL GRC Stage-3 planetary mesh
    n_harmonics: int = 4             # HAR_2 .. HAR_4 -> use up to 4x
    sub_bands_hz: List = field(
        default_factory=lambda: [(0.0, 1000.0), (1000.0, 5000.0), (5000.0, 12_800.0)]
    )


# --------------------------------------------------------------------------- #
# Feature dimensions (Sec. 4.1.4)
# --------------------------------------------------------------------------- #
@dataclass
class FeatureConfig:
    n_time_features: int = 15        # F_t
    n_freq_features: int = 23        # F_f
    sequence_length: int = 64        # T

    @property
    def feature_dim(self) -> int:    # d = d_t + d_f
        return self.n_time_features + self.n_freq_features


# --------------------------------------------------------------------------- #
# Model architecture (Sec. 4.1.5, Table 4)
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 4
    dim_feedforward: int = 512
    transformer_dropout: float = 0.1
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    n_classes: int = 5               # 5-class diagnosis; set 2 for binary


# --------------------------------------------------------------------------- #
# Training (Sec. 4.3, Table 6)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    early_stopping_patience: int = 20
    use_smote: bool = True
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 0


# --------------------------------------------------------------------------- #
# Alerting (Sec. 4.1.6, 4.2)
# --------------------------------------------------------------------------- #
@dataclass
class AlertConfig:
    decision_threshold: float = 0.70         # tau
    realtime_consecutive_windows: int = 6    # ~60 ms confirmation
    leadtime_sustained_hours: int = 6        # lead-time confirmation
    advisory_range: tuple = (0.70, 0.85)
    warning_range: tuple = (0.85, 0.95)
    critical_min: float = 0.95


@dataclass
class Config:
    signal: SignalConfig = field(default_factory=SignalConfig)
    fault_freq: FaultFrequencyConfig = field(default_factory=FaultFrequencyConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)

    class_names: List = field(default_factory=lambda: [
        "Normal Operation",
        "Gearbox Tooth Fault",
        "Inner Race Bearing",
        "Outer Race Bearing",
        "Rotor Imbalance",
    ])


def get_config() -> Config:
    """Factory returning the default configuration."""
    return Config()
