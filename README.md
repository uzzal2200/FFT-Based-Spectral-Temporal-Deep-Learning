# FFT-Based Spectral–Temporal Deep Learning Framework for Wind Turbine Fault Diagnosis

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-Academic-orange" alt="License"></a>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/requirements-pinned-lightgrey" alt="Requirements"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/paper-Citation-blueviolet" alt="Paper"></a>
  <img src="https://img.shields.io/badge/research-FFT%20Spectral--Temporal-brightgreen" alt="Research">
  <img src="https://img.shields.io/badge/GPU-Optional-yellow" alt="GPU optional">
  <img src="https://img.shields.io/badge/docker-ready-lightgrey" alt="Docker ready">
  <img src="https://img.shields.io/badge/tests-none-lightgrey" alt="Tests">
</p>

A modular, reproducible PyTorch implementation of an **FFT-enhanced spectral–temporal deep learning framework** for wind turbine fault diagnosis and predictive maintenance. The framework fuses physically interpretable frequency-domain features (extracted via the Fast Fourier Transform) with time-domain statistics, and classifies the resulting dual-domain representation using a hybrid **Transformer–LSTM** architecture.

This repository accompanies the paper _"FFT-Based Spectral–Temporal Deep Learning Framework for Wind Turbine Fault Diagnosis and Predictive Maintenance"_ and reproduces its complete methodology: dual-domain feature engineering, the Transformer–LSTM model, the training pipeline, and the evaluation / alerting logic.

---

## Key Features

- **Dual-domain feature engineering** — a 23-dimensional spectral feature vector (spectral energy, centroid, spread, entropy, harmonic amplitude ratios, bearing/gearbox defect-frequency amplitudes, and PSD shape statistics) fused with a 15-dimensional time-domain vector into a unified **38-dimensional** representation.
- **Hybrid Transformer–LSTM model** — sinusoidal positional encoding, a 4-layer / 8-head Transformer encoder for global spectral dependencies, followed by a 2-layer LSTM for temporal fault-progression modeling.
- **Faithful training pipeline** — Adam optimization, gradient-norm clipping, `ReduceLROnPlateau` scheduling, early stopping, Xavier initialization, and SMOTE class balancing (applied to the training partition only).
- **Comprehensive evaluation** — per-class precision/recall/F1, macro-averaged metrics, one-vs-rest ROC/AUC, confusion matrices, and an ablation harness.
- **Predictive-maintenance alerting** — three-tier (Advisory / Warning / Critical) thresholding with consecutive-window confirmation, plus the lead-time analysis logic.
- **Runnable out of the box** — a physically motivated synthetic vibration generator lets you exercise the full pipeline without the proprietary SCADA / benchmark datasets.

---

## Project Structure

```
.
├── configs/
│   └── config.py              # All hyperparameters (Tables 4 & 6, Sec. 3–5)
├── src/
│   ├── preprocessing.py       # Butterworth filtering, normalization, Hann windowing (Sec. 4.1.2)
│   ├── fft_features.py        # 23-D spectral feature extraction (Sec. 3.5, Table 3)
│   ├── time_features.py       # 15-D time-domain feature extraction (Sec. 4.1.4)
│   ├── feature_fusion.py      # Dual-domain fusion + sequence assembly (Eq. 23)
│   ├── model.py               # Transformer–LSTM architecture (Eq. 24–39)
│   ├── dataset.py             # Splitting, SMOTE, torch DataLoaders
│   ├── train.py               # Training loop (Algorithm 1, Sec. 4.3)
│   ├── evaluate.py            # Metrics, ROC, alerting, lead-time (Sec. 4.1.6, 4.2, 6.x)
│   └── visualization.py       # 600-DPI result figures (Fig. 6–10, 12)
├── data/
│   └── synthetic_generator.py # Synthetic vibration signals for demos
├── figures/                   # Result figures and plots
├── scripts/
│   ├── run_training.py        # End-to-end training entry point
│   └── run_inference.py       # Streaming inference + alerting demo
├── requirements.txt
└── README.md
```

---

## Installation

Requires **Python 3.9+**.

```bash
# (recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

> A CUDA-enabled GPU is optional. The code automatically uses `cuda` when available and falls back to CPU otherwise.

---

## Quick Start

Run all commands from the project root so the package imports resolve.

### Train (synthetic demo data)

```bash
python -m scripts.run_training --epochs 50 --n-per-class 40
```

This generates a synthetic dataset, trains the Transformer–LSTM model, prints a full classification report on the held-out test set, and writes result figures to `outputs/figures/`. The best checkpoint (lowest validation loss) is saved to `outputs/checkpoints/best_model.pt`.

### Streaming inference + alerting

```bash
python -m scripts.run_inference --checkpoint outputs/checkpoints/best_model.pt --fault-class 3
```

Streams a synthetic faulty signal through the trained model window-by-window and triggers a tiered fault alert once the configured confirmation criterion is met.

---

## Configuration

All hyperparameters live in a single dataclass-based file, `configs/config.py`, mirroring the paper:

| Group    | Key settings                                                                         |
| -------- | ------------------------------------------------------------------------------------ |
| Signal   | `fs = 25,600 Hz`, `N = 1024`, `H = 256` (75% overlap), Hann window                   |
| Features | 15 time-domain + 23 frequency-domain = **38-D**, sequence length `T = 64`            |
| Model    | `d_model = 128`, 8 heads, 4 encoder layers, `d_ff = 512`, 2 LSTM layers (hidden 128) |
| Training | Adam (`lr = 1e-3`), batch 32, 100 epochs, grad-clip 1.0, ReduceLROnPlateau, SMOTE    |
| Alerting | `τ = 0.70`, 6-window real-time confirmation, 6-hour lead-time confirmation           |

Edit this file to reproduce or sweep experiments — no other module hard-codes these values.

---

## Using Real Data

The synthetic generator is a stand-in for development. To run on real measurements:

1. **NREL Gearbox Reliability Collaborative (GRC)** — available at <https://doi.org/10.2172/1048981>.
2. **CWRU Bearing Data Center** — the 48 kHz drive-end recordings, decimated to 25.6 kHz.
3. **SCADA operational data** — your own 1 Hz / 10-minute-average channels.

Replace the call to `generate_dataset(...)` in `scripts/run_training.py` with a loader that yields raw 1-D vibration signals (plus optional normalized power/RPM), then pass them through `FeaturePipeline.fuse_signal(...)` and `FeaturePipeline.make_sequences(...)`. The rest of the pipeline is dataset-agnostic. Apply the harmonization steps from Section 5.2 (resampling to 25.6 kHz, label mapping to the five-class taxonomy, global z-score normalization).

---

## Architecture Summary

| Layer                   | Configuration           | Output      |
| ----------------------- | ----------------------- | ----------- |
| Input feature sequence  | `T = 64`, `d = 38`      | `(64, 38)`  |
| Linear projection       | `38 → 128`              | `(64, 128)` |
| Positional encoding     | Sinusoidal              | `(64, 128)` |
| Transformer encoder × 4 | 8 heads, `d_ff = 512`   | `(64, 128)` |
| LSTM × 2                | hidden 128, dropout 0.2 | `(128,)`    |
| FC + Softmax            | `128 → C`               | `(C,)`      |

Binary detection uses `C = 2`; five-class diagnosis uses `C = 5` (Normal, Gearbox Tooth Fault, Inner Race Bearing, Outer Race Bearing, Rotor Imbalance).

---

## Reproducibility Notes

- A global seed (`config.train.seed = 42`) is applied to NumPy, PyTorch, and the data splitter.
- SMOTE is applied to the **training partition only**; validation and test sets retain the original class distribution so reported metrics reflect the real imbalance.
- The model in `src/model.py` is the _full_ implementation: it includes the linear projection, sinusoidal positional encoding, dropout, and the explicit `dim_feedforward = 512` that the paper's simplified code listing omits for brevity.
- The accuracy figures quoted in the paper (95.8% binary, 96.1% five-class) require the real benchmark datasets and full training schedule; the synthetic generator is intended for pipeline validation, not for reproducing those headline numbers.

---

## License

Released for academic and research use. Please cite the accompanying paper if you use this code in your work.

## Citation

```bibtex
@article{debnath2026fft,
  title   = {FFT-Based Spectral--Temporal Deep Learning Framework for Wind
             Turbine Fault Diagnosis and Predictive Maintenance},
  author  = {Debnath, Sajib and Mia, Md. Uzzal and Abubakkar, Md and
             Biswas, Arindam Kishor},
  journal = {Journal of Computational and Cognitive Engineering},
  year    = {2026}
}
```
