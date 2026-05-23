"""Inference + alerting demonstration.

Usage:
    python -m scripts.run_inference --checkpoint outputs/checkpoints/best_model.pt

Loads a trained model, runs streaming inference over a synthetic faulty
signal, and demonstrates the three-tier alerting logic (Sec. 4.1.6).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from configs.config import get_config
from src.model import build_model
from src.feature_fusion import FeaturePipeline
from src.evaluate import FaultAlerter
from data.synthetic_generator import generate_signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Run streaming inference + alerting.")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--fault-class", type=int, default=3)
    args = parser.parse_args()

    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rng = np.random.default_rng(0)
    pipeline = FeaturePipeline(cfg)
    fit_sig = generate_signal(0, cfg, 1.0, rng)
    pipeline.fit(fit_sig)

    sig = generate_signal(args.fault_class, cfg, 1.5, rng)
    fused = pipeline.fuse_signal(sig)
    sequences = pipeline.make_sequences(fused)

    alerter = FaultAlerter(cfg)
    print(f"Streaming {len(sequences)} windows "
          f"(true class = {cfg.class_names[args.fault_class]})\n")

    with torch.no_grad():
        for i, seq in enumerate(sequences):
            x = torch.as_tensor(seq[None], dtype=torch.float32, device=device)
            proba = torch.softmax(model(x), dim=1).cpu().numpy()[0]
            alert = alerter.update(proba)
            if alert.triggered:
                print(f"  window {i:3d}: ALERT [{alert.severity}] "
                      f"type={cfg.class_names[alert.fault_type]} "
                      f"P_fault={alert.p_fault:.3f}")
                break
        else:
            print("  No sustained alert triggered.")


if __name__ == "__main__":
    main()
