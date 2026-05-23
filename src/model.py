"""Transformer-LSTM architecture (Section 4.1.5, Equations 24-39, Table 4).

Pipeline per input sequence (T x 38):
  Linear projection (Eq. 24)  ->  Sinusoidal positional encoding (Eq. 25-26)
  ->  L=4 Transformer encoder layers (Eq. 27-32)
  ->  2-layer LSTM (Eq. 33-38)
  ->  Fully connected + softmax classification head (Eq. 39)

This is the *full* implementation referenced in the paper's
"Implementation Details and Reproducibility Clarification" -- it includes
positional encoding, dropout, and the explicit feed-forward dimension that
the simplified Listing 1 omits.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from configs.config import ModelConfig, FeatureConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Eq. 25-26)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerLSTM(nn.Module):
    """FFT-based Spectral-Temporal classifier."""

    def __init__(self, model_cfg: ModelConfig, feat_cfg: FeatureConfig):
        super().__init__()
        self.cfg = model_cfg

        # Eq. 24 -- learned linear projection 38 -> d_model.
        self.input_proj = nn.Linear(feat_cfg.feature_dim, model_cfg.d_model)

        # Eq. 25-26 -- sinusoidal positional encoding.
        self.pos_enc = SinusoidalPositionalEncoding(
            model_cfg.d_model, max_len=max(512, feat_cfg.sequence_length)
        )

        # Eq. 27-32 -- stacked Transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_cfg.d_model,
            nhead=model_cfg.n_heads,
            dim_feedforward=model_cfg.dim_feedforward,   # 512, overrides default 2048
            dropout=model_cfg.transformer_dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=model_cfg.n_encoder_layers
        )

        # Eq. 33-38 -- two-layer LSTM.
        self.lstm = nn.LSTM(
            input_size=model_cfg.d_model,
            hidden_size=model_cfg.lstm_hidden,
            num_layers=model_cfg.lstm_layers,
            dropout=model_cfg.lstm_dropout,
            batch_first=True,
        )

        # Eq. 39 -- classification head.
        self.classifier = nn.Linear(model_cfg.lstm_hidden, model_cfg.n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialization (Algorithm 1, line 14)."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ):
        """``x``: (batch, T, 38). Returns class logits (batch, C)."""
        z = self.input_proj(x)            # Eq. 24
        z = self.pos_enc(z)               # Eq. 25-26
        z = self.encoder(z)               # Eq. 27-32
        _, (h_n, _) = self.lstm(z)        # Eq. 33-38
        logits = self.classifier(h_n[-1])  # Eq. 39 (last layer hidden state)

        if return_attention:
            return logits, self._collect_attention(x)
        return logits

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _collect_attention(self, x: torch.Tensor):
        """Return first-layer multi-head attention weights for visualization.

        Shape: (batch, n_heads, T, T).
        """
        z = self.pos_enc(self.input_proj(x))
        layer0 = self.encoder.layers[0]
        attn = layer0.self_attn
        _, weights = attn(
            z, z, z, need_weights=True, average_attn_weights=False
        )
        return weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg) -> TransformerLSTM:
    """Construct the model from a global :class:`Config`."""
    return TransformerLSTM(cfg.model, cfg.features)
