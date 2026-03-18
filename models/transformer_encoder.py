"""
ScalForex — Transformer Encoder
Self-Attention encoder for M5 feature sequences.
Causal masking prevents look-ahead bias.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (batch, seq, d_model)."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ScalpTransformerEncoder(nn.Module):
    """Transformer encoder for scalping feature sequences.

    Takes (batch, seq_len, n_features) input and outputs
    (batch, d_model) representation using causal self-attention.

    Args:
        n_features: Number of input features per timestep.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length.
    """

    def __init__(
        self,
        n_features: int = 27,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 60,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len + 10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(d_model)

    def generate_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Generate causal (lower-triangular) attention mask.

        Returns:
            Float mask of shape (seq_len, seq_len) where
            upper triangle = -inf, lower triangle + diagonal = 0.
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len) * float("-inf"),
            diagonal=1,
        )
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns last token's representation.

        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch, d_model) — last time step's encoded representation.
        """
        all_out = self.forward_all_positions(x)
        return all_out[:, -1, :]  # Last token

    def forward_all_positions(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning all position outputs (for testing).

        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)

        # Project features → d_model
        h = self.input_proj(x)
        h = self.pos_enc(h)

        # Causal mask
        mask = self.generate_causal_mask(seq_len).to(h.device)

        # Transformer encoder with causal mask
        h = self.transformer(h, mask=mask)
        h = self.norm(h)

        return h
