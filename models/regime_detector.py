"""
ScalForex — Regime Detector
Learnable MLP for market regime classification (end-to-end trainable).
Regimes: trending, ranging, volatile.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegimeDetector(nn.Module):
    """Learnable regime detector — classifies market state.

    Trained end-to-end with the RL agent (not separate HMM).
    Output is a probability distribution over regimes.

    Args:
        input_dim: Input feature dimension (d_model from encoder).
        hidden_dim: Hidden layer dimension.
        num_regimes: Number of market regimes.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        num_regimes: int = 3,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_regimes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns regime probabilities.

        Args:
            x: (batch, input_dim) — encoded representation.

        Returns:
            (batch, num_regimes) — softmax probabilities.
        """
        logits = self.get_logits(x)
        return F.softmax(logits, dim=-1)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits (for cross-entropy loss).

        Args:
            x: (batch, input_dim)

        Returns:
            (batch, num_regimes)
        """
        h = self.mlp(x)
        return self.classifier(h)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get regime embedding (hidden features, not probabilities).

        Useful as additional input to the policy network.

        Args:
            x: (batch, input_dim)

        Returns:
            (batch, hidden_dim)
        """
        return self.mlp(x)
