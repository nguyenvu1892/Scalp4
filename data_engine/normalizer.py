"""
ScalForex — Welford's Running Normalizer
Online normalization: z-score ≈ N(0,1) without storing full history.
Critical for live inference: normalize new data with same stats as training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class WelfordNormalizer:
    """Welford's online algorithm for running mean and variance.

    Features:
        - Numerically stable incremental computation
        - Saves/loads state for live inference
        - Per-feature normalization

    Usage:
        norm = WelfordNormalizer(n_features=29)
        norm.update(batch)           # Update stats during training
        normalized = norm.normalize(data)  # Apply normalization
    """

    def __init__(self, n_features: int) -> None:
        """Initialize normalizer.

        Args:
            n_features: Number of features to normalize.
        """
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")

        self.n_features = n_features
        self.count: int = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)

    @property
    def variance(self) -> np.ndarray:
        """Current variance estimate."""
        if self.count < 2:
            return np.ones(self.n_features, dtype=np.float64)
        return self.m2 / self.count

    @property
    def std(self) -> np.ndarray:
        """Current standard deviation estimate."""
        return np.sqrt(self.variance)

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new data.

        Args:
            x: New data, shape (n_features,) or (batch_size, n_features).
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {x.shape[1]}"
            )

        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.m2 += delta * delta2

    def normalize(self, x: np.ndarray, clip_range: float = 5.0) -> np.ndarray:
        """Normalize data to z-score, clipped to [-clip_range, clip_range].

        Args:
            x: Data to normalize, shape (n_features,) or (batch_size, n_features).
            clip_range: Clip z-scores to this range (default 5.0).

        Returns:
            Normalized array, same shape as input.
        """
        std = self.std
        std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero

        z = (x - self.mean) / std
        return np.clip(z, -clip_range, clip_range)

    def save(self, path: Path) -> None:
        """Save normalizer state to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            n_features=self.n_features,
            count=self.count,
            mean=self.mean,
            m2=self.m2,
        )

    @classmethod
    def load(cls, path: Path) -> "WelfordNormalizer":
        """Load normalizer state from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Normalizer file not found: {path}")

        data = np.load(path)
        norm = cls(n_features=int(data["n_features"]))
        norm.count = int(data["count"])
        norm.mean = data["mean"]
        norm.m2 = data["m2"]
        return norm

    def reset(self) -> None:
        """Reset all statistics."""
        self.count = 0
        self.mean = np.zeros(self.n_features, dtype=np.float64)
        self.m2 = np.zeros(self.n_features, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"WelfordNormalizer(n_features={self.n_features}, "
            f"count={self.count})"
        )
