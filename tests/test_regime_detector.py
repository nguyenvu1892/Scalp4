"""TDD Tests for Regime Detector — models/regime_detector.py
Written BEFORE implementation (Red phase).
"""
from __future__ import annotations

import torch
import pytest


class TestRegimeDetector:
    """Tests for the learnable regime detector MLP."""

    def _make_detector(self, **kwargs):
        from models.regime_detector import RegimeDetector
        defaults = dict(input_dim=64, hidden_dim=64, num_regimes=3)
        defaults.update(kwargs)
        return RegimeDetector(**defaults)

    # ── Shape ──

    def test_output_shape(self) -> None:
        """Output should be (batch, num_regimes) probabilities."""
        det = self._make_detector()
        x = torch.randn(8, 64)
        out = det(x)
        assert out.shape == (8, 3)

    def test_output_sums_to_one(self) -> None:
        """Output should be valid probability distribution."""
        det = self._make_detector()
        x = torch.randn(16, 64)
        out = det(x)
        sums = out.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(16), atol=1e-5, rtol=1e-5)

    def test_output_non_negative(self) -> None:
        """All probabilities should be >= 0."""
        det = self._make_detector()
        x = torch.randn(16, 64)
        out = det(x)
        assert (out >= 0).all()

    # ── Different Configs ──

    def test_custom_regimes(self) -> None:
        """Should support different number of regimes."""
        det = self._make_detector(num_regimes=5)
        x = torch.randn(4, 64)
        out = det(x)
        assert out.shape == (4, 5)

    def test_regime_embedding(self) -> None:
        """Should provide regime embedding for downstream use."""
        det = self._make_detector()
        x = torch.randn(4, 64)
        embedding = det.get_embedding(x)
        assert embedding.shape == (4, 64)  # hidden_dim

    # ── Gradient Flow ──

    def test_gradients_flow(self) -> None:
        """Gradients should flow for end-to-end training."""
        det = self._make_detector()
        x = torch.randn(4, 64, requires_grad=True)
        out = det(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_regime_labels_training(self) -> None:
        """Should be trainable with cross-entropy loss."""
        det = self._make_detector()
        x = torch.randn(8, 64)
        labels = torch.randint(0, 3, (8,))
        logits = det.get_logits(x)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        # Should not crash
        assert loss.item() > 0
