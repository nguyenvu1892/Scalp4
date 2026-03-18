"""TDD Tests for Transformer Encoder — models/transformer_encoder.py
Written BEFORE implementation (Red phase).
Focus: causal masking, no look-ahead, gradient flow.
"""
from __future__ import annotations

import torch
import pytest


class TestTransformerEncoder:
    """Tests for the Transformer encoder module."""

    def _make_encoder(self, **kwargs):
        from models.transformer_encoder import ScalpTransformerEncoder
        defaults = dict(
            n_features=27, d_model=64, nhead=4,
            num_layers=2, dim_feedforward=128,
            dropout=0.0, max_seq_len=60,
        )
        defaults.update(kwargs)
        return ScalpTransformerEncoder(**defaults)

    # ── Shape Tests ──

    def test_output_shape(self) -> None:
        """Output should be (batch, d_model) — CLS token or last token."""
        enc = self._make_encoder()
        x = torch.randn(8, 60, 27)  # (batch, seq, features)
        out = enc(x)
        assert out.shape == (8, 64)  # (batch, d_model)

    def test_variable_batch_size(self) -> None:
        """Should handle different batch sizes."""
        enc = self._make_encoder()
        for bs in [1, 4, 16]:
            x = torch.randn(bs, 60, 27)
            out = enc(x)
            assert out.shape == (bs, 64)

    def test_shorter_sequence(self) -> None:
        """Should handle sequences shorter than max_seq_len."""
        enc = self._make_encoder()
        x = torch.randn(4, 30, 27)
        out = enc(x)
        assert out.shape == (4, 64)

    # ── Causal Mask Tests ──

    def test_causal_mask_shape(self) -> None:
        """Causal mask should be (seq_len, seq_len)."""
        enc = self._make_encoder()
        mask = enc.generate_causal_mask(60)
        assert mask.shape == (60, 60)

    def test_causal_mask_upper_triangle_blocked(self) -> None:
        """Upper triangle should be -inf (blocked)."""
        enc = self._make_encoder()
        mask = enc.generate_causal_mask(10)
        # Position (i, j) where j > i should be -inf
        for i in range(10):
            for j in range(i + 1, 10):
                assert mask[i, j] == float("-inf") or mask[i, j] < -1e4

    def test_causal_mask_lower_triangle_allowed(self) -> None:
        """Lower triangle + diagonal should be 0 (allowed)."""
        enc = self._make_encoder()
        mask = enc.generate_causal_mask(10)
        for i in range(10):
            for j in range(i + 1):
                assert mask[i, j] == 0.0

    def test_no_lookahead_output_independence(self) -> None:
        """CRITICAL: Changing future tokens must NOT affect past outputs.

        This is THE most important test in the entire project.
        If this fails, the model is cheating with future data.
        """
        enc = self._make_encoder(dropout=0.0)
        enc.eval()

        # Create input
        x = torch.randn(1, 20, 27)

        # Get full output with hooks to capture per-position outputs
        # Strategy: Run twice — once with original data, once with
        # future tokens replaced. Outputs at earlier positions must match.
        with torch.no_grad():
            out1 = enc.forward_all_positions(x)  # (1, 20, d_model)

            # Modify future tokens (positions 10-19)
            x_modified = x.clone()
            x_modified[:, 10:, :] = torch.randn_like(x_modified[:, 10:, :])

            out2 = enc.forward_all_positions(x_modified)

        # Positions 0-9 should be IDENTICAL regardless of future changes
        for t in range(10):
            torch.testing.assert_close(
                out1[:, t, :], out2[:, t, :],
                msg=f"Position {t} was affected by future data! LOOK-AHEAD LEAK!",
            )

    # ── Gradient Flow ──

    def test_gradients_flow(self) -> None:
        """Gradients should flow through the entire model."""
        enc = self._make_encoder()
        enc.train()
        x = torch.randn(4, 60, 27, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_no_nan_gradients(self) -> None:
        """No NaN gradients during training."""
        enc = self._make_encoder()
        enc.train()
        x = torch.randn(4, 60, 27, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        for name, param in enc.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"

    # ── Performance ──

    def test_inference_latency(self) -> None:
        """Single forward pass should be < 50ms."""
        import time
        enc = self._make_encoder()
        enc.eval()
        x = torch.randn(1, 60, 27)

        # Warmup
        for _ in range(5):
            enc(x)

        start = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                enc(x)
        elapsed_ms = (time.perf_counter() - start) / 100 * 1000

        assert elapsed_ms < 50, f"Too slow: {elapsed_ms:.1f}ms (target < 50ms)"
