"""TDD Tests for Cross-Attention MTF — models/cross_attention_mtf.py
Written BEFORE implementation (Red phase).
CRITICAL: test_no_lookahead verifies M5 cannot attend to future H1/H4 bars.
"""
from __future__ import annotations

import torch
import pytest


class TestCrossAttentionMTF:
    """Tests for multi-timeframe cross-attention."""

    def _make_module(self, **kwargs):
        from models.cross_attention_mtf import CrossAttentionMTF
        defaults = dict(
            d_model=64, nhead=4, context_window=24, use_causal_mask=True,
        )
        defaults.update(kwargs)
        return CrossAttentionMTF(**defaults)

    # ── Shape Tests ──

    def test_output_shape(self) -> None:
        """Output shape should match query (M5) shape."""
        mod = self._make_module()
        m5 = torch.randn(4, 60, 64)    # (batch, m5_seq, d_model)
        htf = torch.randn(4, 24, 64)   # (batch, htf_seq, d_model)
        out = mod(query=m5, context=htf)
        assert out.shape == (4, 60, 64)

    def test_different_context_lengths(self) -> None:
        """Should handle variable HTF context lengths."""
        mod = self._make_module()
        m5 = torch.randn(2, 60, 64)
        for ctx_len in [12, 24, 48]:
            htf = torch.randn(2, ctx_len, 64)
            out = mod(query=m5, context=htf)
            assert out.shape == (2, 60, 64)

    # ── Causal Mask (CRITICAL) ──

    def test_causal_mask_prevents_future_context(self) -> None:
        """CRITICAL: M5 bar at time T must NOT attend to H1/H4 bars after T.

        This is the look-ahead bias prevention test for cross-attention.
        """
        mod = self._make_module(use_causal_mask=True)
        mod.eval()

        m5 = torch.randn(1, 20, 64)
        htf = torch.randn(1, 10, 64)

        # Provide timestamp indices: M5 bar i corresponds to time i
        # HTF bar j corresponds to time j*3 (e.g., H1 = 12 M5 bars)
        m5_times = torch.arange(20).unsqueeze(0).float()
        htf_times = (torch.arange(10) * 3).unsqueeze(0).float()

        with torch.no_grad():
            out1 = mod(query=m5, context=htf,
                       query_times=m5_times, context_times=htf_times)

            # Modify future context (HTF bars 7-9, times 21-27)
            htf_modified = htf.clone()
            htf_modified[:, 7:, :] = torch.randn_like(htf_modified[:, 7:, :])

            out2 = mod(query=m5, context=htf_modified,
                       query_times=m5_times, context_times=htf_times)

        # M5 bars 0-20 that can only see HTF bars 0-6 should be identical
        # HTF bar 6 has time=18, so M5 bars 0-17 should be unaffected
        for t in range(18):
            torch.testing.assert_close(
                out1[:, t, :], out2[:, t, :],
                msg=f"M5 position {t} was affected by future H1/H4! LOOK-AHEAD LEAK!",
            )

    def test_no_causal_mask_mode(self) -> None:
        """When use_causal_mask=False, all context is visible."""
        mod = self._make_module(use_causal_mask=False)
        mod.eval()

        m5 = torch.randn(1, 20, 64)
        htf = torch.randn(1, 10, 64)

        with torch.no_grad():
            out = mod(query=m5, context=htf)
        assert out.shape == (1, 20, 64)

    # ── Gradient Flow ──

    def test_gradients_flow(self) -> None:
        """Gradients should flow through cross-attention."""
        mod = self._make_module()
        m5 = torch.randn(4, 60, 64, requires_grad=True)
        htf = torch.randn(4, 24, 64, requires_grad=True)
        out = mod(query=m5, context=htf)
        loss = out.sum()
        loss.backward()
        assert m5.grad is not None
        assert htf.grad is not None
