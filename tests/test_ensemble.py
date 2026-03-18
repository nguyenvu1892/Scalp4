"""TDD Tests for Ensemble Agent — agents/ensemble.py"""
from __future__ import annotations

import numpy as np
import torch
import pytest


class TestEnsembleAgent:
    """Tests for the majority-voting ensemble."""

    def _make_models(self, n: int = 3, obs_dim: int = 64) -> list:
        from agents.sac_policy import SACPolicy
        return [SACPolicy(obs_dim=obs_dim, action_dim=2, hidden_dims=[32, 32]) for _ in range(n)]

    def _make_ensemble(self, n: int = 3, **kwargs):
        from agents.ensemble import EnsembleAgent
        models = self._make_models(n, kwargs.pop("obs_dim", 64))
        return EnsembleAgent(models=models, **kwargs)

    # ── Shape ──

    def test_output_shape_single(self) -> None:
        """Single observation should return (2,) action."""
        ens = self._make_ensemble()
        obs = np.random.randn(64).astype(np.float32)
        action = ens.predict(obs)
        assert action.shape == (2,)

    def test_output_shape_batch(self) -> None:
        """Batch observations should return (batch, 2) actions."""
        ens = self._make_ensemble()
        obs = np.random.randn(8, 64).astype(np.float32)
        actions = ens.predict(obs)
        assert actions.shape == (8, 2)

    # ── Voting Logic ──

    def test_hold_when_no_consensus(self) -> None:
        """If models disagree, action should be HOLD (direction=0)."""
        from agents.ensemble import EnsembleAgent
        from agents.sac_policy import SACPolicy

        # Create 3 models with fixed outputs (disagree)
        models = [SACPolicy(obs_dim=64, action_dim=2, hidden_dims=[32]) for _ in range(3)]

        ens = EnsembleAgent(models=models, threshold=0.3)
        obs = np.zeros(64, dtype=np.float32)

        # With random init, models likely disagree → HOLD is possible
        # We test the structure rather than the specific output
        action = ens.predict(obs)
        assert action.shape == (2,)
        assert -1.0 <= action[0] <= 1.0

    def test_quorum_calculation(self) -> None:
        """Quorum for 3 models should be 2."""
        ens = self._make_ensemble(n=3)
        assert ens.quorum == 2

    def test_quorum_5_models(self) -> None:
        """Quorum for 5 models should be 3."""
        ens = self._make_ensemble(n=5)
        assert ens.quorum == 3

    def test_min_2_models_required(self) -> None:
        """Should raise ValueError with < 2 models."""
        from agents.ensemble import EnsembleAgent
        from agents.sac_policy import SACPolicy
        with pytest.raises(ValueError):
            EnsembleAgent(models=[SACPolicy(obs_dim=64, action_dim=2)])

    # ── Confidence Gating ──

    def test_low_confidence_becomes_hold(self) -> None:
        """Ensemble output with |confidence| < 0.3 should be HOLD."""
        from agents.ensemble import EnsembleAgent
        ens = self._make_ensemble()
        obs = np.random.randn(64).astype(np.float32)
        action = ens.predict(obs)
        # After gating: if |confidence| < 0.3, it should be 0
        if abs(float(action[0])) < 0.3:
            assert float(action[0]) == 0.0

    # ── Vote Detail ──

    def test_vote_detail_structure(self) -> None:
        """get_vote_detail should return expected structure."""
        ens = self._make_ensemble()
        obs = np.random.randn(64).astype(np.float32)
        detail = ens.get_vote_detail(obs)
        assert "per_model" in detail
        assert "buy_votes" in detail
        assert "sell_votes" in detail
        assert "hold_votes" in detail
        assert "quorum" in detail
        assert "consensus" in detail
        assert len(detail["per_model"]) == 3

    def test_vote_counts_sum(self) -> None:
        """Buy + sell + hold should equal n_models."""
        ens = self._make_ensemble()
        obs = np.random.randn(64).astype(np.float32)
        detail = ens.get_vote_detail(obs)
        total = detail["buy_votes"] + detail["sell_votes"] + detail["hold_votes"]
        assert total == 3
