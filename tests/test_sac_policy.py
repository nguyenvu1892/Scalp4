"""TDD Tests for SAC Policy & Action Gating
Written BEFORE implementation (Red phase).
"""
from __future__ import annotations

import torch
import numpy as np
import pytest


class TestSACPolicy:
    """Tests for SAC actor-critic policy."""

    def _make_policy(self, **kwargs):
        from agents.sac_policy import SACPolicy
        defaults = dict(
            obs_dim=64, action_dim=2,
            hidden_dims=[256, 256], log_std_range=(-5, 2),
        )
        defaults.update(kwargs)
        return SACPolicy(**defaults)

    # ── Shape Tests ──

    def test_actor_output_shape(self) -> None:
        """Actor should output (batch, 2) actions."""
        policy = self._make_policy()
        obs = torch.randn(8, 64)
        action, log_prob = policy.sample_action(obs)
        assert action.shape == (8, 2)
        assert log_prob.shape == (8, 1) or log_prob.shape == (8,)

    def test_critic_output_shape(self) -> None:
        """Critic should output (batch, 1) Q-values."""
        policy = self._make_policy()
        obs = torch.randn(8, 64)
        action = torch.randn(8, 2)
        q1, q2 = policy.critic(obs, action)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    # ── Action Range ──

    def test_actions_in_range(self) -> None:
        """Actions must be in [-1, 1] (tanh squashed)."""
        policy = self._make_policy()
        policy.eval()
        obs = torch.randn(100, 64)
        with torch.no_grad():
            actions, _ = policy.sample_action(obs)
        assert actions.min() >= -1.0 - 1e-5
        assert actions.max() <= 1.0 + 1e-5

    def test_deterministic_action(self) -> None:
        """Deterministic mode should return mean (no sampling)."""
        policy = self._make_policy()
        policy.eval()
        obs = torch.randn(4, 64)
        with torch.no_grad():
            a1 = policy.deterministic_action(obs)
            a2 = policy.deterministic_action(obs)
        torch.testing.assert_close(a1, a2)

    # ── Gradient Flow ──

    def test_actor_gradient_flow(self) -> None:
        """Gradients should flow through actor."""
        policy = self._make_policy()
        obs = torch.randn(4, 64)
        action, log_prob = policy.sample_action(obs)
        loss = log_prob.sum()
        loss.backward()
        for name, p in policy.actor.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in actor.{name}"

    def test_critic_gradient_flow(self) -> None:
        """Gradients should flow through critic."""
        policy = self._make_policy()
        obs = torch.randn(4, 64)
        action = torch.randn(4, 2)
        q1, q2 = policy.critic(obs, action)
        loss = (q1 + q2).sum()
        loss.backward()
        for name, p in policy.critic_net.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in critic.{name}"


class TestActionGating:
    """Tests for action gating (confidence threshold)."""

    def test_hold_below_threshold(self) -> None:
        """Action with |confidence| < 0.3 should become HOLD."""
        from agents.action_gating import apply_action_gating
        # Low confidence actions
        raw = np.array([[0.1, 0.5], [-0.2, 0.8], [0.29, 0.3]], dtype=np.float32)
        gated = apply_action_gating(raw, threshold=0.3)
        # All should be HOLD (direction zeroed)
        assert np.allclose(gated[:, 0], 0.0)

    def test_pass_above_threshold(self) -> None:
        """Action with |confidence| >= 0.3 should pass through."""
        from agents.action_gating import apply_action_gating
        raw = np.array([[0.8, 0.5], [-0.5, 0.3]], dtype=np.float32)
        gated = apply_action_gating(raw, threshold=0.3)
        assert gated[0, 0] == 0.8
        assert gated[1, 0] == -0.5

    def test_risk_fraction_unaffected(self) -> None:
        """Risk fraction (action[1]) should never be modified by gating."""
        from agents.action_gating import apply_action_gating
        raw = np.array([[0.1, 0.7]], dtype=np.float32)
        gated = apply_action_gating(raw, threshold=0.3)
        assert gated[0, 1] == 0.7

    def test_batch_gating(self) -> None:
        """Should work on batches."""
        from agents.action_gating import apply_action_gating
        raw = np.random.uniform(-1, 1, size=(100, 2)).astype(np.float32)
        gated = apply_action_gating(raw, threshold=0.3)
        # All low-confidence should be zeroed
        low_conf = np.abs(raw[:, 0]) < 0.3
        assert np.allclose(gated[low_conf, 0], 0.0)
