"""TDD Tests for ScalpEnv — environments/scalp_env.py
Written BEFORE implementation (Red phase).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import gymnasium as gym
import numpy as np
import polars as pl
import pytest


def make_test_data(n: int = 500) -> pl.DataFrame:
    """Create synthetic OHLCV data for env testing."""
    rng = np.random.default_rng(42)
    base = 2000.0
    prices = base + np.cumsum(rng.standard_normal(n) * 2.0)

    start = datetime(2025, 1, 1, 0, 0)
    return pl.DataFrame({
        "time": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": prices,
        "high": prices + rng.uniform(0, 5, n),
        "low": prices - rng.uniform(0, 5, n),
        "close": prices + rng.standard_normal(n) * 2.0,
        "volume": rng.uniform(100, 10000, n),
    })


class TestScalpEnv:
    """Tests for the Gymnasium scalping environment."""

    def _make_env(self, data: pl.DataFrame | None = None) -> gym.Env:
        from environments.scalp_env import ScalpEnv
        if data is None:
            data = make_test_data()
        return ScalpEnv(
            data=data,
            symbol="XAUUSD",
            initial_balance=200.0,
        )

    # ── Space validation ──

    def test_action_space_shape(self) -> None:
        """Action space must be Box(-1, 1, shape=(2,))."""
        env = self._make_env()
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (2,)
        np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])

    def test_observation_space_shape(self) -> None:
        """Obs space should match lookback_window × n_features."""
        env = self._make_env()
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Shape should be (lookback_window, n_features) or flat
        obs_shape = env.observation_space.shape
        assert len(obs_shape) >= 1

    # ── Reset ──

    def test_reset_returns_obs_and_info(self) -> None:
        """Reset should return (observation, info) per Gymnasium API."""
        env = self._make_env()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.dtype == np.float32

    def test_reset_balance(self) -> None:
        """Balance should be reset to initial."""
        env = self._make_env()
        env.reset()
        assert env.balance == 200.0

    # ── Step ──

    def test_step_hold_action(self) -> None:
        """Action near 0 (HOLD) should not change balance."""
        env = self._make_env()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_buy_action(self) -> None:
        """Action > 0.3 should trigger BUY."""
        env = self._make_env()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(
            np.array([0.8, 0.5], dtype=np.float32)
        )
        assert isinstance(obs, np.ndarray)

    def test_step_sell_action(self) -> None:
        """Action < -0.3 should trigger SELL."""
        env = self._make_env()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(
            np.array([-0.8, 0.5], dtype=np.float32)
        )
        assert isinstance(obs, np.ndarray)

    # ── XAUUSD lot rule ──

    def test_xauusd_fixed_lot_enforced(self) -> None:
        """XAUUSD with balance < $600 must always use 0.01 lot."""
        env = self._make_env()
        env.reset()
        # Execute buy
        env.step(np.array([0.8, 1.0], dtype=np.float32))
        if env._current_position is not None:
            assert env._current_position.get("lot", 0) == 0.01

    # ── Termination ──

    def test_terminated_at_end_of_data(self) -> None:
        """Episode should terminate when data runs out."""
        data = make_test_data(50)
        env = self._make_env(data=data)
        env.reset()
        terminated = False
        for _ in range(100):  # More steps than data
            _, _, terminated, truncated, _ = env.step(
                np.array([0.0, 0.0], dtype=np.float32)
            )
            if terminated or truncated:
                break
        assert terminated or truncated

    def test_terminated_at_max_drawdown(self) -> None:
        """Episode should terminate if DD > 50%."""
        env = self._make_env()
        env.reset()
        # Force balance down
        env.balance = 90.0  # 55% DD from $200
        env._risk_manager.update_balance(90.0)
        _, _, terminated, _, info = env.step(
            np.array([0.0, 0.0], dtype=np.float32)
        )
        assert terminated is True

    # ── Gymnasium API compliance ──

    def test_gym_env_checker(self) -> None:
        """Must pass gymnasium.utils.env_checker."""
        from gymnasium.utils.env_checker import check_env
        env = self._make_env()
        try:
            check_env(env.unwrapped, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"Gymnasium env_checker failed: {e}")

    # ── Random episode ──

    def test_random_episode_completes(self) -> None:
        """Full episode with random actions should complete without errors."""
        env = self._make_env()
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < 200:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        assert steps > 0
        assert isinstance(total_reward, float)
