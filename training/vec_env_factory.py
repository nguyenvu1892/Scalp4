"""
ScalForex — Vectorized Environment Factory
Creates SubprocVecEnv with 30-40 parallel ScalpEnv instances.
Optimized for Dual Xeon 56 threads — massively parallel rollout collection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import polars as pl

from utils.logger import setup_logger

log = setup_logger("training.vec_env_factory")


def make_env_fn(
    data: pl.DataFrame,
    symbol: str,
    initial_balance: float,
    lookback_window: int,
    config_dir: Optional[Path] = None,
    seed: int = 0,
) -> Callable:
    """Create a factory function for one ScalpEnv instance.

    Args:
        data: OHLCV data.
        symbol: Trading symbol.
        initial_balance: Starting balance.
        lookback_window: Feature lookback window.
        config_dir: Path to configs.
        seed: Environment seed.

    Returns:
        Callable that creates a ScalpEnv.
    """
    def _init() -> Any:
        from environments.scalp_env import ScalpEnv
        env = ScalpEnv(
            data=data,
            symbol=symbol,
            initial_balance=initial_balance,
            lookback_window=lookback_window,
            config_dir=config_dir,
        )
        env.reset(seed=seed)
        return env
    return _init


def create_vec_env(
    data: pl.DataFrame,
    symbol: str = "XAUUSD",
    n_envs: int = 32,
    initial_balance: float = 200.0,
    lookback_window: int = 60,
    config_dir: Optional[Path] = None,
    use_subprocess: bool = True,
) -> Any:
    """Create vectorized environment (SubprocVecEnv or DummyVecEnv).

    For Dual Xeon 56 threads: use n_envs=30-40 for optimal parallelism.
    SubprocVecEnv spawns each env in a separate process.

    Args:
        data: OHLCV data shared across envs.
        symbol: Trading symbol.
        n_envs: Number of parallel environments.
        initial_balance: Starting balance per env.
        lookback_window: Feature lookback window.
        config_dir: Path to configs.
        use_subprocess: Use SubprocVecEnv (True) or DummyVecEnv (False).

    Returns:
        Vectorized env instance.
    """
    env_fns = [
        make_env_fn(
            data=data,
            symbol=symbol,
            initial_balance=initial_balance,
            lookback_window=lookback_window,
            config_dir=config_dir,
            seed=i * 42,
        )
        for i in range(n_envs)
    ]

    if use_subprocess:
        try:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            log.info(f"Creating SubprocVecEnv with {n_envs} parallel envs")
            return SubprocVecEnv(env_fns)
        except ImportError:
            log.warning("stable_baselines3 not available, falling back to DummyVecEnv")

    # Fallback: sequential (for testing or when SB3 not available)
    from gymnasium.vector import SyncVectorEnv
    log.info(f"Creating SyncVectorEnv with {n_envs} envs")
    return SyncVectorEnv(env_fns)
