"""
ScalForex — Scalping Gymnasium Environment
Gymnasium-compatible env for DRL Scalping training.

Action Space: Box(-1, 1, shape=(2,))
  - action[0]: Direction & Confidence (< -0.3 SELL, > 0.3 BUY, else HOLD)
  - action[1]: Risk Fraction (mapped from [-1,1] to [0,1] via (x+1)/2)

Observation Space: Box(-inf, inf, shape=(lookback_window * n_features,))
  - Flattened window of normalized features

Rules:
  - XAUUSD: fixed 0.01 lot when balance < $600
  - Max 2 open positions
  - Force terminate at 50% DD
  - Pessimistic intra-candle execution
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

from configs.validator import (
    FeatureSettings,
    RewardConfig,
    load_symbols,
    load_trading_rules,
    load_train_config,
)
from data_engine.feature_builder import build_features
from data_engine.normalizer import WelfordNormalizer
from environments.market_sim import MarketSim
from environments.reward_engine import RewardEngine
from live.risk_manager import RiskManager
from utils.logger import setup_logger

log = setup_logger("environments.scalp_env")

CONFIG_DIR = Path(__file__).parent.parent / "configs"


class ScalpEnv(gym.Env):
    """Gymnasium environment for DRL Scalping training.

    Simulates a realistic trading session on one symbol with accurate
    spread, slippage, and risk management.

    Args:
        data: Raw OHLCV DataFrame (will be feature-engineered).
        symbol: Trading symbol name.
        initial_balance: Starting account balance.
        lookback_window: Number of bars in observation window.
        config_dir: Path to configs directory.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        data: pl.DataFrame,
        symbol: str = "XAUUSD",
        initial_balance: float = 200.0,
        lookback_window: int = 60,
        config_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()

        cfg_dir = config_dir or CONFIG_DIR

        # Load configs
        self._trading_rules = load_trading_rules(cfg_dir)
        self._symbols_cfg = load_symbols(cfg_dir)
        self._train_cfg = load_train_config(cfg_dir)

        self._symbol = symbol
        self._initial_balance = initial_balance
        self._lookback = lookback_window

        # Feature engineering
        feat_settings = self._symbols_cfg.features
        self._feature_data = build_features(data, config=feat_settings)
        self._n_features = 27  # 29 - 2 multi-TF (not available in single-symbol env)

        # Feature columns to use (excluding time, OHLCV, and multi-TF)
        self._feature_cols = [
            "candle_body_ratio", "upper_wick_ratio", "lower_wick_ratio",
            "pin_bar", "engulfing", "inside_bar",
            "relative_volume", "volume_delta", "climax_volume",
            "swing_high", "swing_low", "bos", "choch",
            "order_block_bull", "order_block_bear",
            "fvg_bull", "fvg_bear",
            "liquidity_above", "liquidity_below",
            "trend_direction",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_id",
            "log_return", "range_pct",
        ]

        # Normalizer
        self._normalizer = WelfordNormalizer(n_features=self._n_features)
        self._warmup_normalizer()

        # Action space: [-1, 1] × 2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        # Observation space: flat vector
        obs_dim = self._lookback * self._n_features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        # Sub-systems
        symbol_spec = self._symbols_cfg.symbols.get(symbol)
        self._market_sim = MarketSim(
            symbol=symbol,
            avg_spread_pips=symbol_spec.avg_spread_pips if symbol_spec else 20,
            pip_size=symbol_spec.pip_size if symbol_spec else 0.01,
            spread_mode="variable",
        )
        self._reward_engine = RewardEngine(self._train_cfg.reward)
        self._risk_manager = RiskManager(
            config=self._trading_rules,
            balance=initial_balance,
        )

        # State
        self.balance = initial_balance
        self._step_idx = 0
        self._current_position: Optional[dict[str, Any]] = None
        self._trades_today: int = 0
        self._total_trades: int = 0
        self._episode_pnl: float = 0.0

    @property
    def symbol(self) -> str:
        return self._symbol

    def _warmup_normalizer(self) -> None:
        """Train normalizer on available data."""
        feat_matrix = self._feature_data.select(self._feature_cols).to_numpy()
        # Remove rows with NaN
        valid = ~np.isnan(feat_matrix).any(axis=1)
        feat_matrix = feat_matrix[valid]
        if len(feat_matrix) > 0:
            self._normalizer.update(feat_matrix)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed.
            options: Additional reset options.

        Returns:
            (observation, info) tuple.
        """
        super().reset(seed=seed)

        # Reseed MarketSim RNG for deterministic behavior
        rng_seed = seed if seed is not None else 42
        self._market_sim._rng = np.random.default_rng(rng_seed)

        self.balance = self._initial_balance
        self._risk_manager = RiskManager(
            config=self._trading_rules,
            balance=self._initial_balance,
        )
        self._step_idx = self._lookback
        self._current_position = None
        self._trades_today = 0
        self._total_trades = 0
        self._episode_pnl = 0.0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Array of shape (2,)
                action[0]: direction & confidence [-1, 1]
                action[1]: risk fraction [-1, 1] → mapped to [0, 1]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        direction_confidence = float(action[0])
        raw_risk = float(action[1])
        risk_fraction = (raw_risk + 1.0) / 2.0  # Map [-1,1] → [0,1]
        risk_fraction = np.clip(risk_fraction, 0.01, 1.0)

        confidence_threshold = self._trading_rules.risk.confidence_threshold
        reward = 0.0
        terminated = False
        truncated = False

        # Get current candle
        if self._step_idx >= len(self._feature_data):
            terminated = True
            obs = self._get_observation(clip_idx=True)
            return obs, 0.0, terminated, truncated, self._get_info()

        candle = self._feature_data.row(self._step_idx, named=True)
        current_price = candle["close"]
        candle_high = candle["high"]
        candle_low = candle["low"]
        candle_open = candle["open"]
        hour = candle.get("time")
        hour_val = hour.hour if hasattr(hour, "hour") else 12

        # === Check existing position exit ===
        is_close = False
        realized_pnl = 0.0
        bars_held = 0
        risk_amount = 0.0
        spread_cost = 0.0

        if self._current_position is not None:
            pos = self._current_position
            bars_held = self._step_idx - pos["entry_step"]

            # Check SL/TP
            exit_result = self._market_sim.check_exit(
                direction=pos["direction"],
                entry_price=pos["entry_price"],
                sl_price=pos["sl_price"],
                tp_price=pos["tp_price"],
                candle_high=candle_high,
                candle_low=candle_low,
                candle_open=candle_open,
            )

            if exit_result["exit_type"] != "none":
                # Close position
                exit_price = float(exit_result["exit_price"])
                pnl = (exit_price - pos["entry_price"]) * pos["direction"] * pos["lot"]
                realized_pnl = pnl
                risk_amount = pos.get("risk_amount", 1.0)
                spread_cost = pos.get("spread_cost", 0.0)
                self.balance += pnl
                self._episode_pnl += pnl
                self._current_position = None
                is_close = True
                self._risk_manager.update_balance(self.balance)
                log.debug(
                    f"Closed {pos['direction']} @ {exit_price:.2f} "
                    f"({exit_result['exit_type']}), PnL={pnl:.2f}, "
                    f"Balance={self.balance:.2f}"
                )

        # === Process new action ===
        if self._current_position is None and abs(direction_confidence) > confidence_threshold:
            if self._risk_manager.can_open_position(0):
                direction = 1 if direction_confidence > 0 else -1

                # Calculate entry with spread + slippage
                bid, ask = self._market_sim.apply_spread(current_price, hour_val)
                if direction == 1:
                    entry = self._market_sim.apply_slippage(ask, direction, 1000)
                else:
                    entry = self._market_sim.apply_slippage(bid, direction, 1000)

                # SL/TP based on recent range
                atr_proxy = candle_high - candle_low
                atr_proxy = max(atr_proxy, current_price * 0.001)
                sl_distance = atr_proxy * 1.5
                tp_distance = atr_proxy * 2.0  # 1:1.33 R:R minimum

                if direction == 1:
                    sl_price = entry - sl_distance
                    tp_price = entry + tp_distance
                else:
                    sl_price = entry + sl_distance
                    tp_price = entry - tp_distance

                # Calculate lot
                lot = self._risk_manager.calculate_lot(
                    symbol=self._symbol,
                    entry_price=entry,
                    sl_price=sl_price,
                    risk_fraction=risk_fraction,
                )

                trade_cost = self._market_sim.compute_trade_cost(lot, current_price, hour_val)

                self._current_position = {
                    "direction": direction,
                    "entry_price": entry,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "lot": lot,
                    "entry_step": self._step_idx,
                    "risk_amount": sl_distance * lot,
                    "spread_cost": trade_cost,
                }
                self._trades_today += 1
                self._total_trades += 1

                log.debug(
                    f"Opened {'BUY' if direction == 1 else 'SELL'} "
                    f"@ {entry:.2f}, lot={lot}, SL={sl_price:.2f}, TP={tp_price:.2f}"
                )

        # === Compute unrealized PnL ===
        unrealized_pnl = 0.0
        if self._current_position is not None:
            pos = self._current_position
            unrealized_pnl = (
                (current_price - pos["entry_price"]) * pos["direction"] * pos["lot"]
            )
            bars_held = self._step_idx - pos["entry_step"]

        # === Compute reward ===
        reward = self._reward_engine.compute_total_reward(
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            bars_held=bars_held,
            risk_amount=risk_amount if is_close else (
                self._current_position.get("risk_amount", 0) if self._current_position else 0
            ),
            spread_cost=spread_cost,
            commission=0.0,
            trades_today=self._trades_today,
            current_dd=self._risk_manager.current_dd,
            balance=self.balance,
            is_close=is_close,
        )

        # === Check termination ===
        if self._risk_manager.should_force_terminate():
            terminated = True
            log.info(
                f"KILLSWITCH: DD={self._risk_manager.current_dd:.1%} "
                f"Balance={self.balance:.2f}"
            )

        # Advance step
        self._step_idx += 1
        if self._step_idx >= len(self._feature_data):
            truncated = True

        obs = self._get_observation(clip_idx=True)
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def _get_observation(self, clip_idx: bool = False) -> np.ndarray:
        """Get current observation as normalized feature window.

        Returns:
            Flattened array of shape (lookback_window * n_features,).
        """
        idx = min(self._step_idx, len(self._feature_data) - 1) if clip_idx else self._step_idx
        start = max(0, idx - self._lookback)
        end = idx

        if end <= start:
            return np.zeros(
                self._lookback * self._n_features, dtype=np.float32,
            )

        window = self._feature_data.slice(start, end - start)
        feat_matrix = window.select(self._feature_cols).to_numpy()

        # Handle NaN
        feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)

        # Normalize
        normalized = self._normalizer.normalize(feat_matrix)

        # Pad if window is shorter than lookback
        if len(normalized) < self._lookback:
            pad = np.zeros(
                (self._lookback - len(normalized), self._n_features),
                dtype=np.float64,
            )
            normalized = np.vstack([pad, normalized])

        return normalized.flatten().astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        """Get current environment info dict."""
        return {
            "balance": self.balance,
            "drawdown": self._risk_manager.current_dd,
            "episode_pnl": self._episode_pnl,
            "total_trades": self._total_trades,
            "trades_today": self._trades_today,
            "step": self._step_idx,
            "has_position": self._current_position is not None,
        }
