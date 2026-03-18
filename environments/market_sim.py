"""
ScalForex — Market Simulator
Accurate spread, slippage, and intra-candle exit logic for backtesting.
PESSIMISTIC EXECUTION: SL always triggers before TP when both hit in same candle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.logger import setup_logger

log = setup_logger("environments.market_sim")

# Session boundary hours (UTC) — wider spread during transitions
_SESSION_BOUNDARIES: list[int] = [0, 8, 15, 22]


@dataclass
class MarketSim:
    """Market physics simulator — spread, slippage, exit logic.

    Args:
        symbol: Trading symbol name.
        avg_spread_pips: Average spread in pips for this symbol.
        pip_size: Value of 1 pip in price units.
        spread_mode: 'fixed', 'variable', or 'realistic'.
        slippage_factor: Base slippage as fraction of spread.
    """

    symbol: str
    avg_spread_pips: float
    pip_size: float
    spread_mode: str = "fixed"
    slippage_factor: float = 0.1
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42),
        repr=False,
    )

    def apply_spread(
        self,
        mid_price: float,
        hour: int = 12,
    ) -> tuple[float, float]:
        """Apply bid-ask spread around mid price.

        Args:
            mid_price: Market mid price.
            hour: Current UTC hour (for variable spread).

        Returns:
            (bid, ask) tuple.
        """
        spread_pips = self.avg_spread_pips

        if self.spread_mode == "variable":
            # Wider spread near session boundaries
            min_dist = min(abs(hour - b) for b in _SESSION_BOUNDARIES)
            if min_dist <= 1:
                spread_pips *= 1.5  # 50% wider at session change
            # Add random noise ±20%
            spread_pips *= (1.0 + self._rng.uniform(-0.2, 0.2))

        elif self.spread_mode == "realistic":
            # Time-of-day dependent + random
            min_dist = min(abs(hour - b) for b in _SESSION_BOUNDARIES)
            if min_dist <= 1:
                spread_pips *= 2.0  # Double at session change
            spread_pips *= (1.0 + self._rng.uniform(-0.3, 0.5))

        half_spread = (spread_pips * self.pip_size) / 2.0
        bid = mid_price - half_spread
        ask = mid_price + half_spread
        return bid, ask

    def apply_slippage(
        self,
        price: float,
        direction: int,
        volume: float,
    ) -> float:
        """Apply slippage to entry/exit price.

        BUY: price goes UP (worse entry).
        SELL: price goes DOWN (worse entry).

        Args:
            price: Target price.
            direction: 1 for BUY, -1 for SELL.
            volume: Trade volume (higher = more slippage).

        Returns:
            Adjusted price with slippage.
        """
        base_slip = self.slippage_factor * self.avg_spread_pips * self.pip_size
        # Volume-dependent: more volume = more slippage
        vol_factor = 1.0 + np.log1p(volume / 1000.0) * 0.1
        slip = base_slip * vol_factor * self._rng.uniform(0.0, 1.0)
        return price + direction * slip

    def check_exit(
        self,
        direction: int,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        candle_high: float,
        candle_low: float,
        candle_open: float,
    ) -> dict[str, object]:
        """Check if SL or TP is hit during a candle.

        PESSIMISTIC EXECUTION: If both SL and TP are reached in the same
        candle, SL is always triggered first (worst case for trader).

        Args:
            direction: 1 for BUY, -1 for SELL.
            entry_price: Position entry price.
            sl_price: Stop-loss price level.
            tp_price: Take-profit price level.
            candle_high: Candle high price.
            candle_low: Candle low price.
            candle_open: Candle open price.

        Returns:
            Dict with 'exit_type' ('sl', 'tp', 'none') and 'exit_price'.
        """
        if direction == 1:  # BUY position
            sl_hit = candle_low <= sl_price
            tp_hit = candle_high >= tp_price
        else:  # SELL position
            sl_hit = candle_high >= sl_price
            tp_hit = candle_low <= tp_price

        if sl_hit and tp_hit:
            # PESSIMISTIC: SL always first when both hit
            exit_price = sl_price
            return {"exit_type": "sl", "exit_price": exit_price}
        elif sl_hit:
            return {"exit_type": "sl", "exit_price": sl_price}
        elif tp_hit:
            return {"exit_type": "tp", "exit_price": tp_price}
        else:
            return {"exit_type": "none", "exit_price": None}

    def compute_trade_cost(
        self,
        lot: float,
        mid_price: float,
        hour: int = 12,
    ) -> float:
        """Compute total cost of a trade (spread + commission).

        Args:
            lot: Position size in lots.
            mid_price: Current mid price.
            hour: UTC hour for variable spread.

        Returns:
            Total cost in account currency.
        """
        bid, ask = self.apply_spread(mid_price, hour)
        spread_cost = (ask - bid) * lot * (1.0 / self.pip_size)
        # Exness: zero commission on most accounts
        commission = 0.0
        return spread_cost + commission
