"""TDD Tests for MarketSim — environments/market_sim.py
Written BEFORE implementation (Red phase).
"""
from __future__ import annotations

import numpy as np
import pytest


class TestMarketSim:
    """Tests for market simulation engine."""

    def test_init_with_symbol_config(self) -> None:
        """Should initialize with symbol-specific spread/pip config."""
        from environments.market_sim import MarketSim

        sim = MarketSim(
            symbol="XAUUSD",
            avg_spread_pips=20,
            pip_size=0.01,
        )
        assert sim.symbol == "XAUUSD"
        assert sim.avg_spread_pips == 20

    def test_apply_spread(self) -> None:
        """Should widen bid-ask around mid price."""
        from environments.market_sim import MarketSim

        sim = MarketSim(symbol="XAUUSD", avg_spread_pips=20, pip_size=0.01)
        bid, ask = sim.apply_spread(mid_price=2000.0)
        assert bid < 2000.0
        assert ask > 2000.0
        assert ask - bid == pytest.approx(20 * 0.01, abs=0.01)

    def test_variable_spread_wider_at_session_change(self) -> None:
        """Spread should be wider during session transitions."""
        from environments.market_sim import MarketSim

        sim = MarketSim(
            symbol="XAUUSD", avg_spread_pips=20, pip_size=0.01,
            spread_mode="variable",
        )
        bid_normal, ask_normal = sim.apply_spread(2000.0, hour=10)
        bid_trans, ask_trans = sim.apply_spread(2000.0, hour=8)  # Session boundary
        # Session boundary spread should be >= normal spread
        assert (ask_trans - bid_trans) >= (ask_normal - bid_normal)

    def test_apply_slippage(self) -> None:
        """Slippage should worsen entry price."""
        from environments.market_sim import MarketSim

        sim = MarketSim(symbol="XAUUSD", avg_spread_pips=20, pip_size=0.01)
        # BUY: slippage pushes price UP
        entry = sim.apply_slippage(price=2000.0, direction=1, volume=5000)
        assert entry >= 2000.0
        # SELL: slippage pushes price DOWN
        entry = sim.apply_slippage(price=2000.0, direction=-1, volume=5000)
        assert entry <= 2000.0

    def test_pessimistic_execution_sl_before_tp(self) -> None:
        """When both SL and TP are hit in same candle, SL must trigger first."""
        from environments.market_sim import MarketSim

        sim = MarketSim(symbol="XAUUSD", avg_spread_pips=20, pip_size=0.01)

        # BUY position: SL below, TP above
        # Candle where both are hit: low < SL and high > TP
        result = sim.check_exit(
            direction=1,
            entry_price=2000.0,
            sl_price=1990.0,
            tp_price=2010.0,
            candle_high=2015.0,
            candle_low=1985.0,
            candle_open=2000.0,
        )
        # Pessimistic: SL should be triggered
        assert result["exit_type"] == "sl"

    def test_no_exit_when_price_within_range(self) -> None:
        """No exit if candle stays between SL and TP."""
        from environments.market_sim import MarketSim

        sim = MarketSim(symbol="XAUUSD", avg_spread_pips=20, pip_size=0.01)
        result = sim.check_exit(
            direction=1,
            entry_price=2000.0,
            sl_price=1990.0,
            tp_price=2010.0,
            candle_high=2008.0,
            candle_low=1992.0,
            candle_open=2000.0,
        )
        assert result["exit_type"] == "none"

    def test_tp_hit_without_sl(self) -> None:
        """TP should trigger when only TP is reached."""
        from environments.market_sim import MarketSim

        sim = MarketSim(symbol="XAUUSD", avg_spread_pips=20, pip_size=0.01)
        result = sim.check_exit(
            direction=1,
            entry_price=2000.0,
            sl_price=1990.0,
            tp_price=2005.0,
            candle_high=2008.0,
            candle_low=1995.0,
            candle_open=2002.0,
        )
        assert result["exit_type"] == "tp"

    def test_sell_direction_exits(self) -> None:
        """SELL positions: SL is above entry, TP is below."""
        from environments.market_sim import MarketSim

        sim = MarketSim(symbol="XAUUSD", avg_spread_pips=20, pip_size=0.01)
        # SELL: TP hit (price drops)
        result = sim.check_exit(
            direction=-1,
            entry_price=2000.0,
            sl_price=2010.0,
            tp_price=1990.0,
            candle_high=2005.0,
            candle_low=1988.0,
            candle_open=2000.0,
        )
        assert result["exit_type"] == "tp"
