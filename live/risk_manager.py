"""
ScalForex — Risk Manager
Position sizing, DD tracking, killswitch, XAUUSD fixed lot rule.
Used by both Gymnasium env and live execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from configs.validator import TradingRulesConfig, load_symbols
from utils.logger import setup_logger

log = setup_logger("live.risk_manager")


class RiskManager:
    """Position sizing and risk management engine.

    Rules:
        - XAUUSD: fixed 0.01 lot when balance < $600
        - Other symbols: lot = (balance × max_loss × risk_fraction) / sl_distance
        - Max 2 open positions
        - Killswitch at 45% DD, force terminate at 50% DD
        - Confidence threshold: |confidence| > 0.3

    Args:
        config: TradingRulesConfig from trading_rules.yaml.
        balance: Initial account balance.
    """

    def __init__(
        self,
        config: TradingRulesConfig,
        balance: float = 200.0,
    ) -> None:
        self.config = config
        self.balance = balance
        self.peak_balance = balance

        # Load symbol specs for min/max lot
        try:
            symbols_cfg = load_symbols(Path(__file__).parent.parent / "configs")
            self._symbol_specs = symbols_cfg.symbols
        except Exception:
            self._symbol_specs = {}

    def calculate_lot(
        self,
        symbol: str,
        entry_price: float,
        sl_price: float,
        risk_fraction: float = 1.0,
    ) -> float:
        """Calculate position size in lots.

        Args:
            symbol: Trading symbol.
            entry_price: Planned entry price.
            sl_price: Stop-loss price.
            risk_fraction: Fraction of max risk to use (0.0 to 1.0).

        Returns:
            Lot size (clamped to symbol min/max).
        """
        # XAUUSD fixed lot rule
        if symbol == "XAUUSD" and self.balance < self.config.xauusd_rules.dynamic_threshold:
            return self.config.xauusd_rules.fixed_lot

        # Dynamic lot calculation: lot = risk_amount / sl_distance
        max_loss_pct = self.config.risk.max_loss_per_trade_pct
        risk_amount = self.balance * max_loss_pct * max(risk_fraction, 0.01)

        sl_distance = abs(entry_price - sl_price)
        if sl_distance < 1e-10:
            sl_distance = entry_price * 0.001  # 0.1% fallback

        lot = risk_amount / sl_distance

        # Clamp to symbol limits
        spec = self._symbol_specs.get(symbol)
        if spec:
            lot = max(lot, spec.min_lot)
            lot = min(lot, spec.max_lot)
            # Round to lot_step
            lot = round(lot / spec.lot_step) * spec.lot_step
            lot = max(lot, spec.min_lot)
        else:
            lot = max(lot, 0.01)

        return round(lot, 2)

    def update_balance(self, new_balance: float) -> None:
        """Update balance and peak tracking.

        Args:
            new_balance: Current account balance.
        """
        self.balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

    @property
    def current_dd(self) -> float:
        """Current drawdown as fraction (0.0 to 1.0)."""
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.balance) / self.peak_balance

    def should_killswitch(self) -> bool:
        """Check if killswitch threshold is breached (45% DD)."""
        return self.current_dd >= self.config.risk.killswitch_drawdown_pct

    def should_force_terminate(self) -> bool:
        """Check if force terminate threshold is breached (50% DD)."""
        return self.current_dd >= self.config.risk.max_total_drawdown_pct

    def can_open_position(self, current_open: int) -> bool:
        """Check if new position is allowed.

        Args:
            current_open: Number of currently open positions.

        Returns:
            True if allowed.
        """
        return current_open < self.config.risk.max_open_positions

    def passes_confidence(self, confidence: float) -> bool:
        """Check if confidence meets threshold.

        Args:
            confidence: Absolute confidence value from agent.

        Returns:
            True if confidence > threshold.
        """
        return abs(confidence) >= self.config.risk.confidence_threshold
