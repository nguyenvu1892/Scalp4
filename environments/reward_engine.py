"""
ScalForex — Reward Engine
8-component reward system optimized for Scalping.
All weights from configs/train_config.yaml (zero hardcode).
"""

from __future__ import annotations

import math

from configs.validator import RewardConfig
from utils.logger import setup_logger

log = setup_logger("environments.reward_engine")


class RewardEngine:
    """8-component reward function for DRL Scalping agent.

    Components:
        1. R_pnl: Realized PnL (normalized by balance)
        2. R_shaping: Unrealized PnL (small, dẫn đường)
        3. R_scalp_bonus: Fast close with profit
        4. R_rr_bonus: Good risk/reward ratio
        5. R_cost: Spread + commission penalty (×1.5)
        6. R_hold_penalty: Ôm lệnh quá lâu
        7. R_overtrade: Too many trades per day
        8. R_dd_penalty: Exponential drawdown penalty

    All weights are configurable via RewardConfig (from YAML).
    """

    def __init__(self, config: RewardConfig) -> None:
        """Initialize with reward config.

        Args:
            config: RewardConfig from train_config.yaml.
        """
        self.cfg = config

    def compute_pnl_reward(
        self,
        realized_pnl: float,
        balance: float,
    ) -> float:
        """R1: Realized PnL, normalized by balance.

        Args:
            realized_pnl: Actual PnL from closed position.
            balance: Current account balance.

        Returns:
            Weighted PnL reward.
        """
        safe_balance = max(balance, 1.0)
        return self.cfg.w_pnl * (realized_pnl / safe_balance)

    def compute_shaping_reward(
        self,
        unrealized_pnl: float,
        balance: float,
    ) -> float:
        """R2: Unrealized PnL shaping (small signal).

        Args:
            unrealized_pnl: Current unrealized PnL.
            balance: Current account balance.

        Returns:
            Small shaping reward.
        """
        safe_balance = max(balance, 1.0)
        return self.cfg.w_shaping * (unrealized_pnl / safe_balance)

    def compute_scalp_bonus(
        self,
        bars_held: int,
        realized_pnl: float,
    ) -> float:
        """R3: Scalp bonus for fast profitable closes.

        Args:
            bars_held: Number of bars position was held.
            realized_pnl: PnL from the closed trade.

        Returns:
            Bonus (0 if loss or slow close).
        """
        if realized_pnl <= 0:
            return 0.0

        if bars_held < 6:
            bonus = self.cfg.scalp_bonus_fast
        elif bars_held < 30:
            bonus = self.cfg.scalp_bonus_medium
        else:
            return 0.0

        return self.cfg.w_scalp_bonus * bonus

    def compute_rr_bonus(
        self,
        realized_pnl: float,
        risk_amount: float,
    ) -> float:
        """R4: Risk/Reward ratio bonus.

        Args:
            realized_pnl: Actual PnL.
            risk_amount: Amount risked (SL distance × lot).

        Returns:
            Bonus proportional to R:R ratio.
        """
        if realized_pnl <= 0 or risk_amount <= 0:
            return 0.0

        rr_ratio = realized_pnl / risk_amount
        # Cap at 3:1 to prevent extreme bonuses
        capped_rr = min(rr_ratio, 3.0)
        return self.cfg.w_rr_bonus * capped_rr * 0.1

    def compute_cost_penalty(
        self,
        spread_cost: float,
        commission: float,
        balance: float,
    ) -> float:
        """R5: Trading cost penalty (spread + commission × w_cost).

        Args:
            spread_cost: Spread cost of the trade.
            commission: Commission cost.
            balance: Current balance.

        Returns:
            Negative penalty (amplified by w_cost=1.5).
        """
        safe_balance = max(balance, 1.0)
        total_cost = spread_cost + commission
        return -self.cfg.w_cost * (total_cost / safe_balance)

    def compute_hold_penalty(
        self,
        bars_held: int,
    ) -> float:
        """R6: Hold penalty — increases with bars held.

        Args:
            bars_held: Number of bars position has been open.

        Returns:
            Negative penalty (×1.2 to discourage swing trading).
        """
        # Quadratic scaling: penalty grows faster for longer holds
        penalty = 0.001 * (bars_held ** 1.5)
        return -self.cfg.w_hold_penalty * penalty

    def compute_overtrade_penalty(
        self,
        trades_today: int,
    ) -> float:
        """R7: Overtrade penalty — heavy penalty for spam trades.

        Args:
            trades_today: Number of trades executed today.

        Returns:
            0 if within limit, negative penalty if exceeding (×2.0).
        """
        max_trades = self.cfg.max_trades_per_day
        if trades_today <= max_trades:
            return 0.0

        excess = trades_today - max_trades
        return -self.cfg.w_overtrade * 0.1 * excess

    def compute_dd_penalty(
        self,
        current_dd: float,
    ) -> float:
        """R8: Exponential drawdown penalty.

        Formula: -alpha * exp(beta * dd) + alpha
        At dd=0: penalty=0. As dd→50%: penalty explodes.

        Args:
            current_dd: Current drawdown as fraction (0.0 to 1.0).

        Returns:
            Negative penalty (exponential growth near 50% DD).
        """
        alpha = self.cfg.dd_penalty_alpha
        beta = self.cfg.dd_penalty_beta
        # Exponential penalty: grows rapidly near killswitch threshold
        penalty = alpha * (math.exp(beta * current_dd) - 1.0)
        return -penalty

    def compute_total_reward(
        self,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        bars_held: int = 0,
        risk_amount: float = 0.0,
        spread_cost: float = 0.0,
        commission: float = 0.0,
        trades_today: int = 0,
        current_dd: float = 0.0,
        balance: float = 200.0,
        is_close: bool = False,
    ) -> float:
        """Compute total reward from all 8 components.

        Args:
            realized_pnl: PnL from closed trade (0 if no close).
            unrealized_pnl: Current unrealized PnL (0 if no position).
            bars_held: Bars since position opened.
            risk_amount: Amount risked on current trade.
            spread_cost: Spread cost of entry.
            commission: Commission cost.
            trades_today: Trades executed today.
            current_dd: Current account drawdown fraction.
            balance: Current account balance.
            is_close: Whether a position was just closed.

        Returns:
            Total weighted reward (float).
        """
        total = 0.0

        # R1: Realized PnL (only on close)
        if is_close:
            total += self.compute_pnl_reward(realized_pnl, balance)
            total += self.compute_scalp_bonus(bars_held, realized_pnl)
            total += self.compute_rr_bonus(realized_pnl, risk_amount)
            total += self.compute_cost_penalty(spread_cost, commission, balance)

        # R2: Shaping (always, when position open)
        if unrealized_pnl != 0:
            total += self.compute_shaping_reward(unrealized_pnl, balance)

        # R6: Hold penalty (when position open)
        if bars_held > 0:
            total += self.compute_hold_penalty(bars_held)

        # R7: Overtrade (always)
        total += self.compute_overtrade_penalty(trades_today)

        # R8: DD penalty (always)
        total += self.compute_dd_penalty(current_dd)

        return total
