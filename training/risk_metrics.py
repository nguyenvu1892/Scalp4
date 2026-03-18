"""
ScalForex — Risk Metrics Calculator
Quant metrics for OOS backtest evaluation.
Gate: Sharpe > 0.8, MaxDD < 40%, WinRate >= 45%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from utils.logger import setup_logger

log = setup_logger("training.risk_metrics")


@dataclass
class TradeRecord:
    """Single trade record for analysis."""
    pnl: float
    entry_price: float
    exit_price: float
    direction: int  # 1=BUY, -1=SELL
    bars_held: int
    entry_step: int
    exit_step: int


@dataclass
class BacktestResult:
    """Complete backtest results with computed metrics."""
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    initial_balance: float = 200.0

    # Computed metrics (filled by compute_metrics)
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_pnl: float = 0.0
    total_trades: int = 0
    avg_bars_held: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    expectancy: float = 0.0


def compute_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252 * 288,  # M5 bars per year
) -> float:
    """Annualized Sharpe Ratio.

    Args:
        returns: Array of per-period returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Trading periods per year (M5 = 288/day).

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2 or np.std(returns) < 1e-10:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as fraction (0.0 to 1.0).

    Args:
        equity_curve: Array of account equity values.

    Returns:
        Max drawdown fraction.
    """
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.maximum(peak, 1e-10)
    return float(np.max(dd))


def compute_win_rate(pnls: np.ndarray) -> float:
    """Win rate as fraction (0.0 to 1.0).

    Args:
        pnls: Array of trade PnLs.

    Returns:
        Fraction of winning trades.
    """
    if len(pnls) == 0:
        return 0.0
    return float(np.sum(pnls > 0) / len(pnls))


def compute_profit_factor(pnls: np.ndarray) -> float:
    """Profit Factor = gross_profit / gross_loss.

    Args:
        pnls: Array of trade PnLs.

    Returns:
        Profit factor (>1 is profitable). Returns inf if no losses.
    """
    gross_profit = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.0
    if gross_loss < 1e-10:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def compute_expectancy(pnls: np.ndarray) -> float:
    """Mathematical expectancy per trade.

    E = (WinRate × AvgWin) - (LossRate × AvgLoss)
    """
    if len(pnls) == 0:
        return 0.0
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = len(wins) / len(pnls)
    loss_rate = len(losses) / len(pnls)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.0
    return win_rate * avg_win - loss_rate * avg_loss


def compute_all_metrics(result: BacktestResult) -> BacktestResult:
    """Compute all risk metrics for a BacktestResult.

    Args:
        result: BacktestResult with trades and equity_curve filled.

    Returns:
        Same BacktestResult with metrics computed.
    """
    pnls = np.array([t.pnl for t in result.trades], dtype=np.float64)
    equity = np.array(result.equity_curve, dtype=np.float64)

    result.total_trades = len(result.trades)
    result.total_pnl = float(np.sum(pnls)) if len(pnls) > 0 else 0.0

    if len(pnls) > 0:
        result.win_rate = compute_win_rate(pnls)
        result.profit_factor = compute_profit_factor(pnls)
        result.expectancy = compute_expectancy(pnls)
        result.best_trade = float(np.max(pnls))
        result.worst_trade = float(np.min(pnls))

        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        result.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        result.avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.0
        result.avg_bars_held = float(
            np.mean([t.bars_held for t in result.trades])
        )

    if len(equity) > 1:
        returns = np.diff(equity) / np.maximum(equity[:-1], 1e-10)
        result.sharpe_ratio = compute_sharpe(returns)
        result.max_drawdown = compute_max_drawdown(equity)

    return result


def format_tearsheet(result: BacktestResult) -> str:
    """Format backtest result as text Quant Tearsheet.

    Args:
        result: Computed BacktestResult.

    Returns:
        Multi-line formatted string.
    """
    gate_sharpe = "✅" if result.sharpe_ratio > 0.8 else "❌"
    gate_dd = "✅" if result.max_drawdown < 0.40 else "❌"
    gate_wr = "✅" if result.win_rate >= 0.45 else "❌"

    lines = [
        "=" * 60,
        "  SCALFOREX — QUANT TEARSHEET (OOS Backtest)",
        "=" * 60,
        "",
        f"  Initial Balance:  ${result.initial_balance:.2f}",
        f"  Final Balance:    ${result.initial_balance + result.total_pnl:.2f}",
        f"  Total PnL:        ${result.total_pnl:.2f}",
        f"  Total Trades:     {result.total_trades}",
        "",
        "  ── GATE KEEPER METRICS ──",
        f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}   (Gate > 0.8)  {gate_sharpe}",
        f"  Max Drawdown:     {result.max_drawdown:.1%}  (Gate < 40%) {gate_dd}",
        f"  Win Rate:         {result.win_rate:.1%}  (Gate >= 45%) {gate_wr}",
        "",
        "  ── TRADE ANALYSIS ──",
        f"  Profit Factor:    {result.profit_factor:.2f}",
        f"  Expectancy:       ${result.expectancy:.4f}/trade",
        f"  Avg Win:          ${result.avg_win:.4f}",
        f"  Avg Loss:         ${result.avg_loss:.4f}",
        f"  Best Trade:       ${result.best_trade:.4f}",
        f"  Worst Trade:      ${result.worst_trade:.4f}",
        f"  Avg Bars Held:    {result.avg_bars_held:.1f}",
        "",
        "  ── GATE VERDICT ──",
    ]

    all_pass = (
        result.sharpe_ratio > 0.8
        and result.max_drawdown < 0.40
        and result.win_rate >= 0.45
    )
    if all_pass:
        lines.append("  🟢 ALL GATES PASSED — Ready for Demo Trading")
    else:
        lines.append("  🔴 GATES NOT PASSED — Continue training before Demo")

    lines.append("=" * 60)
    return "\n".join(lines)
