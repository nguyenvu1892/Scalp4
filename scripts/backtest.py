"""
ScalForex — OOS Backtest Script
Run trained model on Out-of-Sample data and generate Quant Tearsheet.
Data used here must NEVER overlap with training data.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from agents.sac_policy import SACPolicy
from environments.scalp_env import ScalpEnv
from training.risk_metrics import (
    BacktestResult,
    TradeRecord,
    compute_all_metrics,
    format_tearsheet,
)
from utils.logger import setup_logger

log = setup_logger("scripts.backtest")

CHECKPOINT_PATH = Path("checkpoints/baseline_mlp/baseline_mlp.pt")
LOOKBACK = 60
N_FEATURES = 27
OBS_DIM = LOOKBACK * N_FEATURES


def make_oos_data(n: int = 1000, seed: int = 777) -> pl.DataFrame:
    """Create OOS synthetic data (different seed from training).

    In production, this would load real historical data from a period
    that was NOT used during training.
    """
    rng = np.random.default_rng(seed)
    base = 2050.0  # Different starting price
    # More realistic: trending + mean-reverting mix
    trend = np.cumsum(rng.standard_normal(n) * 1.5 + 0.01)
    noise = rng.standard_normal(n) * 3.0
    prices = base + trend + noise

    start = datetime(2025, 7, 1)  # Different time period
    return pl.DataFrame({
        "time": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": prices,
        "high": prices + rng.uniform(0, 6, n),
        "low": prices - rng.uniform(0, 6, n),
        "close": prices + rng.standard_normal(n) * 2.5,
        "volume": rng.uniform(200, 12000, n),
    })


def run_backtest(
    env: ScalpEnv,
    policy: SACPolicy,
    device: torch.device,
    initial_balance: float = 200.0,
) -> BacktestResult:
    """Run full backtest episode and collect trade records.

    Args:
        env: ScalpEnv with OOS data.
        policy: Trained SAC policy.
        device: Inference device.
        initial_balance: Starting balance.

    Returns:
        BacktestResult with trades and equity curve.
    """
    policy.eval()
    result = BacktestResult(initial_balance=initial_balance)
    result.equity_curve.append(initial_balance)

    obs, _ = env.reset()
    done = False
    prev_balance = initial_balance
    prev_has_position = False
    position_entry_step = 0

    while not done:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy.deterministic_action(obs_tensor)
        action_np = action.cpu().numpy().squeeze(0)

        obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        current_balance = info["balance"]
        has_position = info["has_position"]

        # Detect trade close (had position → no position, or balance changed)
        if prev_has_position and not has_position:
            pnl = current_balance - prev_balance
            trade = TradeRecord(
                pnl=pnl,
                entry_price=0.0,  # Not tracked in detail here
                exit_price=0.0,
                direction=1,  # Simplified
                bars_held=info["step"] - position_entry_step,
                entry_step=position_entry_step,
                exit_step=info["step"],
            )
            result.trades.append(trade)

        if not prev_has_position and has_position:
            position_entry_step = info["step"]
            prev_balance = current_balance

        prev_has_position = has_position
        result.equity_curve.append(current_balance)

    return result


def main() -> None:
    """Run OOS backtest and print Quant Tearsheet."""
    log.info("=" * 60)
    log.info("SCALFOREX — OUT-OF-SAMPLE BACKTEST")
    log.info("=" * 60)

    device = torch.device("cpu")

    # Load trained model
    policy = SACPolicy(obs_dim=OBS_DIM, action_dim=2, hidden_dims=[256, 256])

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        policy.load_state_dict(checkpoint["policy_state"])
        log.info(f"Loaded checkpoint: {CHECKPOINT_PATH}")
        log.info(f"  Trained for {checkpoint.get('global_step', '?')} steps")
    else:
        log.warning(f"No checkpoint at {CHECKPOINT_PATH} — using random policy")

    # Create OOS environment (completely separate data)
    oos_data = make_oos_data(1000, seed=777)
    env = ScalpEnv(data=oos_data, symbol="XAUUSD", initial_balance=200.0)

    # Run backtest
    log.info("Running OOS backtest...")
    start = time.perf_counter()
    result = run_backtest(env, policy, device)
    elapsed = time.perf_counter() - start
    log.info(f"Backtest completed in {elapsed:.1f}s")

    # Compute metrics
    result = compute_all_metrics(result)

    # Print Tearsheet
    tearsheet = format_tearsheet(result)
    print()
    print(tearsheet)
    print()

    # Save tearsheet
    output_path = Path("reports/oos_tearsheet.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tearsheet, encoding="utf-8")
    log.info(f"Tearsheet saved: {output_path}")


if __name__ == "__main__":
    main()
