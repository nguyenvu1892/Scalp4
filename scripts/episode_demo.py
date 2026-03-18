"""Sprint 2 — Random episode demo for Tech Lead review."""
import numpy as np
import polars as pl
from datetime import datetime, timedelta

# Create data
rng = np.random.default_rng(42)
n = 300
base = 2000.0
prices = base + np.cumsum(rng.standard_normal(n) * 2.0)
start = datetime(2025, 1, 1, 0, 0)
data = pl.DataFrame({
    "time": [start + timedelta(minutes=5*i) for i in range(n)],
    "open": prices,
    "high": prices + rng.uniform(0, 5, n),
    "low": prices - rng.uniform(0, 5, n),
    "close": prices + rng.standard_normal(n) * 2.0,
    "volume": rng.uniform(100, 10000, n),
})

from environments.scalp_env import ScalpEnv
env = ScalpEnv(data=data, symbol="XAUUSD", initial_balance=200.0)
obs, info = env.reset(seed=42)
print("=== RANDOM EPISODE ===")
print("Start: Balance=$200.00")
total_r = 0.0
for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_r += reward
    if step % 30 == 0 or terminated or truncated:
        print(f"  Step {step:3d}: Balance=${info['balance']:.2f}  "
              f"DD={info['drawdown']:.1%}  Trades={info['total_trades']}  "
              f"Reward={reward:+.4f}  Position={'YES' if info['has_position'] else 'NO'}")
    if terminated:
        print(f"  >>> TERMINATED (DD={info['drawdown']:.1%})")
        break
    if truncated:
        print("  >>> TRUNCATED (end of data)")
        break

print()
print("=== EPISODE SUMMARY ===")
print(f"Final Balance: ${info['balance']:.2f}")
print(f"Total PnL:     ${info['episode_pnl']:.2f}")
print(f"Max DD:        {info['drawdown']:.1%}")
print(f"Total Trades:  {info['total_trades']}")
print(f"Total Reward:  {total_r:+.4f}")
print(f"Steps:         {step+1}")
