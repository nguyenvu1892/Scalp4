"""
ScalForex — Baseline MLP Training Script (T3.7 + S4.2)
Train a simple MLP policy as benchmark before Transformer.
Hardware: Dual Xeon 56t + 96GB RAM + GTX 750 Ti (no AMP).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from agents.sac_policy import SACPolicy
from configs.validator import load_train_config
from environments.scalp_env import ScalpEnv
from training.per_buffer import PrioritizedReplayBuffer
from training.trainer import SACTrainer
from utils.logger import setup_logger

log = setup_logger("scripts.train_baseline")

# ============================================================
# CONFIG
# ============================================================
TOTAL_STEPS = 200_000
BATCH_SIZE = 64           # Small for 750 Ti (2GB VRAM)
BUFFER_SIZE = 200_000     # ~200K for baseline (not 1M)
WARMUP_STEPS = 1_000      # Collect random before training
UPDATE_EVERY = 4          # Update every N env steps
EVAL_EVERY = 10_000       # Eval frequency
SAVE_DIR = Path("checkpoints/baseline_mlp")
LOOKBACK = 60
N_FEATURES = 27
OBS_DIM = LOOKBACK * N_FEATURES  # 1620


def make_synthetic_data(n: int = 2000, seed: int = 42) -> pl.DataFrame:
    """Create synthetic OHLCV data for training."""
    rng = np.random.default_rng(seed)
    base = 2000.0
    prices = base + np.cumsum(rng.standard_normal(n) * 2.0)
    start = datetime(2025, 1, 1)
    return pl.DataFrame({
        "time": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": prices,
        "high": prices + rng.uniform(0, 5, n),
        "low": prices - rng.uniform(0, 5, n),
        "close": prices + rng.standard_normal(n) * 2.0,
        "volume": rng.uniform(100, 10000, n),
    })


def evaluate(env: ScalpEnv, policy: SACPolicy, device: torch.device, n_episodes: int = 3) -> dict:
    """Run evaluation episodes."""
    policy.eval()
    total_pnl = 0.0
    total_trades = 0
    total_steps = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy.deterministic_action(obs_t)
            action_np = action.cpu().numpy().squeeze(0)
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            total_steps += 1

        total_pnl += info.get("episode_pnl", 0)
        total_trades += info.get("total_trades", 0)

    policy.train()
    return {
        "avg_pnl": total_pnl / n_episodes,
        "avg_trades": total_trades / n_episodes,
        "avg_steps": total_steps / n_episodes,
    }


def main() -> None:
    """Run baseline MLP training."""
    log.info("=" * 60)
    log.info("BASELINE MLP TRAINING — ScalForex")
    log.info(f"Steps: {TOTAL_STEPS:,} | Batch: {BATCH_SIZE} | Buffer: {BUFFER_SIZE:,}")
    log.info("=" * 60)

    # Hardware check
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        log.info(f"GPU: {gpu} ({mem:.1f}GB) — NO mixed precision (Maxwell)")
    else:
        log.info("No GPU — training on CPU (Dual Xeon)")

    # Create env
    data = make_synthetic_data(2000)
    env = ScalpEnv(data=data, symbol="XAUUSD", initial_balance=200.0)
    eval_env = ScalpEnv(data=make_synthetic_data(500, seed=99), symbol="XAUUSD", initial_balance=200.0)

    # Create policy (MLP baseline — no Transformer)
    policy = SACPolicy(obs_dim=OBS_DIM, action_dim=2, hidden_dims=[256, 256])

    # Create PER buffer (CPU RAM)
    buffer = PrioritizedReplayBuffer(
        capacity=BUFFER_SIZE, obs_dim=OBS_DIM, action_dim=2,
    )

    # Create trainer
    trainer = SACTrainer(
        policy=policy,
        buffer=buffer,
        obs_dim=OBS_DIM,
        action_dim=2,
        batch_size=BATCH_SIZE,
        device="auto",
    )

    # === Warmup: random rollout ===
    log.info(f"Warmup: collecting {WARMUP_STEPS} random transitions...")
    obs, _ = env.reset()
    for _ in range(WARMUP_STEPS):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated or truncated)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    # === Training loop ===
    log.info("Starting training...")
    start_time = time.perf_counter()
    step = 0
    episode_rewards: list[float] = []
    episode_reward = 0.0
    obs, _ = env.reset()

    while step < TOTAL_STEPS:
        # Collect one step
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(trainer.device)
        with torch.no_grad():
            action, _ = policy.sample_action(obs_tensor)
        action_np = action.cpu().numpy().squeeze(0)

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        buffer.add(obs, action_np, reward, next_obs, done)
        episode_reward += reward
        step += 1

        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

        # Update
        if step % UPDATE_EVERY == 0 and len(buffer) >= BATCH_SIZE:
            metrics = trainer.update()

        # Eval & log
        if step % EVAL_EVERY == 0:
            elapsed = time.perf_counter() - start_time
            sps = step / max(elapsed, 1)
            eval_result = evaluate(eval_env, policy, trainer.device)

            avg_ep_r = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            log.info(
                f"Step {step:>7,}/{TOTAL_STEPS:,} | "
                f"{sps:.0f} sps | "
                f"AvgR={avg_ep_r:+.2f} | "
                f"PnL=${eval_result['avg_pnl']:.2f} | "
                f"Trades={eval_result['avg_trades']:.0f} | "
                f"Alpha={trainer.alpha.item():.4f}"
            )
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1e6
                mem_total = torch.cuda.get_device_properties(0).total_mem / 1e6
                log.info(f"  GPU Mem: {mem_used:.0f}/{mem_total:.0f}MB")

    # === Final results ===
    elapsed = time.perf_counter() - start_time
    final_eval = evaluate(eval_env, policy, trainer.device)

    log.info("=" * 60)
    log.info("BASELINE MLP TRAINING COMPLETE")
    log.info(f"Total Steps:    {TOTAL_STEPS:,}")
    log.info(f"Total Time:     {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info(f"Speed:          {TOTAL_STEPS/elapsed:.0f} steps/sec")
    log.info(f"Final Avg PnL:  ${final_eval['avg_pnl']:.2f}")
    log.info(f"Final Trades:   {final_eval['avg_trades']:.0f}")
    log.info(f"Episodes:       {len(episode_rewards)}")
    log.info("=" * 60)

    # Save checkpoint
    trainer.save_checkpoint(SAVE_DIR / "baseline_mlp.pt")
    log.info(f"Model saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
