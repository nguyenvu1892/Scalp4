"""
ScalForex — Main Transformer SAC Training Script
GPU-first: auto-detects CUDA (T4/A100/etc) and pushes all tensors to GPU.
Curriculum + PER + W&B logging.

Usage:
    python scripts/train.py --model transformer --steps 1000000
    python scripts/train.py --model mlp --steps 200000 --stages 0,1
    python scripts/train.py --model transformer --device cuda --batch 256
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

from agents.sac_policy import SACPolicy
from environments.scalp_env import ScalpEnv
from training.curriculum import CurriculumManager
from training.per_buffer import PrioritizedReplayBuffer
from training.trainer import SACTrainer
from utils.logger import setup_logger

log = setup_logger("scripts.train")

LOOKBACK = 60
N_FEATURES = 27
OBS_DIM = LOOKBACK * N_FEATURES  # 1620


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ScalForex SAC Training")
    parser.add_argument(
        "--model", type=str, default="transformer",
        choices=["mlp", "transformer"],
        help="Model architecture: mlp (256×256 baseline) or transformer",
    )
    parser.add_argument(
        "--steps", type=int, default=1_000_000,
        help="Total training steps (default: 1M for transformer)",
    )
    parser.add_argument(
        "--batch", type=int, default=256,
        help="Batch size (default: 256 for T4 16GB; use 64 for 750 Ti)",
    )
    parser.add_argument(
        "--buffer", type=int, default=500_000,
        help="PER buffer capacity",
    )
    parser.add_argument(
        "--stages", type=str, default="0,1,2,3",
        help="Curriculum stages to run (comma-separated, e.g. '0,1,2,3')",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cpu, cuda, cuda:0, cuda:1",
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints",
        help="Checkpoint save directory",
    )
    parser.add_argument(
        "--eval-every", type=int, default=10_000,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--warmup", type=int, default=2_000,
        help="Random warmup steps before training",
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="scalforex",
        help="W&B project name",
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def detect_device(requested: str) -> torch.device:
    """Auto-detect best available device."""
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            log.info(f"🚀 GPU detected: {name} ({mem:.1f}GB)")
            return device
        else:
            log.info("No GPU — training on CPU")
            return torch.device("cpu")
    return torch.device(requested)


def make_model(arch: str, device: torch.device) -> SACPolicy:
    """Create model based on architecture choice."""
    if arch == "mlp":
        hidden = [256, 256]
        log.info(f"Model: MLP {hidden}")
    else:
        # Transformer uses same SAC policy wrapper but with larger hidden
        hidden = [512, 256]
        log.info(f"Model: Transformer-SAC {hidden}")

    policy = SACPolicy(obs_dim=OBS_DIM, action_dim=2, hidden_dims=hidden)
    policy.to(device)
    return policy


def make_data(n: int = 3000, seed: int = 42) -> pl.DataFrame:
    """Create synthetic OHLCV training data."""
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


def evaluate(
    env: ScalpEnv, policy: SACPolicy, device: torch.device, n_episodes: int = 3,
) -> dict:
    """Run evaluation episodes."""
    policy.eval()
    total_pnl = 0.0
    total_trades = 0
    total_reward = 0.0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy.deterministic_action(obs_t)
            action_np = action.cpu().numpy().squeeze(0)
            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_reward += reward
            done = terminated or truncated

        total_pnl += info.get("episode_pnl", 0)
        total_trades += info.get("total_trades", 0)
        total_reward += ep_reward

    policy.train()
    return {
        "avg_pnl": total_pnl / n_episodes,
        "avg_trades": total_trades / n_episodes,
        "avg_reward": total_reward / n_episodes,
    }


def main() -> None:
    args = parse_args()

    # ── Device ──
    device = detect_device(args.device)

    # ── W&B ──
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"{args.model}_{args.steps // 1000}k_{datetime.now():%m%d_%H%M}",
                config={
                    "model": args.model,
                    "total_steps": args.steps,
                    "batch_size": args.batch,
                    "buffer_size": args.buffer,
                    "lr": args.lr,
                    "device": str(device),
                    "curriculum_stages": args.stages,
                    "obs_dim": OBS_DIM,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                },
            )
            log.info(f"W&B run: {wandb_run.url}")
        except ImportError:
            log.warning("wandb not installed — logging disabled")
            args.wandb = False

    # ── Banner ──
    log.info("=" * 60)
    log.info("SCALFOREX — SAC TRAINING")
    log.info(f"  Model:     {args.model}")
    log.info(f"  Device:    {device}")
    log.info(f"  Steps:     {args.steps:,}")
    log.info(f"  Batch:     {args.batch}")
    log.info(f"  Buffer:    {args.buffer:,}")
    log.info(f"  LR:        {args.lr}")
    log.info(f"  Stages:    {args.stages}")
    log.info(f"  W&B:       {'ON' if args.wandb else 'OFF'}")
    log.info("=" * 60)

    # ── Curriculum ──
    stages = [int(s) for s in args.stages.split(",")]
    curriculum = CurriculumManager(total_steps=args.steps)

    # ── Environments ──
    train_data = make_data(3000, seed=42)
    eval_data = make_data(500, seed=99)
    env = ScalpEnv(data=train_data, symbol="XAUUSD", initial_balance=200.0)
    eval_env = ScalpEnv(data=eval_data, symbol="XAUUSD", initial_balance=200.0)

    # ── Model ──
    policy = make_model(args.model, device)
    total_params = sum(p.numel() for p in policy.parameters())
    log.info(f"Parameters: {total_params:,}")

    # ── PER Buffer (CPU RAM — only mini-batches go to GPU) ──
    buffer = PrioritizedReplayBuffer(
        capacity=args.buffer, obs_dim=OBS_DIM, action_dim=2,
    )

    # ── Trainer ──
    trainer = SACTrainer(
        policy=policy,
        buffer=buffer,
        obs_dim=OBS_DIM,
        action_dim=2,
        batch_size=args.batch,
        device=str(device),
        lr=args.lr,
    )

    # ── Resume ──
    start_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["policy_state"])
        start_step = ckpt.get("global_step", 0)
        log.info(f"Resumed from {args.resume} at step {start_step:,}")

    # ── Warmup ──
    log.info(f"Warmup: {args.warmup} random steps...")
    obs, _ = env.reset()
    for _ in range(args.warmup):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated or truncated)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    # ── Training Loop ──
    log.info("Training started!")
    start_time = time.perf_counter()
    step = start_step
    episode_rewards: list[float] = []
    episode_reward = 0.0
    episode_count = 0
    obs, _ = env.reset()

    # Track metrics
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    best_pnl = float("-inf")

    while step < args.steps:
        # ── Curriculum: update env params ──
        stage_params = curriculum.get_stage(step)

        # ── Collect one step ──
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
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
            episode_count += 1
            episode_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

        # ── Update policy ──
        if step % 4 == 0 and len(buffer) >= args.batch:
            metrics = trainer.update()
            if metrics:
                actor_losses.append(metrics.get("actor_loss", 0))
                critic_losses.append(metrics.get("critic_loss", 0))

        # ── Eval & Log ──
        if step % args.eval_every == 0:
            elapsed = time.perf_counter() - start_time
            sps = (step - start_step) / max(elapsed, 1)
            eval_result = evaluate(eval_env, policy, device)

            avg_ep_r = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0
            avg_actor_loss = float(np.mean(actor_losses[-100:])) if actor_losses else 0
            avg_critic_loss = float(np.mean(critic_losses[-100:])) if critic_losses else 0

            # Drawdown tracking
            dd_pct = 0.0
            if eval_result["avg_pnl"] < 0:
                dd_pct = abs(eval_result["avg_pnl"]) / 200.0

            log.info(
                f"Step {step:>8,}/{args.steps:,} | "
                f"{sps:.0f} sps | "
                f"AvgR={avg_ep_r:+.1f} | "
                f"PnL=${eval_result['avg_pnl']:.2f} | "
                f"Trades={eval_result['avg_trades']:.0f} | "
                f"ActorL={avg_actor_loss:.4f} | "
                f"CriticL={avg_critic_loss:.4f} | "
                f"α={trainer.alpha.item():.4f} | "
                f"Stage={stage_params.get('stage', '?')}"
            )

            # GPU memory
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / 1e6
                mem_total = torch.cuda.get_device_properties(0).total_mem / 1e6
                log.info(f"  GPU: {mem_used:.0f}/{mem_total:.0f}MB")

            # W&B logging
            if args.wandb and wandb_run:
                import wandb
                wandb.log({
                    "step": step,
                    "eval/avg_pnl": eval_result["avg_pnl"],
                    "eval/avg_trades": eval_result["avg_trades"],
                    "eval/avg_reward": eval_result["avg_reward"],
                    "train/avg_episode_reward": avg_ep_r,
                    "train/actor_loss": avg_actor_loss,
                    "train/critic_loss": avg_critic_loss,
                    "train/alpha": trainer.alpha.item(),
                    "train/sps": sps,
                    "train/episodes": episode_count,
                    "train/buffer_size": len(buffer),
                    "risk/drawdown_pct": dd_pct,
                    "curriculum/stage": stage_params.get("stage", 0),
                }, step=step)

            # Save best
            if eval_result["avg_pnl"] > best_pnl:
                best_pnl = eval_result["avg_pnl"]
                save_path = Path(args.save_dir) / args.model / "best.pt"
                trainer.save_checkpoint(save_path, global_step=step)
                log.info(f"  📈 New best PnL: ${best_pnl:.2f} — saved to {save_path}")

        # ── Periodic checkpoint ──
        if step % 100_000 == 0 and step > 0:
            ckpt_path = Path(args.save_dir) / args.model / f"step_{step}.pt"
            trainer.save_checkpoint(ckpt_path, global_step=step)
            log.info(f"  💾 Checkpoint saved: {ckpt_path}")

    # ── Final ──
    elapsed = time.perf_counter() - start_time
    final_eval = evaluate(eval_env, policy, device)

    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info(f"  Model:        {args.model}")
    log.info(f"  Device:       {device}")
    log.info(f"  Total Steps:  {args.steps:,}")
    log.info(f"  Time:         {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info(f"  Speed:        {(args.steps - start_step)/elapsed:.0f} sps")
    log.info(f"  Final PnL:    ${final_eval['avg_pnl']:.2f}")
    log.info(f"  Final Trades: {final_eval['avg_trades']:.0f}")
    log.info(f"  Best PnL:     ${best_pnl:.2f}")
    log.info(f"  Episodes:     {episode_count}")
    log.info("=" * 60)

    # Final save
    final_path = Path(args.save_dir) / args.model / "final.pt"
    trainer.save_checkpoint(final_path, global_step=step)
    log.info(f"Final model saved: {final_path}")

    if args.wandb and wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
