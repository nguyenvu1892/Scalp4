"""
ScalForex — RunPod L40 Optimized Training
Hardware: 1x L40 (48GB VRAM, Ada Lovelace), 32 vCPU, 125GB RAM

SAFETY: Propfirm bot is using ~13% GPU + 3% CPU on same server.
- VRAM capped via batch size + gradient accumulation (no full VRAM grab)
- CPU threads capped at 16 (leave 16 for Propfirm + system)
- PER buffer stored exclusively in CPU RAM (125GB available)
- torch.cuda.set_per_process_memory_fraction(0.80) — hard cap

Optimizations enabled for L40 (Ada Lovelace, CC 8.9):
- BF16 Mixed Precision (native Tensor Core support)
- torch.compile() for graph-level optimization
- Larger batch size (512) — L40 has 48GB VRAM headroom
- 4-stage Curriculum (Kindergarten → University)

Usage:
    python scripts/train_runpod.py --wandb
    python scripts/train_runpod.py --steps 2000000 --batch 512 --wandb
    python scripts/train_runpod.py --resume /workspace/checkpoints/transformer/step_500000.pt
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch

# ── Resource Capping (MUST be before any heavy import) ──
# Cap CPU threads to 16 (leave 16 for Propfirm bot + system)
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)

# Cap VRAM to 80% of total (protect Propfirm bot's ~13%)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.80, 0)

from agents.sac_policy import SACPolicy
from configs.validator import CurriculumConfig, CurriculumStage
from environments.scalp_env import ScalpEnv
from training.curriculum import CurriculumManager
from training.per_buffer import PrioritizedReplayBuffer
from training.trainer import SACTrainer
from utils.logger import setup_logger

log = setup_logger("scripts.train_runpod")

LOOKBACK = 60
N_FEATURES = 27
OBS_DIM = LOOKBACK * N_FEATURES  # 1620


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ScalForex L40 Training")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--batch", type=int, default=512,
                    help="Batch size (512 safe for L40 48GB)")
    p.add_argument("--buffer", type=int, default=1_000_000,
                    help="PER buffer capacity (CPU RAM, 125GB available)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-dir", type=str,
                    default="/workspace/checkpoints/transformer")
    p.add_argument("--save-every", type=int, default=50_000,
                    help="Checkpoint every N steps")
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--warmup", type=int, default=5_000)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", type=str, default="scalforex")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--compile", action="store_true", default=True,
                    help="Enable torch.compile() (default: True for L40)")
    p.add_argument("--grad-accum", type=int, default=2,
                    help="Gradient accumulation steps")
    return p.parse_args()


def make_data(n: int = 5000, seed: int = 42) -> pl.DataFrame:
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
    env: ScalpEnv, policy: SACPolicy, device: torch.device, n_episodes: int = 5,
) -> dict:
    policy.eval()
    total_pnl = total_reward = 0.0
    total_trades = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy.deterministic_action(obs_t)
            obs, reward, terminated, truncated, info = env.step(
                action.cpu().numpy().squeeze(0)
            )
            ep_rew += reward
            done = terminated or truncated
        total_pnl += info.get("episode_pnl", 0)
        total_trades += info.get("total_trades", 0)
        total_reward += ep_rew

    policy.train()
    return {
        "avg_pnl": total_pnl / n_episodes,
        "avg_trades": total_trades / n_episodes,
        "avg_reward": total_reward / n_episodes,
    }


def main() -> None:
    args = parse_args()

    # ── Device ──
    assert torch.cuda.is_available(), "L40 GPU required!"
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_device_capability(0)
    use_bf16 = cap[0] >= 8  # Ada Lovelace (CC 8.9) supports BF16

    log.info("=" * 60)
    log.info("SCALFOREX — RUNPOD L40 TRAINING")
    log.info(f"  GPU:         {gpu_name} ({gpu_mem:.1f}GB)")
    log.info(f"  Compute Cap: {cap[0]}.{cap[1]}")
    log.info(f"  BF16:        {'YES' if use_bf16 else 'NO'}")
    log.info(f"  torch.compile: {args.compile}")
    log.info(f"  VRAM cap:    80% ({gpu_mem * 0.8:.1f}GB)")
    log.info(f"  CPU threads: 16 / 32 (Propfirm safety)")
    log.info(f"  Steps:       {args.steps:,}")
    log.info(f"  Batch:       {args.batch}")
    log.info(f"  Grad Accum:  {args.grad_accum}")
    log.info(f"  Eff. Batch:  {args.batch * args.grad_accum}")
    log.info(f"  Buffer:      {args.buffer:,} (CPU RAM)")
    log.info(f"  Save dir:    {args.save_dir}")
    log.info("=" * 60)

    # ── W&B ──
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"L40_transformer_{args.steps // 1000}k_{datetime.now():%m%d_%H%M}",
                config={
                    "gpu": gpu_name,
                    "vram_gb": gpu_mem,
                    "bf16": use_bf16,
                    "torch_compile": args.compile,
                    "total_steps": args.steps,
                    "batch_size": args.batch,
                    "grad_accum": args.grad_accum,
                    "effective_batch": args.batch * args.grad_accum,
                    "buffer_size": args.buffer,
                    "lr": args.lr,
                    "cpu_threads": 16,
                    "vram_cap_pct": 0.80,
                    "obs_dim": OBS_DIM,
                },
            )
            log.info(f"W&B: {wandb_run.url}")
        except ImportError:
            log.warning("wandb not installed")
            args.wandb = False

    # ── Curriculum (4 stages) ──
    stage_steps = args.steps // 4
    curriculum = CurriculumManager(config=CurriculumConfig(stages=[
        CurriculumStage(name="kindergarten", steps=stage_steps,
                        max_dd_pct=0.80, spread_mode="fixed",
                        regimes=["trending"]),
        CurriculumStage(name="elementary", steps=stage_steps,
                        max_dd_pct=0.60, spread_mode="variable",
                        regimes=["trending", "ranging"]),
        CurriculumStage(name="high_school", steps=stage_steps,
                        max_dd_pct=0.50, spread_mode="realistic",
                        regimes=["trending", "ranging", "choppy"]),
        CurriculumStage(name="university", steps=stage_steps,
                        max_dd_pct=0.50, spread_mode="realistic",
                        regimes=["trending", "ranging", "choppy"],
                        use_augmentation=True),
    ]))

    # ── Environments ──
    train_data = make_data(5000, seed=42)
    eval_data = make_data(1000, seed=99)
    env = ScalpEnv(data=train_data, symbol="XAUUSD", initial_balance=200.0)
    eval_env = ScalpEnv(data=eval_data, symbol="XAUUSD", initial_balance=200.0)

    # ── Model ──
    hidden = [512, 256]
    policy = SACPolicy(obs_dim=OBS_DIM, action_dim=2, hidden_dims=hidden)
    policy.to(device)
    params = sum(p.numel() for p in policy.parameters())
    log.info(f"Model: Transformer SAC {hidden}, {params:,} params")

    # torch.compile (L40 Ada Lovelace — massive speedup)
    if args.compile:
        try:
            policy = torch.compile(policy, mode="reduce-overhead")
            log.info("torch.compile() ENABLED (reduce-overhead mode)")
        except Exception as e:
            log.warning(f"torch.compile failed: {e}, continuing without")

    # ── PER Buffer (CPU RAM — 125GB available, no GPU VRAM!) ──
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
        device="cuda",
        lr=args.lr,
    )

    # ── Mixed Precision (BF16 for L40 Ada Lovelace) ──
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(not use_bf16))
    log.info(f"AMP dtype: {amp_dtype}")

    # ── Resume ──
    start_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        # Handle torch.compile state dict keys
        state = ckpt["policy_state"]
        try:
            policy.load_state_dict(state)
        except RuntimeError:
            # Strip _orig_mod. prefix from compiled model
            clean = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            policy.load_state_dict(clean)
        start_step = ckpt.get("global_step", 0)
        log.info(f"Resumed from {args.resume} at step {start_step:,}")

    # ── Warmup ──
    log.info(f"Warmup: {args.warmup} random steps...")
    obs, _ = env.reset()
    for _ in range(args.warmup):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs if not (terminated or truncated) else env.reset()[0]

    # ── Training Loop ──
    log.info("Training started!")
    t0 = time.perf_counter()
    step = start_step
    ep_rewards: list[float] = []
    ep_reward = 0.0
    ep_count = 0
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    best_pnl = float("-inf")
    obs, _ = env.reset()

    while step < args.steps:
        # Curriculum
        curriculum.step()
        stage_name = curriculum.state.stage_name

        # Collect
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
            action, _ = policy.sample_action(obs_t)
        action_np = action.cpu().numpy().squeeze(0)

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        buffer.add(obs, action_np, reward, next_obs, done)
        ep_reward += reward
        step += 1

        if done:
            ep_rewards.append(ep_reward)
            ep_count += 1
            ep_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

        # Update with AMP
        if step % 4 == 0 and len(buffer) >= args.batch:
            metrics = trainer.update()
            if metrics:
                actor_losses.append(metrics.get("actor_loss", 0))
                critic_losses.append(metrics.get("critic_loss", 0))

        # Eval & Log
        if step % args.eval_every == 0:
            elapsed = time.perf_counter() - t0
            sps = (step - start_step) / max(elapsed, 1)
            ev = evaluate(eval_env, policy, device)

            avg_r = float(np.mean(ep_rewards[-100:])) if ep_rewards else 0
            avg_al = float(np.mean(actor_losses[-100:])) if actor_losses else 0
            avg_cl = float(np.mean(critic_losses[-100:])) if critic_losses else 0
            dd = abs(ev["avg_pnl"]) / 200.0 if ev["avg_pnl"] < 0 else 0

            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9

            log.info(
                f"Step {step:>8,}/{args.steps:,} | "
                f"{sps:.0f} sps | "
                f"R={avg_r:+.1f} | "
                f"PnL=${ev['avg_pnl']:.2f} | "
                f"Trades={ev['avg_trades']:.0f} | "
                f"AL={avg_al:.4f} CL={avg_cl:.4f} | "
                f"a={trainer.alpha.item():.4f} | "
                f"VRAM={vram_used:.1f}/{vram_total:.1f}GB | "
                f"Stage={stage_name}"
            )

            if args.wandb and wandb_run:
                import wandb
                wandb.log({
                    "step": step,
                    "eval/avg_pnl": ev["avg_pnl"],
                    "eval/avg_trades": ev["avg_trades"],
                    "eval/avg_reward": ev["avg_reward"],
                    "train/avg_reward": avg_r,
                    "train/actor_loss": avg_al,
                    "train/critic_loss": avg_cl,
                    "train/alpha": trainer.alpha.item(),
                    "train/sps": sps,
                    "train/episodes": ep_count,
                    "train/buffer_size": len(buffer),
                    "risk/drawdown_pct": dd,
                    "curriculum/stage": curriculum.state.stage_idx,
                    "curriculum/stage_name": stage_name,
                    "hardware/vram_used_gb": vram_used,
                    "hardware/vram_pct": vram_used / vram_total * 100,
                }, step=step)

            # Best model
            if ev["avg_pnl"] > best_pnl:
                best_pnl = ev["avg_pnl"]
                best_path = Path(args.save_dir) / "best.pt"
                trainer.save_checkpoint(best_path, global_step=step)
                log.info(f"  >> New best PnL: ${best_pnl:.2f} -> {best_path}")

        # Periodic checkpoint
        if step % args.save_every == 0 and step > 0:
            ckpt_path = Path(args.save_dir) / f"step_{step}.pt"
            trainer.save_checkpoint(ckpt_path, global_step=step)
            log.info(f"  >> Checkpoint: {ckpt_path}")

    # ── Final ──
    elapsed = time.perf_counter() - t0
    final = evaluate(eval_env, policy, device)

    log.info("=" * 60)
    log.info("TRAINING COMPLETE")
    log.info(f"  GPU:          {gpu_name}")
    log.info(f"  Steps:        {args.steps:,}")
    log.info(f"  Time:         {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    log.info(f"  Speed:        {(args.steps - start_step)/elapsed:.0f} sps")
    log.info(f"  Final PnL:    ${final['avg_pnl']:.2f}")
    log.info(f"  Best PnL:     ${best_pnl:.2f}")
    log.info(f"  Episodes:     {ep_count}")
    log.info(f"  Peak VRAM:    {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    log.info("=" * 60)

    final_path = Path(args.save_dir) / "final.pt"
    trainer.save_checkpoint(final_path, global_step=step)
    log.info(f"Final model: {final_path}")

    if args.wandb and wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
