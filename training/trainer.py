"""
ScalForex — SAC Trainer
Hardware-adapted training loop for Dual Xeon + GTX 750 Ti.
NO mixed precision (Maxwell arch). Small batch. CPU-side PER buffer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.sac_policy import SACPolicy
from training.per_buffer import PrioritizedReplayBuffer
from training.curriculum import CurriculumManager
from configs.validator import load_train_config, load_trading_rules
from utils.logger import setup_logger

log = setup_logger("training.trainer")


class SACTrainer:
    """SAC training loop optimized for weak GPU + strong CPU.

    Strategy:
        - PER buffer on CPU RAM (96GB available)
        - Only mini-batches sent to GPU for backward pass
        - No mixed precision (GTX 750 Ti has no Tensor Cores)
        - 30-40 parallel envs for fast rollout via Dual Xeon

    Args:
        policy: SAC policy network.
        buffer: PER buffer instance.
        curriculum: Curriculum manager.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        batch_size: Mini-batch size (64-128 for 750 Ti).
        gamma: Discount factor.
        tau: Soft update coefficient.
        lr: Learning rate.
        device: Training device.
    """

    def __init__(
        self,
        policy: SACPolicy,
        buffer: PrioritizedReplayBuffer,
        curriculum: Optional[CurriculumManager] = None,
        obs_dim: int = 1620,
        action_dim: int = 2,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        device: str = "auto",
    ) -> None:
        # Device selection — prefer GPU but expect 750 Ti limits
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
                log.info(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
            else:
                self.device = torch.device("cpu")
                log.info("No GPU found — training on CPU")
        else:
            self.device = torch.device(device)

        self.policy = policy.to(self.device)
        self.buffer = buffer
        self.curriculum = curriculum
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # Target network (for stable Q-value estimation)
        self.target_policy = SACPolicy(
            obs_dim=obs_dim, action_dim=action_dim,
        ).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.policy.actor.parameters(), lr=lr,
        )
        self.critic_optimizer = optim.Adam(
            self.policy.critic_net.parameters(), lr=lr,
        )

        # Auto entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Metrics
        self.global_step = 0
        self.update_count = 0
        self._metrics: dict[str, list[float]] = {
            "critic_loss": [], "actor_loss": [], "alpha": [],
            "episode_reward": [], "episode_pnl": [],
        }

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def collect_rollout(
        self,
        env: Any,
        n_steps: int = 1,
    ) -> dict[str, float]:
        """Collect experience from environment.

        Args:
            env: Gymnasium environment (or VecEnv).
            n_steps: Steps to collect.

        Returns:
            Dict with rollout stats.
        """
        total_reward = 0.0
        obs, _ = env.reset()

        for _ in range(n_steps):
            # Get action from policy
            obs_tensor = torch.from_numpy(
                np.array(obs, dtype=np.float32)
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, _ = self.policy.sample_action(obs_tensor)
            action_np = action.cpu().numpy().squeeze(0)

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            self.buffer.add(obs, action_np, reward, next_obs, done)
            total_reward += reward
            self.global_step += 1

            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs

        return {"total_reward": total_reward, "steps": n_steps}

    def update(self) -> dict[str, float]:
        """One SAC update step (critic + actor + alpha).

        Returns:
            Dict with loss values.
        """
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size, device=self.device)

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        weights = batch["weights"]

        # === Critic update ===
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample_action(next_states)
            next_q1, next_q2 = self.target_policy.critic(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * next_q

        q1, q2 = self.policy.critic(states, actions)
        td_errors = (q1 - target_q).detach().cpu().numpy().flatten()

        critic_loss = (
            (weights * F.mse_loss(q1, target_q, reduction="none")).mean()
            + (weights * F.mse_loss(q2, target_q, reduction="none")).mean()
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.critic_net.parameters(), 1.0)
        self.critic_optimizer.step()

        # === Actor update ===
        new_actions, log_probs = self.policy.sample_action(states)
        q1_new, q2_new = self.policy.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # === Alpha update ===
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # === Soft update target network ===
        self._soft_update()

        # Update PER priorities
        self.buffer.update_priorities(batch["tree_indices"], td_errors)

        self.update_count += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
        }

    def _soft_update(self) -> None:
        """Polyak averaging for target network."""
        for target_param, param in zip(
            self.target_policy.parameters(),
            self.policy.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state": self.policy.state_dict(),
            "target_state": self.target_policy.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "alpha_opt": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.data,
            "global_step": self.global_step,
            "update_count": self.update_count,
        }, path)
        log.info(f"Checkpoint saved: {path}")
