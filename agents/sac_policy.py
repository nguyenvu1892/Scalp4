"""
ScalForex — SAC Policy
Soft Actor-Critic with twin critics and tanh-squashed actions.
Action: [direction_confidence, risk_fraction] ∈ [-1, 1]².
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    """SAC Actor — outputs tanh-squashed Gaussian actions.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension (2).
        hidden_dims: Hidden layer sizes.
        log_std_range: (min, max) for log standard deviation.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        hidden_dims: list[int] | None = None,
        log_std_range: tuple[float, float] = (-5.0, 2.0),
    ) -> None:
        super().__init__()

        hidden_dims = hidden_dims or [256, 256]
        self.log_std_min, self.log_std_max = log_std_range

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim

        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

    def forward(
        self, obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log_std for action distribution.

        Args:
            obs: (batch, obs_dim)

        Returns:
            (mean, log_std) each (batch, action_dim)
        """
        h = self.net(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class TwinCritic(nn.Module):
    """SAC Twin Q-Network.

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension (2).
        hidden_dims: Hidden layer sizes.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        hidden_dims = hidden_dims or [256, 256]

        def _build_q() -> nn.Sequential:
            layers: list[nn.Module] = []
            in_dim = obs_dim + action_dim
            for h_dim in hidden_dims:
                layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, 1))
            return nn.Sequential(*layers)

        self.q1 = _build_q()
        self.q2 = _build_q()

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute twin Q-values.

        Args:
            obs: (batch, obs_dim)
            action: (batch, action_dim)

        Returns:
            (q1, q2) each (batch, 1)
        """
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class SACPolicy(nn.Module):
    """Soft Actor-Critic policy with twin critics.

    Actions are tanh-squashed to [-1, 1].

    Args:
        obs_dim: Observation dimension.
        action_dim: Action dimension (2).
        hidden_dims: Hidden layer sizes for actor and critic.
        log_std_range: (min, max) for log standard deviation.
    """

    def __init__(
        self,
        obs_dim: int = 64,
        action_dim: int = 2,
        hidden_dims: list[int] | None = None,
        log_std_range: tuple[float, float] = (-5.0, 2.0),
    ) -> None:
        super().__init__()

        hidden_dims = hidden_dims or [256, 256]

        self.actor = Actor(obs_dim, action_dim, hidden_dims, log_std_range)
        self.critic_net = TwinCritic(obs_dim, action_dim, hidden_dims)

    def sample_action(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy with reparameterization.

        Args:
            obs: (batch, obs_dim)

        Returns:
            (action, log_prob) — action in [-1, 1], log_prob (batch, 1).
        """
        mean, log_std = self.actor(obs)
        std = log_std.exp()

        dist = Normal(mean, std)
        # Reparameterization trick
        x_t = dist.rsample()
        action = torch.tanh(x_t)

        # Log prob with tanh correction
        log_prob = dist.log_prob(x_t)
        # Enforce valid log operations
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic_action(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Get deterministic action (for evaluation).

        Args:
            obs: (batch, obs_dim)

        Returns:
            (batch, action_dim) — tanh(mean)
        """
        mean, _ = self.actor(obs)
        return torch.tanh(mean)

    def critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute twin Q-values.

        Args:
            obs: (batch, obs_dim)
            action: (batch, action_dim)

        Returns:
            (q1, q2) each (batch, 1)
        """
        return self.critic_net(obs, action)
