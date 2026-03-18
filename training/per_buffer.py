"""
ScalForex — Prioritized Experience Replay Buffer
Stored in CPU RAM (96GB available). Only mini-batches are sent to GPU.
Optimized for Dual Xeon + weak GPU (GTX 750 Ti).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from utils.logger import setup_logger

log = setup_logger("training.per_buffer")


@dataclass
class Transition:
    """Single transition stored in buffer."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class SumTree:
    """Binary sum tree for O(log n) priority sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data_idx: int) -> None:
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> tuple[int, float, int]:
        """Get (tree_idx, priority, data_idx) for a given cumulative sum."""
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], data_idx


class PrioritizedReplayBuffer:
    """PER buffer stored entirely in CPU RAM.

    With 96GB RAM, can comfortably hold 1M+ transitions.
    Only mini-batches are transferred to GPU for backward pass.

    Args:
        capacity: Max transitions to store.
        alpha: Priority exponent (0=uniform, 1=full prioritization).
        beta_start: Initial importance sampling correction.
        beta_frames: Frames to anneal beta to 1.0.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        obs_dim: int = 1620,
        action_dim: int = 2,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.min_priority = 1e-6

        # Pre-allocate numpy arrays on CPU (memory efficient)
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

        log.info(
            f"PER Buffer: capacity={capacity:,}, "
            f"obs_dim={obs_dim}, action_dim={action_dim}, "
            f"~{capacity * obs_dim * 4 * 2 / 1e9:.1f}GB RAM"
        )

    @property
    def beta(self) -> float:
        """Current importance sampling beta (annealed to 1.0)."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition with max priority."""
        idx = self._ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        priority = self.max_priority ** self.alpha
        self.tree.add(priority, idx)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Sample prioritized batch and move to device (GPU).

        Args:
            batch_size: Number of transitions to sample.
            device: Target device for tensors.

        Returns:
            Dict with states, actions, rewards, next_states, dones,
            weights (IS correction), tree_indices (for priority update).
        """
        self.frame += 1
        beta = self.beta

        indices = np.zeros(batch_size, dtype=np.int64)
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            tree_idx, priority, data_idx = self.tree.get(s)
            data_idx = np.clip(data_idx, 0, self._size - 1)
            tree_indices[i] = tree_idx
            indices[i] = data_idx
            priorities[i] = max(priority, self.min_priority)

        # Importance sampling weights
        probs = priorities / (self.tree.total() + 1e-10)
        weights = (self._size * probs) ** (-beta)
        weights /= weights.max()

        # Only transfer mini-batch to GPU (not entire buffer)
        batch = {
            "states": torch.from_numpy(self.states[indices]).to(device),
            "actions": torch.from_numpy(self.actions[indices]).to(device),
            "rewards": torch.from_numpy(self.rewards[indices]).unsqueeze(1).to(device),
            "next_states": torch.from_numpy(self.next_states[indices]).to(device),
            "dones": torch.from_numpy(self.dones[indices]).unsqueeze(1).to(device),
            "weights": torch.from_numpy(weights.astype(np.float32)).unsqueeze(1).to(device),
            "tree_indices": tree_indices,
        }
        return batch

    def update_priorities(
        self,
        tree_indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """Update priorities based on TD errors."""
        for idx, td_error in zip(tree_indices, td_errors):
            priority = (abs(td_error) + self.min_priority) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(idx), priority)

    def __len__(self) -> int:
        return self._size
