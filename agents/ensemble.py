"""
ScalForex — Ensemble Agent
2/3 majority voting across multiple SAC models.
Only trades when >= 2/3 agree on direction AND confidence > 0.3.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from agents.action_gating import apply_action_gating
from agents.sac_policy import SACPolicy
from utils.logger import setup_logger

log = setup_logger("agents.ensemble")


class EnsembleAgent:
    """Ensemble of N SAC policies with majority voting.

    Trade execution requires >= 2/3 models to agree on:
      1. Direction (BUY vs SELL)
      2. Confidence > threshold

    Args:
        models: List of SAC policy networks.
        threshold: Confidence threshold for trade signal.
        device: Inference device.
    """

    def __init__(
        self,
        models: list[SACPolicy],
        threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models")

        self.models = models
        self.n_models = len(models)
        self.threshold = threshold
        self.device = torch.device(device)
        self.quorum = (self.n_models // 2) + 1  # Majority (2/3 for 3 models)

        # Move all models to device and eval mode
        for m in self.models:
            m.to(self.device)
            m.eval()

        log.info(
            f"Ensemble: {self.n_models} models, "
            f"quorum={self.quorum}, threshold={self.threshold}"
        )

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Get ensemble action via majority voting.

        Args:
            obs: Observation array (1D or 2D).
            deterministic: Use deterministic actions (no sampling).

        Returns:
            Action array [direction_confidence, risk_fraction].
            Direction is zeroed (HOLD) if no quorum.
        """
        single = obs.ndim == 1
        if single:
            obs = obs[np.newaxis, :]  # (1, obs_dim)

        obs_tensor = torch.from_numpy(obs.astype(np.float32)).to(self.device)

        # Collect actions from all models
        all_actions: list[np.ndarray] = []
        with torch.no_grad():
            for model in self.models:
                if deterministic:
                    action = model.deterministic_action(obs_tensor)
                else:
                    action, _ = model.sample_action(obs_tensor)
                all_actions.append(action.cpu().numpy())

        # Stack: (n_models, batch, 2)
        actions_stack = np.stack(all_actions, axis=0)

        # Voting per batch element
        batch_size = obs.shape[0]
        final_actions = np.zeros((batch_size, 2), dtype=np.float32)

        for b in range(batch_size):
            final_actions[b] = self._vote_single(actions_stack[:, b, :])

        # Apply confidence gating
        final_actions = apply_action_gating(final_actions, self.threshold)

        return final_actions.squeeze(0) if single else final_actions

    def _vote_single(self, actions: np.ndarray) -> np.ndarray:
        """Majority vote for a single observation.

        Args:
            actions: (n_models, 2) — actions from each model.

        Returns:
            (2,) — voted action.
        """
        confidences = actions[:, 0]
        risk_fractions = actions[:, 1]

        # Count directional votes (only from confident models)
        buy_votes = np.sum(confidences > self.threshold)
        sell_votes = np.sum(confidences < -self.threshold)
        hold_votes = self.n_models - buy_votes - sell_votes

        # Average risk fraction from agreeing models
        avg_risk = float(np.mean(risk_fractions))

        # Check quorum
        if buy_votes >= self.quorum:
            # Average confidence of BUY voters
            buy_mask = confidences > self.threshold
            avg_conf = float(np.mean(confidences[buy_mask]))
            return np.array([avg_conf, avg_risk], dtype=np.float32)

        elif sell_votes >= self.quorum:
            # Average confidence of SELL voters
            sell_mask = confidences < -self.threshold
            avg_conf = float(np.mean(confidences[sell_mask]))
            return np.array([avg_conf, avg_risk], dtype=np.float32)

        else:
            # No quorum → HOLD
            return np.array([0.0, avg_risk], dtype=np.float32)

    def get_vote_detail(
        self,
        obs: np.ndarray,
    ) -> dict[str, object]:
        """Get detailed voting breakdown for analysis.

        Args:
            obs: Single observation (1D).

        Returns:
            Dict with per-model actions and vote counts.
        """
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]

        obs_tensor = torch.from_numpy(obs.astype(np.float32)).to(self.device)

        per_model: list[dict[str, float]] = []
        with torch.no_grad():
            for i, model in enumerate(self.models):
                action = model.deterministic_action(obs_tensor)
                a = action.cpu().numpy().squeeze(0)
                direction = "BUY" if a[0] > self.threshold else (
                    "SELL" if a[0] < -self.threshold else "HOLD"
                )
                per_model.append({
                    "model_id": i,
                    "confidence": float(a[0]),
                    "risk_fraction": float(a[1]),
                    "direction": direction,
                })

        buy_count = sum(1 for m in per_model if m["direction"] == "BUY")
        sell_count = sum(1 for m in per_model if m["direction"] == "SELL")
        hold_count = sum(1 for m in per_model if m["direction"] == "HOLD")

        final = self.predict(obs.squeeze(0))

        return {
            "per_model": per_model,
            "buy_votes": buy_count,
            "sell_votes": sell_count,
            "hold_votes": hold_count,
            "quorum": self.quorum,
            "final_action": final,
            "consensus": buy_count >= self.quorum or sell_count >= self.quorum,
        }
