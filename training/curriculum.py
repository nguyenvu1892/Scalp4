"""
ScalForex — Curriculum Learning
4-stage training progression (Kindergarten → University).
Auto-adjusts env params based on global_step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from configs.validator import CurriculumConfig, CurriculumStage
from utils.logger import setup_logger

log = setup_logger("training.curriculum")


@dataclass
class CurriculumState:
    """Current curriculum state."""
    stage_idx: int = 0
    stage_name: str = "kindergarten"
    global_step: int = 0
    steps_in_stage: int = 0


class CurriculumManager:
    """Manages 4-stage curriculum progression.

    Stages from train_config.yaml:
        1. Kindergarten: fixed spread, trending only, 80% DD limit
        2. Elementary: variable spread, trending+ranging, 60% DD
        3. High School: realistic spread, all regimes, 50% DD
        4. University: realistic + augmentation, all regimes, 50% DD

    Args:
        config: CurriculumConfig from train_config.yaml.
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self.stages = config.stages
        self.state = CurriculumState()
        self._cumulative_steps = self._build_step_boundaries()
        log.info(
            f"Curriculum: {len(self.stages)} stages, "
            f"boundaries={self._cumulative_steps}"
        )

    def _build_step_boundaries(self) -> list[int]:
        """Build cumulative step boundaries for each stage."""
        boundaries: list[int] = []
        total = 0
        for stage in self.stages:
            total += stage.steps
            boundaries.append(total)
        return boundaries

    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.state.stage_idx]

    def get_env_params(self) -> dict[str, Any]:
        """Get environment parameters for current stage.

        Returns:
            Dict with spread_mode, max_dd_pct, regimes, use_augmentation.
        """
        stage = self.current_stage
        return {
            "spread_mode": stage.spread_mode,
            "max_dd_pct": stage.max_dd_pct,
            "regimes": stage.regimes,
            "use_augmentation": stage.use_augmentation,
        }

    def step(self, n_steps: int = 1) -> bool:
        """Advance global step counter. Returns True if stage changed.

        Args:
            n_steps: Number of steps to advance.

        Returns:
            True if curriculum stage advanced.
        """
        self.state.global_step += n_steps
        self.state.steps_in_stage += n_steps

        # Check if we should advance to next stage
        old_idx = self.state.stage_idx
        for i, boundary in enumerate(self._cumulative_steps):
            if self.state.global_step < boundary:
                self.state.stage_idx = i
                break
        else:
            self.state.stage_idx = len(self.stages) - 1

        if self.state.stage_idx != old_idx:
            self.state.stage_name = self.current_stage.name
            self.state.steps_in_stage = 0
            log.info(
                f"CURRICULUM ADVANCE: Stage {old_idx} → {self.state.stage_idx} "
                f"({self.state.stage_name}) at step {self.state.global_step:,}"
            )
            return True
        return False

    def is_complete(self) -> bool:
        """Check if all curriculum stages are complete."""
        return self.state.global_step >= self._cumulative_steps[-1]
