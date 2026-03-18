"""
ScalForex — Action Gating
Filters agent actions based on confidence threshold.
|confidence| < 0.3 → forced HOLD (prevents overtrade).
"""

from __future__ import annotations

import numpy as np


def apply_action_gating(
    raw_actions: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """Apply confidence gating to raw actions.

    If |action[0]| (direction/confidence) is below threshold,
    the direction component is forced to 0 (HOLD).
    Risk fraction (action[1]) is never modified.

    Args:
        raw_actions: (batch, 2) or (2,) raw actions from policy.
        threshold: Minimum |confidence| to allow trade.

    Returns:
        Gated actions with same shape.
    """
    actions = raw_actions.copy()

    if actions.ndim == 1:
        if abs(actions[0]) < threshold:
            actions[0] = 0.0
        return actions

    # Batch mode
    low_confidence = np.abs(actions[:, 0]) < threshold
    actions[low_confidence, 0] = 0.0
    return actions
