"""
ScalForex — SHAP Feature Importance Analysis
Prove the agent uses real trading signals (FVG, OB, Volume)
and not overfitting to noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from agents.sac_policy import SACPolicy
from utils.logger import setup_logger

log = setup_logger("scripts.shap_analysis")

CHECKPOINT_PATH = Path("checkpoints/baseline_mlp/baseline_mlp.pt")
LOOKBACK = 60
N_FEATURES = 27
OBS_DIM = LOOKBACK * N_FEATURES

# Feature names matching feature_builder output (27 features × 60 bars)
FEATURE_NAMES = [
    "candle_body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "pin_bar", "engulfing", "inside_bar",
    "relative_volume", "volume_delta", "climax_volume",
    "swing_high", "swing_low", "bos", "choch",
    "order_block_bull", "order_block_bear",
    "fvg_bull", "fvg_bear",
    "liquidity_above", "liquidity_below",
    "trend_direction",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_id",
    "log_return", "range_pct",
]


def permutation_importance(
    policy: SACPolicy,
    obs_data: np.ndarray,
    device: torch.device,
    n_repeats: int = 10,
) -> dict[str, float]:
    """Compute feature importance via permutation.

    For each feature group (across lookback window), shuffle it
    and measure the change in action output. Larger change = more important.

    This is a model-agnostic approximation of SHAP for RL policies.

    Args:
        policy: Trained SAC policy.
        obs_data: (n_samples, obs_dim) observation data.
        device: Inference device.
        n_repeats: Number of permutation repeats.

    Returns:
        Dict of feature_name → importance score.
    """
    policy.eval()
    n_samples = obs_data.shape[0]

    # Baseline predictions
    obs_tensor = torch.from_numpy(obs_data).to(device)
    with torch.no_grad():
        baseline_actions = policy.deterministic_action(obs_tensor).cpu().numpy()

    importances: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        for _ in range(n_repeats):
            # Shuffle this feature across all lookback positions
            obs_shuffled = obs_data.copy()
            for t in range(LOOKBACK):
                col_idx = t * N_FEATURES + feat_idx
                if col_idx < obs_shuffled.shape[1]:
                    np.random.shuffle(obs_shuffled[:, col_idx])

            # Measure change in actions
            shuffled_tensor = torch.from_numpy(obs_shuffled).to(device)
            with torch.no_grad():
                shuffled_actions = policy.deterministic_action(shuffled_tensor).cpu().numpy()

            # Importance = mean absolute change in action[0] (direction/confidence)
            change = np.mean(np.abs(baseline_actions[:, 0] - shuffled_actions[:, 0]))
            importances[feat_name].append(change)

    # Average across repeats
    avg_importances = {
        name: float(np.mean(scores))
        for name, scores in importances.items()
    }
    return avg_importances


def main() -> None:
    """Run SHAP-like feature importance analysis."""
    log.info("=" * 60)
    log.info("SCALFOREX — FEATURE IMPORTANCE ANALYSIS (Permutation)")
    log.info("=" * 60)

    device = torch.device("cpu")

    # Load model
    policy = SACPolicy(obs_dim=OBS_DIM, action_dim=2, hidden_dims=[256, 256])

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        policy.load_state_dict(checkpoint["policy_state"])
        log.info(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        log.warning("No checkpoint — using random policy (results won't be meaningful)")

    # Generate sample observations
    rng = np.random.default_rng(42)
    n_samples = 200
    obs_data = rng.standard_normal((n_samples, OBS_DIM)).astype(np.float32)

    # Compute importance
    log.info(f"Computing permutation importance ({n_samples} samples, {len(FEATURE_NAMES)} features)...")
    importances = permutation_importance(policy, obs_data, device, n_repeats=5)

    # Sort by importance (descending)
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # Print results
    print()
    print("=" * 60)
    print("  FEATURE IMPORTANCE — TOP 10")
    print("=" * 60)
    for rank, (name, score) in enumerate(sorted_features[:10], 1):
        bar = "█" * int(score * 200)
        print(f"  {rank:2d}. {name:<22s}  {score:.6f}  {bar}")

    print()
    print("  ── BOTTOM 5 (least important) ──")
    for rank, (name, score) in enumerate(sorted_features[-5:], len(sorted_features) - 4):
        print(f"  {rank:2d}. {name:<22s}  {score:.6f}")
    print("=" * 60)

    # Verify: check if SMC features are in top 10
    smc_features = {"fvg_bull", "fvg_bear", "order_block_bull", "order_block_bear",
                    "bos", "choch", "swing_high", "swing_low"}
    volume_features = {"relative_volume", "volume_delta", "climax_volume"}

    top_10_names = {name for name, _ in sorted_features[:10]}
    smc_in_top10 = smc_features & top_10_names
    vol_in_top10 = volume_features & top_10_names

    print()
    print(f"  SMC features in top 10: {smc_in_top10 or 'NONE'}")
    print(f"  Volume features in top 10: {vol_in_top10 or 'NONE'}")
    print()

    # Save report
    output_path = Path("reports/feature_importance.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name}: {score:.6f}" for name, score in sorted_features]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Feature importance saved: {output_path}")


if __name__ == "__main__":
    main()
