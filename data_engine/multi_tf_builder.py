"""
ScalForex — Multi-Timeframe Builder
Align H1/H4 context data to M5 bars.
Rule: Chỉ dùng COMPLETED bars (no look-ahead bias).
"""

from __future__ import annotations

import polars as pl

from utils.logger import setup_logger

log = setup_logger("data_engine.multi_tf_builder")


def align_higher_tf_to_m5(
    df_m5: pl.DataFrame,
    df_higher: pl.DataFrame,
    feature_col: str = "trend_direction",
    output_alias: str | None = None,
) -> pl.Series:
    """Align a higher timeframe feature to M5 bars using as-of join.

    CRITICAL: Uses the LAST COMPLETED higher-TF bar to prevent look-ahead bias.
    Each M5 bar gets the value from the most recent completed H1/H4 bar.

    Args:
        df_m5: M5 DataFrame with 'time' column.
        df_higher: Higher TF DataFrame with 'time' and feature_col.
        feature_col: Column name to align.
        output_alias: Output column name. Defaults to feature_col.

    Returns:
        Polars Series aligned to M5 timestamps.
    """
    if "time" not in df_m5.columns:
        raise ValueError("df_m5 must have 'time' column")
    if "time" not in df_higher.columns:
        raise ValueError("df_higher must have 'time' column")
    if feature_col not in df_higher.columns:
        raise ValueError(f"df_higher missing column: {feature_col}")

    alias = output_alias or feature_col

    # As-of join: for each M5 bar, find the last H1/H4 bar with time <= M5 time
    # This ensures we only use COMPLETED bars (no look-ahead)
    m5_times = df_m5.select("time").with_row_index("idx")
    htf_data = df_higher.select("time", feature_col).sort("time")

    # Join using join_asof (backward strategy = use last completed bar)
    joined = m5_times.join_asof(
        htf_data,
        on="time",
        strategy="backward",
    )

    result = joined[feature_col].alias(alias)
    log.info(
        f"Aligned {feature_col} from higher TF ({len(df_higher)} bars) "
        f"to M5 ({len(df_m5)} bars)"
    )
    return result


def build_multi_tf_context(
    df_m5: pl.DataFrame,
    df_h1: pl.DataFrame,
    df_h4: pl.DataFrame,
) -> pl.DataFrame:
    """Build H1/H4 trend features and add them to M5 DataFrame.

    Args:
        df_m5: M5 DataFrame (must already have features built).
        df_h1: H1 DataFrame (must already have features built).
        df_h4: H4 DataFrame (must already have features built).

    Returns:
        M5 DataFrame with h1_trend and h4_trend columns added.
    """
    h1_trend = align_higher_tf_to_m5(df_m5, df_h1, "trend_direction", "h1_trend")
    h4_trend = align_higher_tf_to_m5(df_m5, df_h4, "trend_direction", "h4_trend")

    result = df_m5.with_columns([h1_trend, h4_trend])

    # Fill nulls (start of data where no H1/H4 bar exists yet)
    result = result.with_columns([
        pl.col("h1_trend").fill_null(0),
        pl.col("h4_trend").fill_null(0),
    ])

    log.info(f"Multi-TF context added: h1_trend, h4_trend → {len(result)} M5 bars")
    return result
