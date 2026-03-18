"""
ScalForex — Feature Builder
29 features: SMC + Volume + Price Action + Time + Multi-TF.
KHÔNG dùng indicator truyền thống (RSI, ATR, Bollinger, MA).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from utils.logger import setup_logger

log = setup_logger("data_engine.feature_builder")

# Session hours (UTC)
SESSIONS = {
    "asian": (0, 8),       # 00:00 - 08:00 UTC
    "european": (8, 15),   # 08:00 - 15:00 UTC
    "us": (15, 22),        # 15:00 - 22:00 UTC
}


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Build 29 features from raw OHLCV data.

    Input DataFrame must have columns: [time, open, high, low, close, volume].

    Returns:
        DataFrame with 29 feature columns appended.
    """
    _validate_input(df)

    df = df.with_columns([
        # === PRICE ACTION (1-6) ===
        _candle_body_ratio(),
        _upper_wick_ratio(),
        _lower_wick_ratio(),
        _pin_bar(),
        _engulfing(),
        _inside_bar(),

        # === VOLUME (7-9) ===
        _relative_volume(),
        _volume_delta(),
        _climax_volume(),

        # === SMC (10-20) ===
        _swing_high(),
        _swing_low(),
    ])

    # BOS/CHoCH need swing points computed first
    df = _compute_bos_choch(df)

    df = df.with_columns([
        # Order Blocks, FVG, Liquidity (14-19) — distance-based
        _order_block_bull(),
        _order_block_bear(),
        _fvg_bull(),
        _fvg_bear(),
        _liquidity_above(),
        _liquidity_below(),

        # Trend direction from swings (20)
        _trend_direction(),

        # === TIME (21-25) ===
        _hour_sin(),
        _hour_cos(),
        _dow_sin(),
        _dow_cos(),
        _session_id(),

        # === PRICE (26-27) ===
        _log_return(),
        _range_pct(),
    ])

    log.info(f"Built {27} features for {len(df)} bars")
    return df


def add_multi_tf_features(
    df_m5: pl.DataFrame,
    h1_trend: pl.Series,
    h4_trend: pl.Series,
) -> pl.DataFrame:
    """Add H1/H4 trend features (28-29) to M5 DataFrame.

    Args:
        df_m5: M5 DataFrame with features already built.
        h1_trend: H1 trend direction series aligned to M5 bars.
        h4_trend: H4 trend direction series aligned to M5 bars.

    Returns:
        DataFrame with h1_trend and h4_trend columns added.
    """
    return df_m5.with_columns([
        h1_trend.alias("h1_trend"),
        h4_trend.alias("h4_trend"),
    ])


# ============================================================
# VALIDATION
# ============================================================

def _validate_input(df: pl.DataFrame) -> None:
    """Validate input DataFrame has required columns."""
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(df) < 20:
        raise ValueError(f"Need at least 20 bars, got {len(df)}")


# ============================================================
# PRICE ACTION FEATURES (1-6)
# ============================================================

def _candle_body_ratio() -> pl.Expr:
    """Body / Range ratio — đo sức mạnh nến."""
    body = (pl.col("close") - pl.col("open")).abs()
    rng = pl.col("high") - pl.col("low")
    return (body / rng.clip(lower_bound=1e-10)).alias("candle_body_ratio")


def _upper_wick_ratio() -> pl.Expr:
    """Upper wick / Range — áp lực bán."""
    close_open_max = pl.max_horizontal("close", "open")
    upper_wick = pl.col("high") - close_open_max
    rng = pl.col("high") - pl.col("low")
    return (upper_wick / rng.clip(lower_bound=1e-10)).alias("upper_wick_ratio")


def _lower_wick_ratio() -> pl.Expr:
    """Lower wick / Range — áp lực mua."""
    close_open_min = pl.min_horizontal("close", "open")
    lower_wick = close_open_min - pl.col("low")
    rng = pl.col("high") - pl.col("low")
    return (lower_wick / rng.clip(lower_bound=1e-10)).alias("lower_wick_ratio")


def _pin_bar() -> pl.Expr:
    """Pin bar detection: wick > 2× body."""
    body = (pl.col("close") - pl.col("open")).abs()
    rng = pl.col("high") - pl.col("low")
    wick = rng - body
    return (wick > 2.0 * body).cast(pl.Int8).alias("pin_bar")


def _engulfing() -> pl.Expr:
    """Engulfing pattern: +1 bullish, -1 bearish, 0 none."""
    curr_body = pl.col("close") - pl.col("open")
    prev_body = curr_body.shift(1)
    curr_range = curr_body.abs()
    prev_range = prev_body.abs()

    bullish = (curr_body > 0) & (prev_body < 0) & (curr_range > prev_range)
    bearish = (curr_body < 0) & (prev_body > 0) & (curr_range > prev_range)

    return (
        pl.when(bullish).then(1)
        .when(bearish).then(-1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("engulfing")
    )


def _inside_bar() -> pl.Expr:
    """Inside bar: current bar within previous bar's range."""
    return (
        (pl.col("high") < pl.col("high").shift(1))
        & (pl.col("low") > pl.col("low").shift(1))
    ).cast(pl.Int8).alias("inside_bar")


# ============================================================
# VOLUME FEATURES (7-9)
# ============================================================

def _relative_volume() -> pl.Expr:
    """Volume / SMA(volume, 20)."""
    avg_vol = pl.col("volume").rolling_mean(window_size=20)
    return (pl.col("volume") / avg_vol.clip(lower_bound=1e-10)).alias("relative_volume")


def _volume_delta() -> pl.Expr:
    """Estimated buy vol - sell vol (approximation using price direction)."""
    direction = pl.when(pl.col("close") > pl.col("open")).then(1.0).otherwise(-1.0)
    return (direction * pl.col("volume")).alias("volume_delta")


def _climax_volume() -> pl.Expr:
    """Binary: volume > 2.5× average."""
    avg_vol = pl.col("volume").rolling_mean(window_size=20)
    return (pl.col("volume") > 2.5 * avg_vol).cast(pl.Int8).alias("climax_volume")


# ============================================================
# SMC FEATURES (10-20)
# ============================================================

def _swing_high() -> pl.Expr:
    """Fractal swing high: high[i] > high[i-1] and high[i] > high[i+1]."""
    return (
        (pl.col("high") > pl.col("high").shift(1))
        & (pl.col("high") > pl.col("high").shift(-1))
    ).cast(pl.Int8).alias("swing_high")


def _swing_low() -> pl.Expr:
    """Fractal swing low: low[i] < low[i-1] and low[i] < low[i+1]."""
    return (
        (pl.col("low") < pl.col("low").shift(1))
        & (pl.col("low") < pl.col("low").shift(-1))
    ).cast(pl.Int8).alias("swing_low")


def _compute_bos_choch(df: pl.DataFrame) -> pl.DataFrame:
    """Compute BOS and CHoCH from swing structure.

    BOS = Break of Structure (trend continuation)
    CHoCH = Change of Character (trend reversal)

    This is computed row-by-row since it depends on tracked swing states.
    Returns DataFrame with 'bos' and 'choch' columns added.
    """
    n = len(df)
    bos = np.zeros(n, dtype=np.int8)
    choch = np.zeros(n, dtype=np.int8)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    swing_highs = df["swing_high"].to_numpy()
    swing_lows = df["swing_low"].to_numpy()

    last_swing_high = highs[0]
    last_swing_low = lows[0]
    trend = 0  # 0=undefined, 1=up, -1=down

    for i in range(2, n):
        # Track latest swing levels
        if swing_highs[i - 1] == 1:
            last_swing_high = highs[i - 1]
        if swing_lows[i - 1] == 1:
            last_swing_low = lows[i - 1]

        # BOS: continuation — price breaks structure in trend direction
        if trend >= 0 and highs[i] > last_swing_high:
            bos[i] = 1
            trend = 1
        elif trend <= 0 and lows[i] < last_swing_low:
            bos[i] = -1
            trend = -1

        # CHoCH: reversal — price breaks structure against trend
        if trend == 1 and lows[i] < last_swing_low:
            choch[i] = -1
            trend = -1
        elif trend == -1 and highs[i] > last_swing_high:
            choch[i] = 1
            trend = 1

    return df.with_columns([
        pl.Series("bos", bos),
        pl.Series("choch", choch),
    ])


def _order_block_bull() -> pl.Expr:
    """Distance to nearest bullish OB (last bearish candle before a bullish move)."""
    is_bearish = pl.col("close") < pl.col("open")
    next_bullish = pl.col("close").shift(-1) > pl.col("open").shift(-1)
    ob_level = pl.when(is_bearish & next_bullish).then(pl.col("low")).otherwise(None)
    return (
        (pl.col("close") - ob_level.forward_fill()) / pl.col("close").clip(lower_bound=1e-10)
    ).alias("order_block_bull")


def _order_block_bear() -> pl.Expr:
    """Distance to nearest bearish OB (last bullish candle before a bearish move)."""
    is_bullish = pl.col("close") > pl.col("open")
    next_bearish = pl.col("close").shift(-1) < pl.col("open").shift(-1)
    ob_level = pl.when(is_bullish & next_bearish).then(pl.col("high")).otherwise(None)
    return (
        (ob_level.forward_fill() - pl.col("close")) / pl.col("close").clip(lower_bound=1e-10)
    ).alias("order_block_bear")


def _fvg_bull() -> pl.Expr:
    """Bullish Fair Value Gap: low[i+1] > high[i-1]."""
    gap = pl.col("low").shift(-1) - pl.col("high").shift(1)
    return (
        pl.when(gap > 0)
        .then(gap / pl.col("close").clip(lower_bound=1e-10))
        .otherwise(0.0)
        .alias("fvg_bull")
    )


def _fvg_bear() -> pl.Expr:
    """Bearish Fair Value Gap: high[i-1] < low[i+1] (inverted)."""
    gap = pl.col("low").shift(1) - pl.col("high").shift(-1)
    return (
        pl.when(gap > 0)
        .then(gap / pl.col("close").clip(lower_bound=1e-10))
        .otherwise(0.0)
        .alias("fvg_bear")
    )


def _liquidity_above() -> pl.Expr:
    """Distance to recent high (liquidity pool above)."""
    rolling_high = pl.col("high").rolling_max(window_size=20)
    return (
        (rolling_high - pl.col("close")) / pl.col("close").clip(lower_bound=1e-10)
    ).alias("liquidity_above")


def _liquidity_below() -> pl.Expr:
    """Distance to recent low (liquidity pool below)."""
    rolling_low = pl.col("low").rolling_min(window_size=20)
    return (
        (pl.col("close") - rolling_low) / pl.col("close").clip(lower_bound=1e-10)
    ).alias("liquidity_below")


def _trend_direction() -> pl.Expr:
    """Trend direction: +1 if higher highs, -1 if lower lows."""
    hh = pl.col("high") > pl.col("high").rolling_max(window_size=10).shift(1)
    ll = pl.col("low") < pl.col("low").rolling_min(window_size=10).shift(1)
    return (
        pl.when(hh & ~ll).then(1)
        .when(ll & ~hh).then(-1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("trend_direction")
    )


# ============================================================
# TIME FEATURES (21-25)
# ============================================================

def _hour_sin() -> pl.Expr:
    """sin(2π × hour / 24)."""
    hour = pl.col("time").dt.hour().cast(pl.Float64)
    return (hour * (2.0 * math.pi / 24.0)).sin().alias("hour_sin")


def _hour_cos() -> pl.Expr:
    """cos(2π × hour / 24)."""
    hour = pl.col("time").dt.hour().cast(pl.Float64)
    return (hour * (2.0 * math.pi / 24.0)).cos().alias("hour_cos")


def _dow_sin() -> pl.Expr:
    """sin(2π × day_of_week / 7) — crypto trades all 7 days."""
    dow = pl.col("time").dt.weekday().cast(pl.Float64)
    return (dow * (2.0 * math.pi / 7.0)).sin().alias("dow_sin")


def _dow_cos() -> pl.Expr:
    """cos(2π × day_of_week / 7)."""
    dow = pl.col("time").dt.weekday().cast(pl.Float64)
    return (dow * (2.0 * math.pi / 7.0)).cos().alias("dow_cos")


def _session_id() -> pl.Expr:
    """Session: 0=Asian, 1=European, 2=US, 3=Off-hours."""
    hour = pl.col("time").dt.hour()
    return (
        pl.when((hour >= 0) & (hour < 8)).then(0)       # Asian
        .when((hour >= 8) & (hour < 15)).then(1)         # European
        .when((hour >= 15) & (hour < 22)).then(2)        # US
        .otherwise(3)                                     # Off-hours
        .cast(pl.Int8)
        .alias("session_id")
    )


# ============================================================
# PRICE FEATURES (26-27)
# ============================================================

def _log_return() -> pl.Expr:
    """log(close / close_prev)."""
    return (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return")


def _range_pct() -> pl.Expr:
    """(high - low) / close — volatility proxy."""
    return (
        (pl.col("high") - pl.col("low")) / pl.col("close").clip(lower_bound=1e-10)
    ).alias("range_pct")


# ============================================================
# FEATURE LIST (for reference)
# ============================================================

FEATURE_COLUMNS = [
    # Price Action (1-6)
    "candle_body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "pin_bar", "engulfing", "inside_bar",
    # Volume (7-9)
    "relative_volume", "volume_delta", "climax_volume",
    # SMC (10-20)
    "swing_high", "swing_low", "bos", "choch",
    "order_block_bull", "order_block_bear",
    "fvg_bull", "fvg_bear",
    "liquidity_above", "liquidity_below",
    "trend_direction",
    # Time (21-25)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_id",
    # Price (26-27)
    "log_return", "range_pct",
    # Multi-TF (28-29) — added via add_multi_tf_features()
    "h1_trend", "h4_trend",
]
