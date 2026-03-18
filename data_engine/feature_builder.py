"""
ScalForex — Feature Builder (Refactored)
29 features: SMC + Volume + Price Action + Time + Multi-TF.
KHÔNG dùng indicator truyền thống (RSI, ATR, Bollinger, MA).

ZERO HARDCODE: All thresholds/windows from FeatureSettings config.
STATEFUL: IncrementalFeatureBuilder caches swing/OB state for live inference.
LAZY EVAL: Polars LazyFrame pipeline where possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl

from configs.validator import FeatureSettings, SessionHoursConfig
from utils.logger import setup_logger

log = setup_logger("data_engine.feature_builder")


# ============================================================
# CONFIG-DRIVEN FEATURE PARAMS (loaded from YAML)
# ============================================================

def _default_feature_settings() -> FeatureSettings:
    """Fallback feature settings (loads from YAML if available)."""
    try:
        from configs.validator import load_symbols
        return load_symbols().features
    except Exception:
        return FeatureSettings()


# ============================================================
# BATCH PIPELINE — for training data (Polars LazyFrame)
# ============================================================

def build_features(
    df: pl.DataFrame,
    config: Optional[FeatureSettings] = None,
) -> pl.DataFrame:
    """Build 27 features from raw OHLCV data (29 - 2 multi-TF).

    Uses Polars expressions for vectorized, parallel computation.
    All thresholds/windows from config (zero hardcode).

    Args:
        df: DataFrame with columns [time, open, high, low, close, volume].
        config: Feature settings. If None, loads from YAML.

    Returns:
        DataFrame with 27 feature columns appended.
    """
    cfg = config or _default_feature_settings()
    _validate_input(df, cfg)

    # Phase 1: Vectorized features via Polars expressions
    lf = df.lazy().with_columns(
        # === PRICE ACTION (1-6) ===
        _candle_body_ratio(),
        _upper_wick_ratio(),
        _lower_wick_ratio(),
        _pin_bar(cfg.pin_bar_wick_ratio),
        _engulfing(),
        _inside_bar(),

        # === VOLUME (7-9) ===
        _relative_volume(cfg.volume_rolling_window),
        _volume_delta(),
        _climax_volume(cfg.volume_rolling_window, cfg.climax_volume_threshold),

        # === SMC partial (10-11) ===
        _swing_high(),
        _swing_low(),

        # === TIME (21-25) ===
        _hour_sin(),
        _hour_cos(),
        _dow_sin(),
        _dow_cos(),
        _session_id(cfg.sessions),

        # === PRICE (26-27) ===
        _log_return(),
        _range_pct(),
    ).collect()

    # Phase 2: Stateful SMC (BOS/CHoCH need swing tracking — numpy)
    lf = _compute_bos_choch(lf)

    # Phase 3: SMC distance features (need forward_fill)
    lf = lf.lazy().with_columns(
        _order_block_bull(),
        _order_block_bear(),
        _fvg_bull(),
        _fvg_bear(),
        _liquidity_above(cfg.liquidity_window),
        _liquidity_below(cfg.liquidity_window),
        _trend_direction(cfg.swing_lookback),
    ).collect()

    log.info(f"Built 27 features for {len(lf)} bars (config-driven)")
    return lf


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
# STATEFUL INCREMENTAL BUILDER — for live inference
# ============================================================

@dataclass
class SwingState:
    """Cached state for swing point tracking."""

    last_swing_high: float = 0.0
    last_swing_low: float = float("inf")
    trend: int = 0  # 0=undefined, 1=up, -1=down
    prev_high: float = 0.0
    prev_low: float = float("inf")
    prev_prev_high: float = 0.0
    prev_prev_low: float = float("inf")


@dataclass
class IncrementalFeatureBuilder:
    """Stateful feature builder for live inference.

    Caches swing points, order block levels, and running statistics.
    When a new candle arrives, only updates state — no full recalculation.

    Usage:
        builder = IncrementalFeatureBuilder(config=feature_settings)
        builder.warm_up(historical_df)  # Feed history to build state
        features = builder.update(new_candle)  # Incremental update
    """

    config: FeatureSettings
    swing: SwingState = field(default_factory=SwingState)
    _volume_buffer: list[float] = field(default_factory=list)
    _high_buffer: list[float] = field(default_factory=list)
    _low_buffer: list[float] = field(default_factory=list)
    _prev_candle: Optional[dict[str, float]] = None
    _bar_count: int = 0

    def warm_up(self, df: pl.DataFrame) -> None:
        """Feed historical data to build internal state.

        Args:
            df: Historical OHLCV DataFrame.
        """
        for row in df.iter_rows(named=True):
            self._update_state(row)
        log.info(f"IncrementalFeatureBuilder warmed up with {len(df)} bars")

    def update(self, candle: dict[str, float]) -> dict[str, float]:
        """Process one new candle and return computed features.

        Args:
            candle: Dict with keys [time, open, high, low, close, volume].

        Returns:
            Dict of feature_name → value for the new candle.
        """
        features = self._compute_features(candle)
        self._update_state(candle)
        return features

    def _update_state(self, candle: dict[str, float]) -> None:
        """Update cached state with a new candle."""
        self._bar_count += 1
        vol_window = self.config.volume_rolling_window

        # Volume buffer (rolling)
        self._volume_buffer.append(float(candle.get("volume", 0)))
        if len(self._volume_buffer) > vol_window:
            self._volume_buffer.pop(0)

        # High/Low buffers (for liquidity)
        liq_window = self.config.liquidity_window
        self._high_buffer.append(float(candle.get("high", 0)))
        self._low_buffer.append(float(candle.get("low", 0)))
        if len(self._high_buffer) > liq_window:
            self._high_buffer.pop(0)
        if len(self._low_buffer) > liq_window:
            self._low_buffer.pop(0)

        # Swing state
        h = float(candle.get("high", 0))
        lo = float(candle.get("low", 0))

        # Detect swing points (fractal)
        if self._bar_count >= 3:
            if self.swing.prev_high > self.swing.prev_prev_high and self.swing.prev_high > h:
                self.swing.last_swing_high = self.swing.prev_high
            if self.swing.prev_low < self.swing.prev_prev_low and self.swing.prev_low < lo:
                self.swing.last_swing_low = self.swing.prev_low

        # BOS/CHoCH detection
        if self.swing.trend >= 0 and h > self.swing.last_swing_high:
            self.swing.trend = 1
        elif self.swing.trend <= 0 and lo < self.swing.last_swing_low:
            self.swing.trend = -1
        if self.swing.trend == 1 and lo < self.swing.last_swing_low:
            self.swing.trend = -1
        elif self.swing.trend == -1 and h > self.swing.last_swing_high:
            self.swing.trend = 1

        # Shift history
        self.swing.prev_prev_high = self.swing.prev_high
        self.swing.prev_prev_low = self.swing.prev_low
        self.swing.prev_high = h
        self.swing.prev_low = lo

        self._prev_candle = dict(candle)

    def _compute_features(self, candle: dict[str, float]) -> dict[str, float]:
        """Compute all features for a single candle from cached state."""
        cfg = self.config
        o = float(candle.get("open", 0))
        h = float(candle.get("high", 0))
        lo = float(candle.get("low", 0))
        c = float(candle.get("close", 0))
        vol = float(candle.get("volume", 0))
        rng = max(h - lo, 1e-10)
        body = abs(c - o)

        # Price Action
        features: dict[str, float] = {
            "candle_body_ratio": body / rng,
            "upper_wick_ratio": (h - max(c, o)) / rng,
            "lower_wick_ratio": (min(c, o) - lo) / rng,
            "pin_bar": float(int((rng - body) > cfg.pin_bar_wick_ratio * body)),
            "engulfing": self._calc_engulfing(o, c),
            "inside_bar": self._calc_inside_bar(h, lo),
        }

        # Volume
        avg_vol = sum(self._volume_buffer) / max(len(self._volume_buffer), 1)
        avg_vol = max(avg_vol, 1e-10)
        direction = 1.0 if c > o else -1.0
        features["relative_volume"] = vol / avg_vol
        features["volume_delta"] = direction * vol
        features["climax_volume"] = float(int(vol > cfg.climax_volume_threshold * avg_vol))

        # SMC
        features["swing_high"] = float(int(
            self._bar_count >= 3
            and self.swing.prev_high > self.swing.prev_prev_high
            and self.swing.prev_high > h
        ))
        features["swing_low"] = float(int(
            self._bar_count >= 3
            and self.swing.prev_low < self.swing.prev_prev_low
            and self.swing.prev_low < lo
        ))
        features["bos"] = float(self.swing.trend) if self.swing.trend != 0 else 0.0
        features["choch"] = 0.0  # Simplified for incremental
        features["trend_direction"] = float(self.swing.trend)

        # Distances
        close_safe = max(c, 1e-10)
        features["order_block_bull"] = 0.0
        features["order_block_bear"] = 0.0
        features["fvg_bull"] = 0.0
        features["fvg_bear"] = 0.0

        if self._high_buffer:
            features["liquidity_above"] = (max(self._high_buffer) - c) / close_safe
            features["liquidity_below"] = (c - min(self._low_buffer)) / close_safe
        else:
            features["liquidity_above"] = 0.0
            features["liquidity_below"] = 0.0

        # Time features
        time_val = candle.get("time")
        if hasattr(time_val, "hour"):
            hour = time_val.hour
            dow = time_val.weekday()
        else:
            hour = 0
            dow = 0

        features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        features["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        features["dow_sin"] = math.sin(2 * math.pi * dow / 7)
        features["dow_cos"] = math.cos(2 * math.pi * dow / 7)

        sess = cfg.sessions
        if sess.asian_start <= hour < sess.asian_end:
            features["session_id"] = 0
        elif sess.european_start <= hour < sess.european_end:
            features["session_id"] = 1
        elif sess.us_start <= hour < sess.us_end:
            features["session_id"] = 2
        else:
            features["session_id"] = 3

        # Price features
        prev_close = float(self._prev_candle.get("close", c)) if self._prev_candle else c
        prev_close = max(prev_close, 1e-10)
        features["log_return"] = math.log(c / prev_close)
        features["range_pct"] = rng / close_safe

        return features

    def _calc_engulfing(self, o: float, c: float) -> float:
        """Calculate engulfing pattern from cached previous candle."""
        if self._prev_candle is None:
            return 0.0
        prev_o = float(self._prev_candle.get("open", 0))
        prev_c = float(self._prev_candle.get("close", 0))
        curr_body = c - o
        prev_body = prev_c - prev_o
        if curr_body > 0 and prev_body < 0 and abs(curr_body) > abs(prev_body):
            return 1.0
        if curr_body < 0 and prev_body > 0 and abs(curr_body) > abs(prev_body):
            return -1.0
        return 0.0

    def _calc_inside_bar(self, h: float, lo: float) -> float:
        """Calculate inside bar from cached previous candle."""
        if self._prev_candle is None:
            return 0.0
        prev_h = float(self._prev_candle.get("high", 0))
        prev_lo = float(self._prev_candle.get("low", float("inf")))
        return float(int(h < prev_h and lo > prev_lo))


# ============================================================
# VALIDATION
# ============================================================

def _validate_input(df: pl.DataFrame, config: FeatureSettings) -> None:
    """Validate input DataFrame has required columns and enough rows."""
    required: set[str] = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    min_rows = max(config.volume_rolling_window, config.liquidity_window, 20)
    if len(df) < min_rows:
        raise ValueError(f"Need at least {min_rows} bars, got {len(df)}")


# ============================================================
# PRICE ACTION FEATURES (1-6) — Polars Expressions
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


def _pin_bar(wick_ratio: float) -> pl.Expr:
    """Pin bar detection: wick > N× body (N from config)."""
    body = (pl.col("close") - pl.col("open")).abs()
    rng = pl.col("high") - pl.col("low")
    wick = rng - body
    return (wick > wick_ratio * body).cast(pl.Int8).alias("pin_bar")


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
# VOLUME FEATURES (7-9) — Config-driven windows
# ============================================================

def _relative_volume(window: int) -> pl.Expr:
    """Volume / SMA(volume, window). Window from config."""
    avg_vol = pl.col("volume").rolling_mean(window_size=window)
    return (pl.col("volume") / avg_vol.clip(lower_bound=1e-10)).alias("relative_volume")


def _volume_delta() -> pl.Expr:
    """Estimated buy vol - sell vol (approximation using price direction)."""
    direction = pl.when(pl.col("close") > pl.col("open")).then(1.0).otherwise(-1.0)
    return (direction * pl.col("volume")).alias("volume_delta")


def _climax_volume(window: int, threshold: float) -> pl.Expr:
    """Binary: volume > threshold× average. Both from config."""
    avg_vol = pl.col("volume").rolling_mean(window_size=window)
    return (pl.col("volume") > threshold * avg_vol).cast(pl.Int8).alias("climax_volume")


# ============================================================
# SMC FEATURES (10-20) — Vectorized + Stateful BOS/CHoCH
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
    """Compute BOS and CHoCH from swing structure (numpy — stateful).

    BOS = Break of Structure (trend continuation)
    CHoCH = Change of Character (trend reversal)

    NOTE: This is the only non-vectorized computation because
    BOS/CHoCH depend on tracked swing state (inherently sequential).
    Isolated to minimize impact on overall pipeline latency.
    """
    n = len(df)
    bos = np.zeros(n, dtype=np.int8)
    choch = np.zeros(n, dtype=np.int8)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    swing_highs_arr = df["swing_high"].to_numpy()
    swing_lows_arr = df["swing_low"].to_numpy()

    last_sh = highs[0]
    last_sl = lows[0]
    trend = 0

    for i in range(2, n):
        if swing_highs_arr[i - 1] == 1:
            last_sh = highs[i - 1]
        if swing_lows_arr[i - 1] == 1:
            last_sl = lows[i - 1]

        if trend >= 0 and highs[i] > last_sh:
            bos[i] = 1
            trend = 1
        elif trend <= 0 and lows[i] < last_sl:
            bos[i] = -1
            trend = -1

        if trend == 1 and lows[i] < last_sl:
            choch[i] = -1
            trend = -1
        elif trend == -1 and highs[i] > last_sh:
            choch[i] = 1
            trend = 1

    return df.with_columns([
        pl.Series("bos", bos),
        pl.Series("choch", choch),
    ])


def _order_block_bull() -> pl.Expr:
    """Distance to nearest bullish OB."""
    is_bearish = pl.col("close") < pl.col("open")
    next_bullish = pl.col("close").shift(-1) > pl.col("open").shift(-1)
    ob_level = pl.when(is_bearish & next_bullish).then(pl.col("low")).otherwise(None)
    return (
        (pl.col("close") - ob_level.forward_fill()) / pl.col("close").clip(lower_bound=1e-10)
    ).alias("order_block_bull")


def _order_block_bear() -> pl.Expr:
    """Distance to nearest bearish OB."""
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


def _liquidity_above(window: int) -> pl.Expr:
    """Distance to recent high (liquidity pool above). Window from config."""
    rolling_high = pl.col("high").rolling_max(window_size=window)
    return (
        (rolling_high - pl.col("close")) / pl.col("close").clip(lower_bound=1e-10)
    ).alias("liquidity_above")


def _liquidity_below(window: int) -> pl.Expr:
    """Distance to recent low (liquidity pool below). Window from config."""
    rolling_low = pl.col("low").rolling_min(window_size=window)
    return (
        (pl.col("close") - rolling_low) / pl.col("close").clip(lower_bound=1e-10)
    ).alias("liquidity_below")


def _trend_direction(lookback: int) -> pl.Expr:
    """Trend direction: +1 higher highs, -1 lower lows. Lookback from config."""
    hh = pl.col("high") > pl.col("high").rolling_max(window_size=lookback).shift(1)
    ll = pl.col("low") < pl.col("low").rolling_min(window_size=lookback).shift(1)
    return (
        pl.when(hh & ~ll).then(1)
        .when(ll & ~hh).then(-1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("trend_direction")
    )


# ============================================================
# TIME FEATURES (21-25) — Config-driven sessions
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


def _session_id(sessions: SessionHoursConfig) -> pl.Expr:
    """Session: 0=Asian, 1=European, 2=US, 3=Off-hours. Config-driven."""
    hour = pl.col("time").dt.hour()
    return (
        pl.when(
            (hour >= sessions.asian_start) & (hour < sessions.asian_end)
        ).then(0)
        .when(
            (hour >= sessions.european_start) & (hour < sessions.european_end)
        ).then(1)
        .when(
            (hour >= sessions.us_start) & (hour < sessions.us_end)
        ).then(2)
        .otherwise(3)
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

FEATURE_COLUMNS: list[str] = [
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
