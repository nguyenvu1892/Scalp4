"""Tests for feature builder — data_engine/feature_builder.py (Refactored)"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from configs.validator import FeatureSettings
from data_engine.feature_builder import (
    FEATURE_COLUMNS,
    IncrementalFeatureBuilder,
    build_features,
)


def make_sample_ohlcv(n: int = 100, seed: int = 42) -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    rng = np.random.default_rng(seed)

    base_price = 2000.0
    prices = base_price + np.cumsum(rng.standard_normal(n) * 2.0)

    opens = prices
    highs = prices + rng.uniform(0, 5, n)
    lows = prices - rng.uniform(0, 5, n)
    closes = prices + rng.standard_normal(n) * 2.0
    volumes = rng.uniform(100, 10000, n)

    start = datetime(2025, 1, 1, 0, 0)
    times = [start + timedelta(minutes=5 * i) for i in range(n)]

    return pl.DataFrame({
        "time": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


class TestBuildFeatures:
    """Tests for the batch feature builder."""

    def test_build_features_shape(self) -> None:
        """Should add 27 feature columns (29 - 2 multi-TF)."""
        df = make_sample_ohlcv(100)
        result = build_features(df)
        assert result.shape[0] == 100
        for col in ["candle_body_ratio", "pin_bar", "relative_volume",
                     "bos", "choch", "session_id", "log_return"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_config_driven_build(self) -> None:
        """Should accept custom FeatureSettings config."""
        cfg = FeatureSettings(
            volume_rolling_window=10,
            climax_volume_threshold=3.0,
            pin_bar_wick_ratio=1.5,
        )
        df = make_sample_ohlcv(100)
        result = build_features(df, config=cfg)
        assert "relative_volume" in result.columns

    def test_candle_body_ratio_non_negative(self) -> None:
        """Body ratio should be non-negative."""
        df = make_sample_ohlcv(100)
        result = build_features(df)
        ratio = result["candle_body_ratio"].drop_nulls()
        assert ratio.min() >= 0

    def test_pin_bar_binary(self) -> None:
        """Pin bar should be 0 or 1."""
        df = make_sample_ohlcv(100)
        result = build_features(df)
        pin = result["pin_bar"].drop_nulls()
        assert set(pin.unique().to_list()).issubset({0, 1})

    def test_engulfing_values(self) -> None:
        """Engulfing should be -1, 0, or 1."""
        df = make_sample_ohlcv(100)
        result = build_features(df)
        eng = result["engulfing"].drop_nulls()
        assert set(eng.unique().to_list()).issubset({-1, 0, 1})

    def test_session_id_values(self) -> None:
        """Session ID should be 0, 1, 2, or 3."""
        df = make_sample_ohlcv(200)
        result = build_features(df)
        sess = result["session_id"].drop_nulls()
        assert set(sess.unique().to_list()).issubset({0, 1, 2, 3})

    def test_hour_sin_cos_range(self) -> None:
        """Sin/cos should be between -1 and 1."""
        df = make_sample_ohlcv(100)
        result = build_features(df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            vals = result[col].drop_nulls()
            assert vals.min() >= -1.0 - 1e-6
            assert vals.max() <= 1.0 + 1e-6

    def test_climax_volume_binary(self) -> None:
        """Climax volume should be 0 or 1."""
        df = make_sample_ohlcv(100)
        result = build_features(df)
        cv = result["climax_volume"].drop_nulls()
        assert set(cv.unique().to_list()).issubset({0, 1})

    def test_missing_column_raises(self) -> None:
        """Should raise ValueError if required column missing."""
        df = pl.DataFrame({"time": [1, 2, 3], "open": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing columns"):
            build_features(df)

    def test_too_few_bars_raises(self) -> None:
        """Should raise ValueError if fewer than min required bars."""
        df = make_sample_ohlcv(10)
        with pytest.raises(ValueError, match="at least"):
            build_features(df)

    def test_feature_columns_constant(self) -> None:
        """FEATURE_COLUMNS should have 29 entries."""
        assert len(FEATURE_COLUMNS) == 29


# ============================================================
# INCREMENTAL BUILDER TESTS
# ============================================================

class TestIncrementalBuilder:
    """Tests for the stateful incremental feature builder."""

    def test_warm_up(self) -> None:
        """Should warm up without errors."""
        cfg = FeatureSettings()
        builder = IncrementalFeatureBuilder(config=cfg)
        df = make_sample_ohlcv(100)
        builder.warm_up(df)
        assert builder._bar_count == 100

    def test_incremental_update(self) -> None:
        """Should return features dict on update."""
        cfg = FeatureSettings()
        builder = IncrementalFeatureBuilder(config=cfg)
        df = make_sample_ohlcv(50)
        builder.warm_up(df)

        new_candle = {
            "time": datetime(2025, 1, 2, 0, 0),
            "open": 2010.0,
            "high": 2015.0,
            "low": 2008.0,
            "close": 2012.0,
            "volume": 5000.0,
        }
        features = builder.update(new_candle)
        assert isinstance(features, dict)
        assert "candle_body_ratio" in features
        assert "session_id" in features
        assert "log_return" in features

    def test_incremental_feature_count(self) -> None:
        """Should produce 27 features (excluding h1/h4_trend)."""
        cfg = FeatureSettings()
        builder = IncrementalFeatureBuilder(config=cfg)
        df = make_sample_ohlcv(50)
        builder.warm_up(df)

        new_candle = {
            "time": datetime(2025, 1, 2, 0, 0),
            "open": 2010.0, "high": 2015.0,
            "low": 2008.0, "close": 2012.0, "volume": 5000.0,
        }
        features = builder.update(new_candle)
        assert len(features) == 27

    def test_incremental_latency_single_candle(self) -> None:
        """Single candle update should be < 1ms."""
        cfg = FeatureSettings()
        builder = IncrementalFeatureBuilder(config=cfg)
        df = make_sample_ohlcv(200)
        builder.warm_up(df)

        new_candle = {
            "time": datetime(2025, 1, 2, 0, 0),
            "open": 2010.0, "high": 2015.0,
            "low": 2008.0, "close": 2012.0, "volume": 5000.0,
        }
        start = time.perf_counter()
        for _ in range(100):
            builder.update(new_candle)
        elapsed = (time.perf_counter() - start) / 100
        assert elapsed < 0.001, f"Single update took {elapsed*1000:.2f}ms (target < 1ms)"


# ============================================================
# LATENCY BENCHMARK
# ============================================================

class TestLatencyBenchmark:
    """Latency benchmarks for Tech Lead review."""

    def test_batch_1000_bars_latency(self) -> None:
        """Batch feature build on 1000 bars should be < 200ms."""
        df = make_sample_ohlcv(1000, seed=99)

        start = time.perf_counter()
        result = build_features(df)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.shape[0] == 1000
        print(f"\n⏱  BATCH LATENCY: {elapsed_ms:.1f}ms for 1000 bars")
        assert elapsed_ms < 200, f"Too slow: {elapsed_ms:.1f}ms (target < 200ms)"

    def test_incremental_1000_updates_latency(self) -> None:
        """1000 incremental updates should be < 50ms total."""
        cfg = FeatureSettings()
        builder = IncrementalFeatureBuilder(config=cfg)
        warmup = make_sample_ohlcv(100, seed=77)
        builder.warm_up(warmup)

        rng = np.random.default_rng(88)
        candles = []
        for i in range(1000):
            candles.append({
                "time": datetime(2025, 2, 1) + timedelta(minutes=5 * i),
                "open": 2000 + rng.standard_normal() * 5,
                "high": 2005 + rng.uniform(0, 3),
                "low": 1995 - rng.uniform(0, 3),
                "close": 2000 + rng.standard_normal() * 5,
                "volume": rng.uniform(100, 10000),
            })

        start = time.perf_counter()
        for c in candles:
            builder.update(c)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"⏱  INCREMENTAL LATENCY: {elapsed_ms:.1f}ms for 1000 updates "
              f"({elapsed_ms/1000:.3f}ms/candle)")
        assert elapsed_ms < 50, f"Too slow: {elapsed_ms:.1f}ms (target < 50ms)"
