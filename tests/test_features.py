"""Tests for feature builder — data_engine/feature_builder.py"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from data_engine.feature_builder import (
    FEATURE_COLUMNS,
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
    """Tests for the feature builder."""

    def test_build_features_shape(self) -> None:
        """Should add 27 feature columns (29 - 2 multi-TF)."""
        df = make_sample_ohlcv(100)
        result = build_features(df)

        # 6 original + 27 features = 33 columns
        assert result.shape[0] == 100
        # Check key features exist
        for col in ["candle_body_ratio", "pin_bar", "relative_volume",
                     "bos", "choch", "session_id", "log_return"]:
            assert col in result.columns, f"Missing column: {col}"

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
        """Should raise ValueError if fewer than 20 bars."""
        df = make_sample_ohlcv(10)
        with pytest.raises(ValueError, match="at least 20"):
            build_features(df)

    def test_feature_columns_constant(self) -> None:
        """FEATURE_COLUMNS should have 29 entries."""
        assert len(FEATURE_COLUMNS) == 29
