"""Tests for Welford normalizer — data_engine/normalizer.py"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_engine.normalizer import WelfordNormalizer


class TestWelfordNormalizer:
    """Tests for Welford's running normalizer."""

    def test_basic_normalization(self) -> None:
        """Normalized data should be approx N(0,1)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((1000, 5)) * 10 + 50  # mean=50, std=10

        norm = WelfordNormalizer(n_features=5)
        norm.update(data)

        normalized = norm.normalize(data)
        # Mean should be ≈ 0, std ≈ 1
        assert np.abs(normalized.mean(axis=0)).max() < 0.1
        assert np.abs(normalized.std(axis=0) - 1.0).max() < 0.1

    def test_single_sample_update(self) -> None:
        """Should handle single sample updates."""
        norm = WelfordNormalizer(n_features=3)
        norm.update(np.array([1.0, 2.0, 3.0]))
        assert norm.count == 1
        np.testing.assert_array_equal(norm.mean, [1.0, 2.0, 3.0])

    def test_clipping(self) -> None:
        """Extreme values should be clipped."""
        norm = WelfordNormalizer(n_features=2)
        data = np.array([[0.0, 0.0]] * 100)
        norm.update(data)

        extreme = np.array([[1000.0, -1000.0]])
        result = norm.normalize(extreme, clip_range=5.0)
        assert result[0, 0] == 5.0
        assert result[0, 1] == -5.0

    def test_save_load(self, tmp_path: Path) -> None:
        """Should save and load state correctly."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 4))

        norm = WelfordNormalizer(n_features=4)
        norm.update(data)

        save_path = tmp_path / "norm.npz"
        norm.save(save_path)

        loaded = WelfordNormalizer.load(save_path)
        assert loaded.count == norm.count
        np.testing.assert_array_almost_equal(loaded.mean, norm.mean)
        np.testing.assert_array_almost_equal(loaded.m2, norm.m2)

    def test_reset(self) -> None:
        """Reset should clear all statistics."""
        norm = WelfordNormalizer(n_features=3)
        norm.update(np.array([[1.0, 2.0, 3.0]]))
        norm.reset()
        assert norm.count == 0
        np.testing.assert_array_equal(norm.mean, [0.0, 0.0, 0.0])

    def test_wrong_features_raises(self) -> None:
        """Should raise ValueError for wrong number of features."""
        norm = WelfordNormalizer(n_features=3)
        with pytest.raises(ValueError, match="Expected 3"):
            norm.update(np.array([[1.0, 2.0]]))  # 2 features != 3

    def test_invalid_n_features(self) -> None:
        """Should raise ValueError for n_features < 1."""
        with pytest.raises(ValueError, match="n_features"):
            WelfordNormalizer(n_features=0)

    def test_repr(self) -> None:
        """Should have readable repr."""
        norm = WelfordNormalizer(n_features=5)
        assert "n_features=5" in repr(norm)
        assert "count=0" in repr(norm)
