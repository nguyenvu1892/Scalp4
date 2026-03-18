"""Tests for config validation — configs/validator.py (Refactored)"""

from __future__ import annotations

from pathlib import Path

import pytest

from configs.validator import (
    FeatureSettings,
    RiskConfig,
    SessionHoursConfig,
    SymbolsConfig,
    TradingRulesConfig,
    TrainConfig,
    TransformerConfig,
    load_all_configs,
    load_symbols,
    load_trading_rules,
    load_train_config,
)

CONFIG_DIR = Path(__file__).parent.parent / "configs"


# ============================================================
# TRADING RULES
# ============================================================

class TestTradingRules:
    """Tests for trading rules config validation."""

    def test_load_trading_rules_valid(self) -> None:
        """Should load trading_rules.yaml without errors."""
        config = load_trading_rules(CONFIG_DIR)
        assert config.risk.max_loss_per_trade_pct == 0.03
        assert config.risk.max_open_positions == 2
        assert config.xauusd_rules.fixed_lot == 0.01
        assert config.xauusd_rules.dynamic_threshold == 600.0
        assert config.account.starting_balance == 200.0

    def test_killswitch_less_than_max_dd(self) -> None:
        """Killswitch DD must be less than max DD."""
        with pytest.raises(ValueError, match="killswitch_drawdown_pct"):
            RiskConfig(
                max_loss_per_trade_pct=0.03,
                max_total_drawdown_pct=0.50,
                killswitch_drawdown_pct=0.60,
                max_open_positions=2,
                confidence_threshold=0.3,
            )

    def test_max_loss_per_trade_max_5pct(self) -> None:
        """Max loss per trade must be <= 5% (Tech Lead requirement)."""
        with pytest.raises(ValueError):
            RiskConfig(
                max_loss_per_trade_pct=0.06,  # 6% → exceeds 5% cap
                max_total_drawdown_pct=0.50,
                killswitch_drawdown_pct=0.45,
                max_open_positions=2,
                confidence_threshold=0.3,
            )

    def test_max_loss_per_trade_5pct_is_ok(self) -> None:
        """5% should be the max allowed value."""
        config = RiskConfig(
            max_loss_per_trade_pct=0.05,
            max_total_drawdown_pct=0.50,
            killswitch_drawdown_pct=0.45,
            max_open_positions=2,
            confidence_threshold=0.3,
        )
        assert config.max_loss_per_trade_pct == 0.05

    def test_session_config_all_sessions(self) -> None:
        """Session config should allow all sessions."""
        config = load_trading_rules(CONFIG_DIR)
        assert config.session.trade_all_sessions is True
        assert config.session.crypto_24_7 is True
        assert config.session.force_close_eod_forex is True


# ============================================================
# SESSION HOURS CROSS-VALIDATION
# ============================================================

class TestSessionHours:
    """Tests for session hours cross-validation."""

    def test_valid_sessions(self) -> None:
        """Default sessions should be valid."""
        sess = SessionHoursConfig()
        assert sess.asian_start == 0
        assert sess.us_end == 22

    def test_overlap_raises(self) -> None:
        """Overlapping sessions should raise ValueError."""
        with pytest.raises(ValueError, match="overlaps"):
            SessionHoursConfig(
                asian_start=0,
                asian_end=10,      # Overlaps with European start=8
                european_start=8,
                european_end=15,
                us_start=15,
                us_end=22,
            )

    def test_reversed_session_raises(self) -> None:
        """Start >= end should raise ValueError."""
        with pytest.raises(ValueError, match="must be <"):
            SessionHoursConfig(
                asian_start=8,
                asian_end=0,  # End before start
                european_start=8,
                european_end=15,
                us_start=15,
                us_end=22,
            )


# ============================================================
# SYMBOLS + FEATURES
# ============================================================

class TestSymbols:
    """Tests for symbols config validation."""

    def test_load_symbols_valid(self) -> None:
        """Should load symbols.yaml with all 5 symbols."""
        config = load_symbols(CONFIG_DIR)
        assert len(config.symbols) == 5
        assert "XAUUSD" in config.symbols

    def test_crypto_is_marked(self) -> None:
        """Crypto symbols should have is_crypto=True."""
        config = load_symbols(CONFIG_DIR)
        assert config.symbols["ETHUSD"].is_crypto is True
        assert config.symbols["BTCUSD"].is_crypto is True
        assert config.symbols["XAUUSD"].is_crypto is False

    def test_timeframes_configured(self) -> None:
        """Should have M5 primary, M15 secondary, H1/H4 context."""
        config = load_symbols(CONFIG_DIR)
        assert config.timeframes.primary == "M5"
        assert "H1" in config.timeframes.context

    def test_features_count(self) -> None:
        """Should have 29 features configured."""
        config = load_symbols(CONFIG_DIR)
        assert config.features.total_features == 29

    def test_feature_params_loaded(self) -> None:
        """All extracted feature params should load from YAML."""
        config = load_symbols(CONFIG_DIR)
        f = config.features
        assert f.volume_rolling_window == 20
        assert f.climax_volume_threshold == 2.5
        assert f.pin_bar_wick_ratio == 2.0
        assert f.swing_lookback == 10
        assert f.liquidity_window == 20
        assert f.sessions.asian_start == 0
        assert f.sessions.us_end == 22

    def test_feature_params_valid_ranges(self) -> None:
        """Feature params should reject invalid values."""
        with pytest.raises(ValueError):
            FeatureSettings(climax_volume_threshold=0.5)  # Must be > 1.0


# ============================================================
# TRAIN CONFIG
# ============================================================

class TestTrainConfig:
    """Tests for training config validation."""

    def test_load_train_config_valid(self) -> None:
        """Should load train_config.yaml without errors."""
        config = load_train_config(CONFIG_DIR)
        assert config.sac.action_dim == 2
        assert config.transformer.d_model == 64
        assert len(config.curriculum.stages) == 4

    def test_transformer_d_model_nhead_divisible(self) -> None:
        """d_model must be divisible by nhead."""
        with pytest.raises(ValueError, match="divisible"):
            TransformerConfig(
                d_model=65, nhead=4, num_layers=2,
                dim_feedforward=128, dropout=0.1, max_sequence_length=60,
            )

    def test_curriculum_stages_ordered(self) -> None:
        """Curriculum should have progressive stages."""
        config = load_train_config(CONFIG_DIR)
        stages = config.curriculum.stages
        assert stages[0].name == "kindergarten"
        assert stages[-1].name == "university"
        assert stages[-1].use_augmentation is True

    def test_reward_scalping_bonuses(self) -> None:
        """Scalping reward bonuses should be configured."""
        config = load_train_config(CONFIG_DIR)
        assert config.reward.scalp_bonus_fast > config.reward.scalp_bonus_medium


# ============================================================
# LOAD ALL
# ============================================================

class TestLoadAll:
    """Test loading all configs at once."""

    def test_load_all_valid(self) -> None:
        """Should load all 3 configs without errors."""
        trading, symbols, train = load_all_configs(CONFIG_DIR)
        assert isinstance(trading, TradingRulesConfig)
        assert isinstance(symbols, SymbolsConfig)
        assert isinstance(train, TrainConfig)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_trading_rules(tmp_path)
