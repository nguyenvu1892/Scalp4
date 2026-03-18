"""Tests for config validation — configs/validator.py"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from configs.validator import (
    RiskConfig,
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
                killswitch_drawdown_pct=0.60,  # > max_dd → invalid
                max_open_positions=2,
                confidence_threshold=0.3,
            )

    def test_max_loss_per_trade_bounds(self) -> None:
        """Max loss per trade must be between 0 and 10%."""
        with pytest.raises(ValueError):
            RiskConfig(
                max_loss_per_trade_pct=0.15,  # 15% → too high
                max_total_drawdown_pct=0.50,
                killswitch_drawdown_pct=0.45,
                max_open_positions=2,
                confidence_threshold=0.3,
            )

    def test_session_config_all_sessions(self) -> None:
        """Session config should allow all sessions."""
        config = load_trading_rules(CONFIG_DIR)
        assert config.session.trade_all_sessions is True
        assert config.session.crypto_24_7 is True
        assert config.session.force_close_eod_forex is True


# ============================================================
# SYMBOLS
# ============================================================

class TestSymbols:
    """Tests for symbols config validation."""

    def test_load_symbols_valid(self) -> None:
        """Should load symbols.yaml with all 5 symbols."""
        config = load_symbols(CONFIG_DIR)
        assert len(config.symbols) == 5
        assert "XAUUSD" in config.symbols
        assert "ETHUSD" in config.symbols
        assert "BTCUSD" in config.symbols
        assert "US30" in config.symbols
        assert "USTEC" in config.symbols

    def test_crypto_is_marked(self) -> None:
        """Crypto symbols should have is_crypto=True."""
        config = load_symbols(CONFIG_DIR)
        assert config.symbols["ETHUSD"].is_crypto is True
        assert config.symbols["BTCUSD"].is_crypto is True
        assert config.symbols["XAUUSD"].is_crypto is False
        assert config.symbols["US30"].is_crypto is False

    def test_timeframes_configured(self) -> None:
        """Should have M5 primary, M15 secondary, H1/H4 context."""
        config = load_symbols(CONFIG_DIR)
        assert config.timeframes.primary == "M5"
        assert config.timeframes.secondary == "M15"
        assert "H1" in config.timeframes.context
        assert "H4" in config.timeframes.context

    def test_features_count(self) -> None:
        """Should have 29 features configured."""
        config = load_symbols(CONFIG_DIR)
        assert config.features.total_features == 29


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
        assert config.transformer.nhead == 4
        assert len(config.curriculum.stages) == 4

    def test_transformer_d_model_nhead_divisible(self) -> None:
        """d_model must be divisible by nhead."""
        with pytest.raises(ValueError, match="divisible"):
            TransformerConfig(
                d_model=65,  # Not divisible by 4
                nhead=4,
                num_layers=2,
                dim_feedforward=128,
                dropout=0.1,
                max_sequence_length=60,
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
        assert config.reward.scalp_bonus_fast > 0
        assert config.reward.scalp_bonus_medium > 0
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
