"""
ScalForex — Config Validator
Pydantic v2 models để validate TẤT CẢ config YAML.
Sai config = mất tiền thật. Validate TRƯỚC khi chạy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# TRADING RULES
# ============================================================

class RiskConfig(BaseModel):
    """Risk management parameters."""

    max_loss_per_trade_pct: float = Field(
        ..., gt=0, le=0.05,
        description="Max loss per trade as fraction of balance (e.g. 0.03 = 3%, max 5%)",
    )
    max_total_drawdown_pct: float = Field(
        ..., gt=0, le=1.0,
        description="Max total drawdown before killswitch",
    )
    killswitch_drawdown_pct: float = Field(
        ..., gt=0, le=1.0,
        description="DD threshold to trigger killswitch (< max_total_drawdown)",
    )
    max_open_positions: int = Field(..., ge=1, le=10)
    confidence_threshold: float = Field(..., ge=0, le=1.0)

    @model_validator(mode="after")
    def killswitch_before_max_dd(self) -> "RiskConfig":
        if self.killswitch_drawdown_pct >= self.max_total_drawdown_pct:
            raise ValueError(
                f"killswitch_drawdown_pct ({self.killswitch_drawdown_pct}) "
                f"must be < max_total_drawdown_pct ({self.max_total_drawdown_pct})"
            )
        return self


class XauusdRulesConfig(BaseModel):
    """XAUUSD special lot rules."""

    fixed_lot: float = Field(0.01, gt=0, le=1.0)
    dynamic_threshold: float = Field(600.0, gt=0)


class SessionConfig(BaseModel):
    """Session trading rules."""

    trade_all_sessions: bool = True
    force_close_eod_forex: bool = True
    crypto_24_7: bool = True


class ExecutionConfig(BaseModel):
    """Execution parameters."""

    slippage_buffer_pct: float = Field(0.001, ge=0)
    max_spread_multiplier: float = Field(3.0, gt=1.0)
    order_timeout_seconds: int = Field(10, ge=1, le=60)


class AccountConfig(BaseModel):
    """Account info."""

    starting_balance: float = Field(..., gt=0)
    broker: str = "Exness"
    platform: str = "MT5"


class TradingRulesConfig(BaseModel):
    """Top-level trading rules config."""

    risk: RiskConfig
    xauusd_rules: XauusdRulesConfig
    session: SessionConfig
    execution: ExecutionConfig
    account: AccountConfig


# ============================================================
# SYMBOLS
# ============================================================

class SymbolSpec(BaseModel):
    """Specification for one trading symbol."""

    type: str = Field(..., pattern=r"^(commodity|crypto|index|forex)$")
    pip_size: float = Field(..., gt=0)
    pip_value_per_lot: float = Field(..., gt=0)
    min_lot: float = Field(..., gt=0)
    max_lot: float = Field(..., gt=0)
    lot_step: float = Field(..., gt=0)
    avg_spread_pips: float = Field(..., ge=0)
    sessions: list[str]
    is_crypto: bool = False

    @model_validator(mode="after")
    def min_lot_le_max(self) -> "SymbolSpec":
        if self.min_lot > self.max_lot:
            raise ValueError(f"min_lot ({self.min_lot}) > max_lot ({self.max_lot})")
        return self


class TimeframeConfig(BaseModel):
    """Timeframe configuration."""

    primary: str = "M5"
    secondary: str = "M15"
    context: list[str] = ["H1", "H4"]


class SessionHoursConfig(BaseModel):
    """Session time boundaries (UTC hours). Validated for no overlap."""

    asian_start: int = Field(0, ge=0, le=23)
    asian_end: int = Field(8, ge=0, le=23)
    european_start: int = Field(8, ge=0, le=23)
    european_end: int = Field(15, ge=0, le=23)
    us_start: int = Field(15, ge=0, le=23)
    us_end: int = Field(22, ge=0, le=23)

    @model_validator(mode="after")
    def sessions_no_overlap_and_ordered(self) -> "SessionHoursConfig":
        """Validate sessions are ordered and don't overlap."""
        if self.asian_start >= self.asian_end:
            raise ValueError("asian_start must be < asian_end")
        if self.european_start >= self.european_end:
            raise ValueError("european_start must be < european_end")
        if self.us_start >= self.us_end:
            raise ValueError("us_start must be < us_end")
        if self.asian_end > self.european_start:
            raise ValueError(
                f"Asian session end ({self.asian_end}) overlaps "
                f"European start ({self.european_start})"
            )
        if self.european_end > self.us_start:
            raise ValueError(
                f"European session end ({self.european_end}) overlaps "
                f"US start ({self.us_start})"
            )
        return self


class FeatureSettings(BaseModel):
    """Feature pipeline settings — ZERO HARDCODE: all params from YAML."""

    lookback_window: int = Field(60, ge=10, le=500)
    total_features: int = Field(29, ge=1)
    normalize: bool = True
    clip_range: float = Field(5.0, gt=0, le=20.0)

    # Rolling window sizes
    volume_rolling_window: int = Field(20, ge=5, le=200)
    swing_lookback: int = Field(10, ge=3, le=100)
    liquidity_window: int = Field(20, ge=5, le=200)

    # Thresholds
    climax_volume_threshold: float = Field(2.5, gt=1.0, le=10.0)
    pin_bar_wick_ratio: float = Field(2.0, gt=1.0, le=5.0)

    # Session hours
    sessions: SessionHoursConfig = SessionHoursConfig()


class SymbolsConfig(BaseModel):
    """Top-level symbols config."""

    symbols: dict[str, SymbolSpec]
    timeframes: TimeframeConfig
    features: FeatureSettings

    @field_validator("symbols")
    @classmethod
    def at_least_one_symbol(cls, v: dict[str, SymbolSpec]) -> dict[str, SymbolSpec]:
        if not v:
            raise ValueError("Must have at least 1 symbol configured")
        return v


# ============================================================
# TRAIN CONFIG
# ============================================================

class SacConfig(BaseModel):
    """SAC hyperparameters."""

    learning_rate: float = Field(3e-4, gt=0)
    buffer_size: int = Field(500000, ge=1000)
    batch_size: int = Field(256, ge=32)
    gamma: float = Field(0.99, ge=0, le=1.0)
    tau: float = Field(0.005, gt=0, le=1.0)
    ent_coef: str | float = "auto"
    target_entropy: str | float = "auto"
    actor_hidden: list[int] = [256, 256]
    critic_hidden: list[int] = [256, 256]
    action_dim: int = Field(2, ge=1)
    action_bounds: dict[str, list[float]] = {}


class TransformerConfig(BaseModel):
    """Transformer encoder config."""

    d_model: int = Field(64, ge=16)
    nhead: int = Field(4, ge=1)
    num_layers: int = Field(2, ge=1)
    dim_feedforward: int = Field(128, ge=32)
    dropout: float = Field(0.1, ge=0, le=0.5)
    max_sequence_length: int = Field(60, ge=10)

    @model_validator(mode="after")
    def d_model_divisible_by_nhead(self) -> "TransformerConfig":
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )
        return self


class CrossAttentionConfig(BaseModel):
    """Cross-attention for multi-TF."""

    d_model: int = Field(64, ge=16)
    nhead: int = Field(4, ge=1)
    context_window: int = Field(24, ge=1)
    use_causal_mask: bool = True


class RegimeConfig(BaseModel):
    """Regime detector config."""

    hidden_dim: int = Field(64, ge=16)
    num_regimes: int = Field(3, ge=2)
    learnable: bool = True


class CurriculumStage(BaseModel):
    """One curriculum training stage."""

    name: str
    steps: int = Field(..., ge=1000)
    max_dd_pct: float = Field(..., gt=0, le=1.0)
    spread_mode: str = Field(..., pattern=r"^(fixed|variable|realistic)$")
    regimes: list[str]
    use_augmentation: bool = False


class CurriculumConfig(BaseModel):
    """Curriculum learning config."""

    stages: list[CurriculumStage]

    @field_validator("stages")
    @classmethod
    def at_least_one_stage(cls, v: list[CurriculumStage]) -> list[CurriculumStage]:
        if not v:
            raise ValueError("Must have at least 1 curriculum stage")
        return v


class RewardConfig(BaseModel):
    """Scalping-optimized reward weights."""

    pnl_weight: float = Field(1.0, ge=0)
    shaping_weight: float = Field(0.05, ge=0)
    scalp_bonus_fast: float = Field(0.5, ge=0)
    scalp_bonus_medium: float = Field(0.2, ge=0)
    hold_penalty_rate: float = Field(0.01, ge=0)
    dd_penalty_alpha: float = Field(2.0, ge=0)
    dd_penalty_beta: float = Field(3.0, ge=0)
    overtrade_penalty: float = Field(0.3, ge=0)
    rr_bonus_weight: float = Field(0.2, ge=0)
    max_trades_per_day: int = Field(10, ge=1)


class TrainingSettings(BaseModel):
    """Training loop settings."""

    total_steps: int = Field(1000000, ge=1000)
    eval_freq: int = Field(10000, ge=100)
    save_freq: int = Field(50000, ge=1000)
    log_to_wandb: bool = True
    seed: int = 42
    device: str = "auto"


class TrainConfig(BaseModel):
    """Top-level training config."""

    sac: SacConfig
    transformer: TransformerConfig
    cross_attention: CrossAttentionConfig
    regime: RegimeConfig
    curriculum: CurriculumConfig
    reward: RewardConfig
    training: TrainingSettings


# ============================================================
# LOADER — Load + validate tất cả configs
# ============================================================

def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return parsed dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty config file: {path}")
    return data


def load_trading_rules(config_dir: Optional[Path] = None) -> TradingRulesConfig:
    """Load and validate trading_rules.yaml."""
    config_dir = config_dir or Path(__file__).parent
    data = _load_yaml(config_dir / "trading_rules.yaml")
    return TradingRulesConfig(**data)


def load_symbols(config_dir: Optional[Path] = None) -> SymbolsConfig:
    """Load and validate symbols.yaml."""
    config_dir = config_dir or Path(__file__).parent
    data = _load_yaml(config_dir / "symbols.yaml")
    return SymbolsConfig(**data)


def load_train_config(config_dir: Optional[Path] = None) -> TrainConfig:
    """Load and validate train_config.yaml."""
    config_dir = config_dir or Path(__file__).parent
    data = _load_yaml(config_dir / "train_config.yaml")
    return TrainConfig(**data)


def load_all_configs(
    config_dir: Optional[Path] = None,
) -> tuple[TradingRulesConfig, SymbolsConfig, TrainConfig]:
    """Load and validate ALL config files. Raises on any error."""
    config_dir = config_dir or Path(__file__).parent
    trading_rules = load_trading_rules(config_dir)
    symbols = load_symbols(config_dir)
    train_config = load_train_config(config_dir)
    return trading_rules, symbols, train_config
