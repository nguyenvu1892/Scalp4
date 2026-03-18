"""
ScalForex — MT5 Data Fetcher
Kéo historical M5/M15/H1/H4 data từ Exness MT5.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from utils.logger import setup_logger

log = setup_logger("data_engine.mt5_fetcher")

# MT5 Timeframe mapping
TIMEFRAME_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
}


def fetch_mt5_data(
    symbol: str,
    timeframe: str,
    bars: int = 50000,
    from_date: Optional[datetime] = None,
) -> pl.DataFrame:
    """Fetch historical OHLCV data from MT5.

    Args:
        symbol: MT5 symbol name (e.g. "XAUUSD").
        timeframe: Timeframe string (e.g. "M5", "H1").
        bars: Number of bars to fetch.
        from_date: Optional start date. If None, fetches latest bars.

    Returns:
        Polars DataFrame with columns: [time, open, high, low, close, volume, spread].

    Raises:
        RuntimeError: If MT5 is not available or fetch fails.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise RuntimeError(
            "MetaTrader5 package not installed. Install via: pip install MetaTrader5"
        )

    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unknown timeframe: {timeframe}. Valid: {list(TIMEFRAME_MAP.keys())}")

    tf = TIMEFRAME_MAP[timeframe]

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        if from_date is not None:
            rates = mt5.copy_rates_from(symbol, tf, from_date, bars)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"No data returned for {symbol} {timeframe}. "
                f"MT5 error: {mt5.last_error()}"
            )

        log.info(f"Fetched {len(rates)} bars for {symbol} {timeframe}")

        df = pl.DataFrame({
            "time": [datetime.utcfromtimestamp(r[0]) for r in rates],
            "open": np.array([r[1] for r in rates], dtype=np.float64),
            "high": np.array([r[2] for r in rates], dtype=np.float64),
            "low": np.array([r[3] for r in rates], dtype=np.float64),
            "close": np.array([r[4] for r in rates], dtype=np.float64),
            "volume": np.array([r[5] for r in rates], dtype=np.float64),
            "spread": np.array([r[6] for r in rates], dtype=np.float64),
        })

        return df

    finally:
        mt5.shutdown()


def save_to_parquet(df: pl.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    log.info(f"Saved {len(df)} rows to {path}")


def load_from_parquet(path: Path) -> pl.DataFrame:
    """Load DataFrame from parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pl.read_parquet(path)
    log.info(f"Loaded {len(df)} rows from {path}")
    return df


def fetch_all_symbols(
    symbols: list[str],
    timeframes: list[str],
    bars: int = 50000,
    data_dir: Path = Path("data"),
) -> dict[str, dict[str, pl.DataFrame]]:
    """Fetch data for all symbols and timeframes, save to parquet.

    Returns:
        Nested dict: {symbol: {timeframe: DataFrame}}.
    """
    result: dict[str, dict[str, pl.DataFrame]] = {}

    for symbol in symbols:
        result[symbol] = {}
        for tf in timeframes:
            parquet_path = data_dir / f"{symbol}_{tf}.parquet"

            if parquet_path.exists():
                log.info(f"Loading cached {symbol} {tf} from {parquet_path}")
                result[symbol][tf] = load_from_parquet(parquet_path)
            else:
                log.info(f"Fetching {symbol} {tf} from MT5...")
                df = fetch_mt5_data(symbol, tf, bars)
                save_to_parquet(df, parquet_path)
                result[symbol][tf] = df

    return result
