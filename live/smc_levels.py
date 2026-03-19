"""
ScalForex -- SMC Level Calculator
Computes actual PRICE LEVELS for dynamic SL/TP placement.

Returns:
    swing_low_price, swing_high_price: Recent swing structure levels
    order_block_bull_price: Nearest bullish OB (demand zone)
    order_block_bear_price: Nearest bearish OB (supply zone)
    liquidity_above_price: Where buy-side liquidity pools
    liquidity_below_price: Where sell-side liquidity pools
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class SMCLevels(NamedTuple):
    """Raw price levels for SL/TP placement."""
    swing_low: float
    swing_high: float
    order_block_bull: float   # Demand zone (SL for BUY below this)
    order_block_bear: float   # Supply zone (SL for SELL above this)
    liquidity_above: float    # TP target for BUY
    liquidity_below: float    # TP target for SELL


def compute_smc_levels(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    current_price: float,
    lookback: int = 50,
    swing_strength: int = 3,
) -> SMCLevels:
    """Compute SMC price levels from OHLC arrays.

    Args:
        highs: Array of high prices (newest last).
        lows: Array of low prices (newest last).
        closes: Array of close prices (newest last).
        current_price: Current market price.
        lookback: How many bars back to scan.
        swing_strength: Min bars on each side for swing detection.

    Returns:
        SMCLevels with actual price values.
    """
    n = min(len(highs), lookback)
    h = highs[-n:]
    l = lows[-n:]
    c = closes[-n:]

    # ── Swing Highs & Lows ──
    swing_highs: list[float] = []
    swing_lows: list[float] = []

    for i in range(swing_strength, n - swing_strength):
        # Swing high: higher than N bars on each side
        if all(h[i] >= h[i - j] for j in range(1, swing_strength + 1)) and \
           all(h[i] >= h[i + j] for j in range(1, swing_strength + 1)):
            swing_highs.append(float(h[i]))

        # Swing low: lower than N bars on each side
        if all(l[i] <= l[i - j] for j in range(1, swing_strength + 1)) and \
           all(l[i] <= l[i + j] for j in range(1, swing_strength + 1)):
            swing_lows.append(float(l[i]))

    # Nearest swing levels (below and above current price)
    swing_lows_below = [s for s in swing_lows if s < current_price]
    swing_highs_above = [s for s in swing_highs if s > current_price]

    nearest_swing_low = max(swing_lows_below) if swing_lows_below else current_price * 0.995
    nearest_swing_high = min(swing_highs_above) if swing_highs_above else current_price * 1.005

    # ── Order Blocks ──
    # Bullish OB: Last down candle before bullish move (demand zone)
    # Bearish OB: Last up candle before bearish move (supply zone)
    ob_bull = nearest_swing_low  # Default fallback
    ob_bear = nearest_swing_high  # Default fallback

    for i in range(n - 2, max(0, n - 20), -1):
        # Bullish OB: bearish candle followed by strong bullish move
        if c[i] < c[i - 1] and i + 1 < n and c[i + 1] > c[i] * 1.001:
            ob_price = float(l[i])
            if ob_price < current_price:
                ob_bull = ob_price
                break

    for i in range(n - 2, max(0, n - 20), -1):
        # Bearish OB: bullish candle followed by strong bearish move
        if c[i] > c[i - 1] and i + 1 < n and c[i + 1] < c[i] * 0.999:
            ob_price = float(h[i])
            if ob_price > current_price:
                ob_bear = ob_price
                break

    # ── Liquidity Zones ──
    # Liquidity above = cluster of equal highs (buy stops)
    # Liquidity below = cluster of equal lows (sell stops)
    tolerance_pct = 0.001  # 0.1% tolerance for "equal" level

    # Find clusters of highs above current price
    liq_above = nearest_swing_high  # Default
    high_levels = sorted([float(x) for x in h if x > current_price])
    if len(high_levels) >= 2:
        for j in range(len(high_levels) - 1):
            if abs(high_levels[j + 1] - high_levels[j]) / high_levels[j] < tolerance_pct:
                liq_above = high_levels[j + 1] * 1.0005  # Just beyond the cluster
                break

    # Find clusters of lows below current price
    liq_below = nearest_swing_low  # Default
    low_levels = sorted([float(x) for x in l if x < current_price], reverse=True)
    if len(low_levels) >= 2:
        for j in range(len(low_levels) - 1):
            if abs(low_levels[j] - low_levels[j + 1]) / low_levels[j] < tolerance_pct:
                liq_below = low_levels[j + 1] * 0.9995  # Just beyond the cluster
                break

    return SMCLevels(
        swing_low=nearest_swing_low,
        swing_high=nearest_swing_high,
        order_block_bull=ob_bull,
        order_block_bear=ob_bear,
        liquidity_above=liq_above,
        liquidity_below=liq_below,
    )


def compute_dynamic_sl_tp(
    side: str,
    current_price: float,
    levels: SMCLevels,
    min_sl_distance: float = 0.001,
    max_sl_distance: float = 0.01,
) -> tuple[float, float, float]:
    """Compute SL, TP, and RR ratio from SMC levels.

    Args:
        side: "BUY" or "SELL"
        current_price: Entry price.
        levels: SMC price levels.
        min_sl_distance: Min SL distance as fraction of price.
        max_sl_distance: Max SL distance as fraction of price.

    Returns:
        (sl_price, tp_price, rr_ratio)
    """
    if side == "BUY":
        # SL below order block bull or swing low (whichever is closer)
        sl_candidate = min(levels.order_block_bull, levels.swing_low)
        sl = sl_candidate - current_price * 0.0002  # Tiny buffer below

        # TP at liquidity above
        tp = levels.liquidity_above
    else:
        # SL above order block bear or swing high (whichever is closer)
        sl_candidate = max(levels.order_block_bear, levels.swing_high)
        sl = sl_candidate + current_price * 0.0002  # Tiny buffer above

        # TP at liquidity below
        tp = levels.liquidity_below

    # Enforce min/max SL distance
    sl_dist = abs(current_price - sl)
    min_abs = current_price * min_sl_distance
    max_abs = current_price * max_sl_distance

    if sl_dist < min_abs:
        sl = current_price - min_abs if side == "BUY" else current_price + min_abs
        sl_dist = min_abs
    elif sl_dist > max_abs:
        sl = current_price - max_abs if side == "BUY" else current_price + max_abs
        sl_dist = max_abs

    # Calculate RR
    tp_dist = abs(tp - current_price)
    rr = tp_dist / sl_dist if sl_dist > 0 else 0.0

    return sl, tp, rr
