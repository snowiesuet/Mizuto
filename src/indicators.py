"""Indicator helper functions for src/ strategies.

Each function works on lists or arrays and returns scalar values (the latest
reading) for use in bar-by-bar strategy logic.  Uses the ``ta`` library for
ATR and ADX calculations.
"""

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator


def compute_atr(highs, lows, closes, length: int = 14) -> float:
    """Return the latest ATR value.

    Args:
        highs, lows, closes: Array-like of OHLC data (length >= *length* + 1).
        length: ATR window.

    Returns:
        Latest ATR as a float, or NaN if insufficient data.
    """
    indicator = AverageTrueRange(
        high=pd.Series(highs, dtype=float),
        low=pd.Series(lows, dtype=float),
        close=pd.Series(closes, dtype=float),
        window=length,
    )
    vals = indicator.average_true_range()
    last = vals.iloc[-1]
    return float(last)


def compute_adx(highs, lows, closes, length: int = 14) -> tuple[float, float, float]:
    """Return ``(ADX, +DI, -DI)`` for the latest bar.

    Args:
        highs, lows, closes: Array-like of OHLC data.
        length: ADX window.

    Returns:
        Tuple of (ADX, +DI, -DI) floats.
    """
    indicator = ADXIndicator(
        high=pd.Series(highs, dtype=float),
        low=pd.Series(lows, dtype=float),
        close=pd.Series(closes, dtype=float),
        window=length,
    )
    return (
        float(indicator.adx().iloc[-1]),
        float(indicator.adx_pos().iloc[-1]),
        float(indicator.adx_neg().iloc[-1]),
    )


def compute_rolling_max(series, window: int) -> float:
    """Return the rolling maximum of *series* over the last *window* values."""
    s = pd.Series(series, dtype=float)
    return float(s.rolling(window).max().iloc[-1])


def compute_rolling_min(series, window: int) -> float:
    """Return the rolling minimum of *series* over the last *window* values."""
    s = pd.Series(series, dtype=float)
    return float(s.rolling(window).min().iloc[-1])


def compute_pivot_points(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """Classic floor pivot points from a single previous bar.

    Returns:
        Dict with keys PP, S1, S2, S3, R1, R2, R3.
    """
    pp = (prev_high + prev_low + prev_close) / 3
    s1 = 2 * pp - prev_high
    r1 = 2 * pp - prev_low
    s2 = pp - (prev_high - prev_low)
    r2 = pp + (prev_high - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)
    r3 = prev_high + 2 * (pp - prev_low)
    return {"PP": pp, "S1": s1, "S2": s2, "S3": s3, "R1": r1, "R2": r2, "R3": r3}


def compute_rolling_std(closes, window: int = 20) -> float:
    """Return rolling standard deviation of returns over the last *window* periods.

    Args:
        closes: Array-like of close prices (length >= window + 1).
        window: Lookback period.

    Returns:
        Rolling std of returns as float, or NaN if insufficient data.
    """
    if len(closes) < window + 1:
        return float('nan')
    s = pd.Series(closes, dtype=float)
    returns = s.pct_change().dropna()
    return float(returns.rolling(window).std().iloc[-1])
