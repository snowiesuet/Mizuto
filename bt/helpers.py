"""Shared indicator helper functions for backtesting.py strategies.

All functions return numpy arrays so they can be passed directly to
``self.I()`` inside a ``backtesting.Strategy`` subclass.

Uses the ``ta`` library (https://github.com/bukosabino/ta) for ATR/ADX
since ``pandas-ta`` requires Python >=3.12.
"""

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator


def sma(series, window: int):
    """Simple moving average."""
    return pd.Series(series).rolling(window).mean().to_numpy()


def atr(high, low, close, length: int = 14):
    """Average True Range via the ``ta`` library."""
    indicator = AverageTrueRange(
        high=pd.Series(high, dtype=float),
        low=pd.Series(low, dtype=float),
        close=pd.Series(close, dtype=float),
        window=length,
    )
    return indicator.average_true_range().to_numpy()


def adx(high, low, close, length: int = 14):
    """ADX with +DI / -DI via the ``ta`` library.

    Returns:
        Tuple of (ADX, +DI, -DI) numpy arrays.
    """
    indicator = ADXIndicator(
        high=pd.Series(high, dtype=float),
        low=pd.Series(low, dtype=float),
        close=pd.Series(close, dtype=float),
        window=length,
    )
    return (
        indicator.adx().to_numpy(),
        indicator.adx_pos().to_numpy(),
        indicator.adx_neg().to_numpy(),
    )


def rolling_max(series, window: int):
    """Rolling maximum (highest high over *window* bars)."""
    return pd.Series(series).rolling(window).max().to_numpy()


def rolling_min(series, window: int):
    """Rolling minimum (lowest low over *window* bars)."""
    return pd.Series(series).rolling(window).min().to_numpy()


def pivot_points(high, low, close):
    """Classic floor pivot points from the *previous* bar's HLC.

    Returns:
        Dict with keys PP, S1, S2, S3, R1, R2, R3 — each a numpy array
        the same length as the inputs.  The first element is NaN (no
        previous bar).
    """
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)

    # Use previous bar's values
    ph = h.shift(1)
    pl = l.shift(1)
    pc = c.shift(1)

    pp = (ph + pl + pc) / 3
    s1 = 2 * pp - ph
    r1 = 2 * pp - pl
    s2 = pp - (ph - pl)
    r2 = pp + (ph - pl)
    s3 = pl - 2 * (ph - pp)
    r3 = ph + 2 * (pp - pl)

    return {
        "PP": pp.to_numpy(),
        "S1": s1.to_numpy(),
        "S2": s2.to_numpy(),
        "S3": s3.to_numpy(),
        "R1": r1.to_numpy(),
        "R2": r2.to_numpy(),
        "R3": r3.to_numpy(),
    }
