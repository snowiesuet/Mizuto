"""Shared test fixtures and synthetic data generators."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Synthetic data generators (available as module-level functions AND fixtures)
# ---------------------------------------------------------------------------


def make_ohlcv(n=200, base=100.0, volatility=5.0, trend=0.0, seed=42):
    """Generate synthetic OHLCV data with a random walk.

    Parameters
    ----------
    n : int
        Number of bars.
    base : float
        Starting close price.
    volatility : float
        Scale of random noise added each bar.
    trend : float
        Linear drift per bar (positive = uptrend).
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n)
    closes = [base]
    for i in range(1, n):
        closes.append(closes[-1] + trend + rng.randn() * volatility)
    closes = np.array(closes)
    opens = closes + rng.randn(n) * 2
    bar_max = np.maximum(opens, closes)
    bar_min = np.minimum(opens, closes)
    highs = bar_max + rng.uniform(1, volatility, n)
    lows = bar_min - rng.uniform(1, volatility, n)
    volume = rng.randint(1000, 10000, n).astype(float)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume},
        index=dates,
    )


def make_flat_ohlcv(n=200, price=100.0):
    """Constant-price OHLCV — no crossovers should fire."""
    dates = pd.bdate_range(start="2023-01-01", periods=n)
    arr = np.full(n, price)
    return pd.DataFrame(
        {"Open": arr, "High": arr + 0.01, "Low": arr - 0.01, "Close": arr, "Volume": np.ones(n) * 1000},
        index=dates,
    )


def make_trending_ohlcv(n=200, start=50.0, end=150.0, seed=99):
    """Linearly trending OHLCV data — good for triggering breakout entries."""
    close = np.linspace(start, end, n)
    rng = np.random.RandomState(seed)
    noise = rng.randn(n) * 0.5
    close = np.maximum(close + noise, 1.0)
    open_ = close + rng.randn(n) * 0.3
    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)
    high = bar_max + rng.uniform(0.5, 2.0, n)
    low = bar_min - rng.uniform(0.5, 2.0, n)
    low = np.maximum(low, 0.5)
    volume = rng.randint(100, 10000, n).astype(float)
    dates = pd.bdate_range(start="2023-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


