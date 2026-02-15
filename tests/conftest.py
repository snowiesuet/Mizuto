"""Shared test fixtures and synthetic data generators."""

import numpy as np
import pandas as pd
import pytest

from src.bot import TradingBot


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
    highs = closes + rng.uniform(1, volatility, n)
    lows = closes - rng.uniform(1, volatility, n)
    opens = closes + rng.randn(n) * 2
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
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    low = np.maximum(low, 0.5)
    open_ = close + rng.randn(n) * 0.3
    volume = rng.randint(100, 10000, n).astype(float)
    dates = pd.bdate_range(start="2023-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def make_bot(short=5, long=20, trailing_stop_pct=None, stop_loss_pct=None, strategy=None):
    """Create a TradingBot with deterministic settings (no network calls)."""
    return TradingBot(
        symbol="TEST",
        trade_amount=1,
        short_window=short,
        long_window=long,
        trailing_stop_pct=trailing_stop_pct,
        stop_loss_pct=stop_loss_pct,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ohlcv_df():
    """Default 200-bar synthetic OHLCV DataFrame."""
    return make_ohlcv()


@pytest.fixture
def flat_ohlcv_df():
    """200-bar flat OHLCV DataFrame."""
    return make_flat_ohlcv()


@pytest.fixture
def trending_ohlcv_df():
    """200-bar trending OHLCV DataFrame."""
    return make_trending_ohlcv()
