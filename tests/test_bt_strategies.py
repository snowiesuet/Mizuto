"""Tests for bt/ backtesting.py strategies."""

import numpy as np
import pandas as pd
import pytest
from backtesting import Backtest

from bt.strategies import MACrossoverBT, ATRBreakoutStrategy, PivotPointStrategy


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 200, base: float = 100.0, volatility: float = 2.0, seed: int = 42):
    """Generate synthetic OHLCV data with random-walk prices."""
    rng = np.random.RandomState(seed)
    close = base + np.cumsum(rng.randn(n) * volatility)
    close = np.maximum(close, 1.0)  # keep positive

    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    low = np.maximum(low, 0.5)
    open_ = close + rng.randn(n) * 0.5
    volume = rng.randint(100, 10000, n).astype(float)

    dates = pd.bdate_range(start="2023-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _make_flat_ohlcv(n: int = 200, price: float = 100.0):
    """Constant-price OHLCV — no crossovers should fire."""
    dates = pd.bdate_range(start="2023-01-01", periods=n)
    arr = np.full(n, price)
    return pd.DataFrame(
        {"Open": arr, "High": arr + 0.01, "Low": arr - 0.01, "Close": arr, "Volume": np.ones(n) * 1000},
        index=dates,
    )


def _make_trending_ohlcv(n: int = 200, start: float = 50.0, end: float = 150.0):
    """Linearly trending data — good for triggering breakout entries."""
    close = np.linspace(start, end, n)
    rng = np.random.RandomState(99)
    noise = rng.randn(n) * 0.5
    close = close + noise
    close = np.maximum(close, 1.0)

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


# ---------------------------------------------------------------------------
# Smoke tests — each strategy runs without error
# ---------------------------------------------------------------------------


class TestMACrossoverSmoke:
    def test_runs(self):
        data = _make_ohlcv()
        bt = Backtest(data, MACrossoverBT, cash=10_000, commission=0.001)
        stats = bt.run()
        assert stats is not None
        assert "Equity Final [$]" in stats.index

    def test_flat_data_zero_trades(self):
        data = _make_flat_ohlcv()
        bt = Backtest(data, MACrossoverBT, cash=10_000, commission=0.001)
        stats = bt.run()
        assert stats["# Trades"] == 0


class TestATRBreakoutSmoke:
    def test_runs(self):
        data = _make_ohlcv(n=300, volatility=3.0)
        bt = Backtest(data, ATRBreakoutStrategy, cash=10_000, commission=0.001)
        stats = bt.run()
        assert stats is not None
        assert "Equity Final [$]" in stats.index

    def test_runs_trending(self):
        data = _make_trending_ohlcv(n=300)
        bt = Backtest(data, ATRBreakoutStrategy, cash=10_000, commission=0.001)
        stats = bt.run()
        assert stats is not None


class TestPivotPointSmoke:
    def test_runs(self):
        data = _make_ohlcv()
        bt = Backtest(data, PivotPointStrategy, cash=10_000, commission=0.001)
        stats = bt.run()
        assert stats is not None
        assert "Equity Final [$]" in stats.index

    def test_runs_with_s2_r2(self):
        Strat = type("PivotS2R2", (PivotPointStrategy,), {"use_s2_r2": True})
        data = _make_ohlcv()
        bt = Backtest(data, Strat, cash=10_000, commission=0.001)
        stats = bt.run()
        assert stats is not None


# ---------------------------------------------------------------------------
# Custom parameter tests
# ---------------------------------------------------------------------------


class TestCustomParams:
    def test_ma_crossover_custom_windows(self):
        Strat = type("MACust", (MACrossoverBT,), {"short_window": 3, "long_window": 10})
        data = _make_ohlcv()
        bt = Backtest(data, Strat, cash=10_000)
        stats = bt.run()
        assert stats is not None

    def test_atr_breakout_custom_params(self):
        Strat = type(
            "ATRCust",
            (ATRBreakoutStrategy,),
            {"adx_threshold": 20, "atr_multiplier": 2.0, "lookback_period": 10},
        )
        data = _make_ohlcv(n=300, volatility=3.0)
        bt = Backtest(data, Strat, cash=10_000)
        stats = bt.run()
        assert stats is not None
