"""Tests for configurable timeframes and periods_per_year (Step 3.5)."""

import pytest
import numpy as np
import pandas as pd

from src.backtest import run_backtest_on_data, infer_periods_per_year
from src.metrics import compute_sharpe_ratio
from tests.conftest import make_ohlcv


class TestInferPeriodsPerYear:
    def test_daily_data(self):
        data = make_ohlcv(n=100)  # business day index
        result = infer_periods_per_year(data)
        assert result == 252

    def test_hourly_data(self):
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
        data = pd.DataFrame({"Close": np.linspace(100, 150, 100)}, index=dates)
        result = infer_periods_per_year(data)
        assert result == int(252 * 6.5)

    def test_5min_data(self):
        dates = pd.date_range(start="2023-01-01", periods=100, freq="5min")
        data = pd.DataFrame({"Close": np.linspace(100, 150, 100)}, index=dates)
        result = infer_periods_per_year(data)
        assert result == int(252 * 6.5 * 12)

    def test_15min_data(self):
        dates = pd.date_range(start="2023-01-01", periods=100, freq="15min")
        data = pd.DataFrame({"Close": np.linspace(100, 150, 100)}, index=dates)
        result = infer_periods_per_year(data)
        assert result == int(252 * 6.5 * 4)

    def test_weekly_data(self):
        dates = pd.date_range(start="2023-01-01", periods=100, freq="W")
        data = pd.DataFrame({"Close": np.linspace(100, 150, 100)}, index=dates)
        result = infer_periods_per_year(data)
        assert result == 52

    def test_insufficient_data_defaults_daily(self):
        dates = pd.date_range(start="2023-01-01", periods=1, freq="1D")
        data = pd.DataFrame({"Close": [100.0]}, index=dates)
        result = infer_periods_per_year(data)
        assert result == 252


class TestPeriodsPerYearPassthrough:
    def test_sharpe_differs_with_periods(self):
        equity = list(np.linspace(10000, 10500, 100))
        sharpe_daily = compute_sharpe_ratio(equity, periods_per_year=252)
        sharpe_hourly = compute_sharpe_ratio(equity, periods_per_year=int(252 * 6.5))
        # Hourly annualization should produce a different (larger) Sharpe
        assert sharpe_hourly != sharpe_daily
        assert sharpe_hourly > sharpe_daily

    def test_backtest_accepts_periods_per_year(self):
        data = make_ohlcv(n=200, volatility=5.0, trend=0.5, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0, commission_pct=0, quiet=True,
            periods_per_year=1638,  # hourly
        )
        # Should complete without error
        if result is not None:
            assert 'sharpe_ratio' in result

    def test_auto_detection(self):
        data = make_ohlcv(n=200, volatility=5.0, trend=0.5, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0, commission_pct=0, quiet=True,
            periods_per_year='auto',
        )
        if result is not None:
            assert 'sharpe_ratio' in result
