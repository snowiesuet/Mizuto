"""Tests for src.metrics — risk-adjusted performance metrics."""

import math
import pytest
import numpy as np
import pandas as pd

from src.metrics import (
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_cagr,
    compute_annualized_return,
    compute_buy_and_hold_return,
    compute_all_metrics,
)
from src.backtest import run_backtest_on_data
from tests.conftest import make_ohlcv, make_flat_ohlcv


class TestMaxDrawdown:
    def test_known_values(self):
        equity = [100, 110, 90, 95, 80, 100]
        result = compute_max_drawdown(equity)
        # Max drawdown: 80 vs peak 110 = (80-110)/110 = -27.27%
        assert result['max_drawdown_pct'] == pytest.approx(-30 / 110, rel=1e-4)

    def test_no_drawdown_monotonic(self):
        equity = [100, 110, 120, 130, 140]
        result = compute_max_drawdown(equity)
        assert result['max_drawdown_pct'] == 0.0
        assert result['max_drawdown_duration'] == 0

    def test_drawdown_duration(self):
        # Peak at 110, drawdown for 3 bars (90, 95, 80), recovery at 110
        equity = [100, 110, 90, 95, 80, 100, 110]
        result = compute_max_drawdown(equity)
        # Duration from bar after peak to recovery: 4 bars (90, 95, 80, 100)
        assert result['max_drawdown_duration'] >= 3

    def test_single_value(self):
        result = compute_max_drawdown([100])
        assert result['max_drawdown_pct'] == 0.0

    def test_drawdown_series_length(self):
        equity = [100, 110, 90, 100]
        result = compute_max_drawdown(equity)
        assert len(result['drawdown_series']) == len(equity)


class TestSharpeRatio:
    def test_constant_equity(self):
        equity = [100] * 50
        assert compute_sharpe_ratio(equity) == 0.0

    def test_positive_returns(self):
        # Consistently rising equity
        equity = [100 + i * 0.5 for i in range(100)]
        sharpe = compute_sharpe_ratio(equity)
        assert sharpe > 0

    def test_insufficient_data(self):
        assert compute_sharpe_ratio([100]) == 0.0
        assert compute_sharpe_ratio([]) == 0.0

    def test_negative_returns(self):
        equity = [100 - i * 0.5 for i in range(50)]
        sharpe = compute_sharpe_ratio(equity)
        assert sharpe < 0


class TestSortinoRatio:
    def test_no_downside(self):
        equity = [100 + i for i in range(50)]
        sortino = compute_sortino_ratio(equity)
        assert sortino == float('inf')

    def test_constant_equity(self):
        equity = [100] * 50
        assert compute_sortino_ratio(equity) == 0.0

    def test_with_downside(self):
        # Mix of up and down
        equity = [100, 105, 103, 108, 106, 110]
        sortino = compute_sortino_ratio(equity)
        assert isinstance(sortino, float)


class TestCAGR:
    def test_known_values(self):
        # 100 to 200 over ~1 year (252 trading days)
        dates = pd.bdate_range(start="2023-01-01", periods=253)
        equity = np.linspace(100, 200, 253).tolist()
        cagr = compute_cagr(equity, dates)
        # Should be close to 100% annual (tolerance for bday vs calendar day mismatch)
        assert cagr == pytest.approx(1.0, abs=0.1)

    def test_single_point(self):
        assert compute_cagr([100], [pd.Timestamp("2023-01-01")]) == 0.0

    def test_zero_start(self):
        assert compute_cagr([0, 100], [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-01")]) == 0.0


class TestAnnualizedReturn:
    def test_positive(self):
        equity = np.linspace(100, 200, 253).tolist()
        ann = compute_annualized_return(equity)
        assert ann > 0

    def test_constant(self):
        equity = [100] * 100
        assert compute_annualized_return(equity) == 0.0


class TestBuyAndHoldReturn:
    def test_known_values(self):
        dates = pd.bdate_range(start="2023-01-01", periods=5)
        data = pd.DataFrame({"Close": [100, 110, 120, 130, 150]}, index=dates)
        result = compute_buy_and_hold_return(data, 10000.0)
        assert result['bh_return_pct'] == pytest.approx(50.0)
        assert result['bh_equity_final'] == pytest.approx(15000.0)
        assert len(result['bh_equity_curve']) == 5

    def test_flat_prices(self):
        dates = pd.bdate_range(start="2023-01-01", periods=3)
        data = pd.DataFrame({"Close": [100, 100, 100]}, index=dates)
        result = compute_buy_and_hold_return(data, 10000.0)
        assert result['bh_return_pct'] == pytest.approx(0.0)


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        dates = pd.bdate_range(start="2023-01-01", periods=50)
        equity = np.linspace(100, 150, 50).tolist()
        data = pd.DataFrame({"Close": np.linspace(10, 15, 50)}, index=dates)

        result = compute_all_metrics(equity, dates.tolist(), data, 100.0)

        expected_keys = {
            'max_drawdown_pct', 'max_drawdown_duration', 'drawdown_series',
            'sharpe_ratio', 'sortino_ratio', 'cagr', 'annualized_return',
            'bh_return_pct', 'bh_equity_final', 'bh_equity_curve',
        }
        assert expected_keys.issubset(set(result.keys()))


class TestMetricsIntegration:
    def test_backtest_result_includes_metrics(self):
        data = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
        )
        if result is not None:
            assert 'equity_curve' in result
            assert 'sharpe_ratio' in result
            assert 'max_drawdown_pct' in result
            assert 'bh_return_pct' in result
            assert len(result['equity_curve']) > 0

    def test_equity_curve_starts_at_initial_capital(self):
        data = make_flat_ohlcv(n=100, price=100.0)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
            initial_capital=25000.0,
        )
        # Flat data = no trades = None result, but let's test with trending data
        data2 = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        result2 = run_backtest_on_data(
            data2, short_window=5, long_window=20,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
            initial_capital=25000.0,
        )
        if result2 is not None:
            assert result2['initial_capital'] == 25000.0
            # First equity value should be initial_capital (no trade on first bar typically)
            assert result2['equity_curve'][0] == pytest.approx(25000.0, rel=0.1)

    def test_equity_curve_ends_near_pnl(self):
        data = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        capital = 10000.0
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
            initial_capital=capital,
        )
        if result is not None and result['trade_count'] > 0:
            # If no open position at end, equity should be close to initial + pnl
            final_equity = result['equity_curve'][-1]
            expected = capital + result['pnl']
            # Allow some tolerance for open positions
            assert final_equity == pytest.approx(expected, rel=0.15)
