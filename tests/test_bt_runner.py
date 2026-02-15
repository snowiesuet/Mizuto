"""Tests for bt/runner.py — run_bt, optimize_bt, and strategy registry."""

import pytest
import pandas as pd
from backtesting import Backtest

from bt.runner import run_bt, optimize_bt, STRATEGIES
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


class TestStrategyRegistry:
    def test_all_strategies_registered(self):
        assert "ma_crossover" in STRATEGIES
        assert "atr_breakout" in STRATEGIES
        assert "pivot_points" in STRATEGIES

    def test_invalid_strategy_raises(self):
        with pytest.raises(KeyError):
            run_bt("nonexistent_strategy", data=make_ohlcv())


# ---------------------------------------------------------------------------
# run_bt
# ---------------------------------------------------------------------------


class TestRunBt:
    def test_returns_stats_series(self):
        data = make_ohlcv(200)
        stats = run_bt("ma_crossover", data=data)
        assert isinstance(stats, pd.Series)
        assert "Equity Final [$]" in stats.index

    def test_kwargs_override_params(self):
        data = make_ohlcv(200)
        stats = run_bt("ma_crossover", data=data, short_window=3, long_window=10)
        assert isinstance(stats, pd.Series)

    def test_non_fractional_backtest(self):
        data = make_ohlcv(200)
        stats = run_bt("ma_crossover", data=data, fractional=False)
        assert isinstance(stats, pd.Series)

    def test_atr_breakout_runs(self):
        data = make_ohlcv(300, volatility=3.0, seed=123)
        stats = run_bt("atr_breakout", data=data)
        assert isinstance(stats, pd.Series)

    def test_pivot_points_runs(self):
        data = make_ohlcv(200, volatility=10.0, seed=456)
        stats = run_bt("pivot_points", data=data)
        assert isinstance(stats, pd.Series)


# ---------------------------------------------------------------------------
# optimize_bt
# ---------------------------------------------------------------------------


class TestOptimizeBt:
    def test_optimize_ma_crossover(self):
        data = make_ohlcv(200)
        stats = optimize_bt(
            "ma_crossover",
            data=data,
            short_window=range(3, 8),
            long_window=range(15, 25, 5),
        )
        assert isinstance(stats, pd.Series)
        assert "Equity Final [$]" in stats.index
