"""Tests for bt/ backtesting.py strategies."""

import pytest
from backtesting import Backtest

from bt.strategies import MACrossoverBT, ATRBreakoutStrategy, PivotPointStrategy
from tests.conftest import make_ohlcv, make_flat_ohlcv, make_trending_ohlcv


# ---------------------------------------------------------------------------
# Smoke tests — each strategy runs without error
# ---------------------------------------------------------------------------


class TestMACrossoverSmoke:
    def test_runs(self):
        data = make_ohlcv()
        bt = Backtest(data, MACrossoverBT, cash=10_000, commission=0.001)
        stats = bt.run()
        assert stats is not None
        assert "Equity Final [$]" in stats.index

    def test_flat_data_zero_trades(self):
        data = make_flat_ohlcv()
        bt = Backtest(data, MACrossoverBT, cash=10_000, commission=0.001)
        stats = bt.run()
        assert stats["# Trades"] == 0


class TestATRBreakoutSmoke:
    def test_runs(self):
        data = make_ohlcv(n=300, volatility=3.0)
        bt = Backtest(data, ATRBreakoutStrategy, cash=10_000, commission=0.001)
        stats = bt.run()
        assert stats is not None
        assert "Equity Final [$]" in stats.index

    def test_runs_trending(self):
        data = make_trending_ohlcv(n=300)
        bt = Backtest(data, ATRBreakoutStrategy, cash=10_000, commission=0.001)
        stats = bt.run()
        assert stats is not None


class TestPivotPointSmoke:
    def test_runs(self):
        data = make_ohlcv()
        bt = Backtest(data, PivotPointStrategy, cash=10_000, commission=0.001)
        stats = bt.run()
        assert stats is not None
        assert "Equity Final [$]" in stats.index

    def test_runs_with_s2_r2(self):
        Strat = type("PivotS2R2", (PivotPointStrategy,), {"use_s2_r2": True})
        data = make_ohlcv()
        bt = Backtest(data, Strat, cash=10_000, commission=0.001)
        stats = bt.run()
        assert stats is not None


# ---------------------------------------------------------------------------
# Custom parameter tests
# ---------------------------------------------------------------------------


class TestCustomParams:
    def test_ma_crossover_custom_windows(self):
        Strat = type("MACust", (MACrossoverBT,), {"short_window": 3, "long_window": 10})
        data = make_ohlcv()
        bt = Backtest(data, Strat, cash=10_000)
        stats = bt.run()
        assert stats is not None

    def test_atr_breakout_custom_params(self):
        Strat = type(
            "ATRCust",
            (ATRBreakoutStrategy,),
            {"adx_threshold": 20, "atr_multiplier": 2.0, "lookback_period": 10},
        )
        data = make_ohlcv(n=300, volatility=3.0)
        bt = Backtest(data, Strat, cash=10_000)
        stats = bt.run()
        assert stats is not None
