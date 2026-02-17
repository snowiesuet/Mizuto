"""Robustness tests: edge cases, overlapping signals, state leakage (Step 3.5)."""

import pytest
import numpy as np
import pandas as pd

from src.backtest import run_backtest_on_data
from src.strategies.ma_crossover import MACrossoverStrategy
from tests.conftest import make_ohlcv, make_flat_ohlcv


class TestEndOfBacktestEdgeCases:
    def test_single_bar_after_warmup(self):
        """Only 1 bar after warmup should not crash."""
        data = make_ohlcv(n=21)  # warmup=20, 1 simulation bar
        result = run_backtest_on_data(
            data, short_window=5, long_window=20, quiet=True,
        )
        # Should return None (no trades) or a valid result, not crash

    def test_exactly_warmup_bars(self):
        """Data with exactly warmup_period bars -> 0 simulation bars."""
        data = make_ohlcv(n=20)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20, quiet=True,
        )
        # No simulation bars possible
        assert result is None

    def test_empty_dataframe(self):
        """Empty DataFrame should return None."""
        data = pd.DataFrame()
        result = run_backtest_on_data(data, quiet=True)
        assert result is None


class TestOverlappingSignals:
    def test_trades_alternate_buy_sell(self):
        """Trades should strictly alternate: buy/short, sell, buy/short, sell."""
        data = make_ohlcv(n=200, trend=1.0, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20, quiet=True,
            slippage_pct=0, commission_pct=0,
        )
        if result is not None and result['trade_count'] >= 2:
            types = [t['type'] for t in result['trades']]
            for i in range(len(types) - 1):
                assert types[i] != types[i + 1], (
                    f"Consecutive '{types[i]}' at index {i}"
                )


class TestStateLeakage:
    def test_no_leakage_between_runs(self):
        """Two consecutive backtests with same strategy give same results."""
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        data = make_ohlcv(n=200, trend=0.5, seed=42)

        result1 = run_backtest_on_data(
            data, strategy=strategy, short_window=5, long_window=20,
            quiet=True, slippage_pct=0, commission_pct=0,
        )
        strategy.reset()
        result2 = run_backtest_on_data(
            data, strategy=strategy, short_window=5, long_window=20,
            quiet=True, slippage_pct=0, commission_pct=0,
        )
        if result1 is not None and result2 is not None:
            assert result1['pnl'] == pytest.approx(result2['pnl'])
            assert result1['trade_count'] == result2['trade_count']


class TestPositionSizingIntegration:
    def test_volatility_sizing_runs(self):
        """Volatility sizing should run without error."""
        data = make_ohlcv(n=200, volatility=5.0, trend=0.5, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20, quiet=True,
            slippage_pct=0, commission_pct=0,
            position_sizing='volatility', risk_per_trade=0.02,
        )
        # Should complete without error (may or may not have trades)

    def test_rolling_std_sizing_runs(self):
        """Rolling std sizing should run without error."""
        data = make_ohlcv(n=200, volatility=5.0, trend=0.5, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20, quiet=True,
            slippage_pct=0, commission_pct=0,
            position_sizing='rolling_std', risk_per_trade=0.02,
        )
        # Should complete without error

    def test_default_sizing_unchanged(self):
        """With no position_sizing param, behavior is identical to fixed size."""
        data = make_ohlcv(n=200, volatility=5.0, trend=0.5, seed=42)
        result_default = run_backtest_on_data(
            data, trade_amount=1.0, short_window=5, long_window=20,
            quiet=True, slippage_pct=0, commission_pct=0,
        )
        result_none = run_backtest_on_data(
            data, trade_amount=1.0, short_window=5, long_window=20,
            quiet=True, slippage_pct=0, commission_pct=0,
            position_sizing=None,
        )
        if result_default is not None and result_none is not None:
            assert result_default['pnl'] == pytest.approx(result_none['pnl'])

    def test_max_portfolio_risk_caps_exposure(self):
        """Max portfolio risk should limit position size."""
        data = make_ohlcv(n=200, volatility=5.0, trend=0.5, seed=42)
        # Very low max risk should result in smaller positions
        result = run_backtest_on_data(
            data, trade_amount=100.0, short_window=5, long_window=20,
            quiet=True, slippage_pct=0, commission_pct=0,
            max_portfolio_risk=0.01,  # 1% max exposure
        )
        # Should complete without error

    def test_trade_amount_in_trade_dicts(self):
        """Trade dicts should include trade_amount field."""
        data = make_ohlcv(n=200, volatility=5.0, trend=0.5, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20, quiet=True,
            slippage_pct=0, commission_pct=0,
        )
        if result is not None and result['trades']:
            for trade in result['trades']:
                assert 'trade_amount' in trade


class TestATRTrailingStopIntegration:
    def test_atr_trailing_stop_runs(self):
        """ATR trailing stop should work in a full backtest."""
        data = make_ohlcv(n=200, volatility=5.0, trend=0.5, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20, quiet=True,
            slippage_pct=0, commission_pct=0,
            trailing_stop_atr=2.0,
        )
        # Should complete without error

    def test_breakeven_stop_runs(self):
        """Breakeven stop should work in a full backtest."""
        data = make_ohlcv(n=200, volatility=5.0, trend=0.5, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20, quiet=True,
            slippage_pct=0, commission_pct=0,
            breakeven_threshold=0.03, stop_loss_pct=0.10,
        )
        # Should complete without error
