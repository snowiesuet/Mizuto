"""Tests for walk-forward integration into run_backtest_on_data (Step 3.5)."""

import pytest
from src.backtest import run_backtest_on_data
from tests.conftest import make_ohlcv


class TestWalkForwardIntegration:
    @pytest.fixture
    def data(self):
        return make_ohlcv(n=300, volatility=5.0, trend=0.5, seed=42)

    def test_walk_forward_flag_returns_wf_result(self, data):
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.001, commission_pct=0.001, quiet=True,
            walk_forward=True,
            walk_forward_kwargs={
                'short_window_range': range(3, 8),
                'long_window_range': range(15, 25, 5),
            },
        )
        assert 'train_params' in result
        assert 'test_metric' in result
        assert 'overfit_ratio' in result
        assert 'full_backtest' in result

    def test_full_backtest_is_valid(self, data):
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.001, commission_pct=0.001, quiet=True,
            walk_forward=True,
            walk_forward_kwargs={
                'short_window_range': range(3, 8),
                'long_window_range': range(15, 25, 5),
            },
        )
        fb = result['full_backtest']
        if fb is not None:
            assert 'pnl' in fb
            assert 'trades' in fb

    def test_walk_forward_false_returns_normal(self, data):
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.001, commission_pct=0.001, quiet=True,
            walk_forward=False,
        )
        if result is not None:
            assert 'pnl' in result
            assert 'train_params' not in result

    def test_train_params_valid(self, data):
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.001, commission_pct=0.001, quiet=True,
            walk_forward=True,
            walk_forward_kwargs={
                'short_window_range': range(3, 8),
                'long_window_range': range(15, 25, 5),
            },
        )
        bp = result['train_params']
        assert bp['short_window'] < bp['long_window']
