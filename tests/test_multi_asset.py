"""Tests for src.multi_asset — multi-symbol backtesting."""

import pytest
import pandas as pd

from src.multi_asset import run_multi_backtest_on_data
from tests.conftest import make_ohlcv, make_flat_ohlcv


class TestMultiBacktestOnData:
    @pytest.fixture
    def two_datasets(self):
        return {
            'SYM-A': make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42),
            'SYM-B': make_ohlcv(n=200, volatility=5.0, trend=-0.5, seed=99),
        }

    def test_returns_expected_keys(self, two_datasets):
        result = run_multi_backtest_on_data(two_datasets)
        assert 'per_symbol' in result
        assert 'summary' in result
        assert 'aggregate' in result

    def test_per_symbol_has_both(self, two_datasets):
        result = run_multi_backtest_on_data(two_datasets)
        assert 'SYM-A' in result['per_symbol']
        assert 'SYM-B' in result['per_symbol']

    def test_summary_dataframe_shape(self, two_datasets):
        result = run_multi_backtest_on_data(two_datasets)
        summary = result['summary']
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2

    def test_aggregate_total_pnl(self, two_datasets):
        result = run_multi_backtest_on_data(two_datasets)
        # Total PnL should be sum of individual PnLs
        valid = {k: v for k, v in result['per_symbol'].items() if v is not None}
        if valid:
            expected = sum(v['pnl'] for v in valid.values())
            assert result['aggregate']['total_pnl'] == pytest.approx(expected)

    def test_aggregate_symbols_tested(self, two_datasets):
        result = run_multi_backtest_on_data(two_datasets)
        assert result['aggregate']['symbols_tested'] == 2

    def test_handles_no_trades_symbol(self):
        datasets = {
            'ACTIVE': make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42),
            'FLAT': make_flat_ohlcv(n=200, price=100.0),
        }
        result = run_multi_backtest_on_data(datasets)
        assert result['per_symbol']['FLAT'] is None
        assert result['aggregate']['symbols_tested'] == 2

    def test_summary_columns(self, two_datasets):
        result = run_multi_backtest_on_data(two_datasets)
        summary = result['summary']
        expected_cols = {'symbol', 'pnl', 'trade_count', 'win_rate',
                        'profit_factor', 'sharpe_ratio', 'max_drawdown_pct'}
        assert expected_cols.issubset(set(summary.columns))
