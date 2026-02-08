"""
Tests for Monte Carlo Permutation Testing components.
"""

import pytest
import numpy as np
import pandas as pd
from tests.bar_permute import get_permutation
from src.backtest import run_backtest_on_data
from src.optimize import optimize_strategy_fast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlc_df(n_bars=100, start_price=100.0, seed=42):
    """Generate a synthetic OHLC DataFrame with capitalized columns (yfinance format)."""
    np.random.seed(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n_bars)
    close = start_price + np.cumsum(np.random.randn(n_bars) * 2)
    close = np.maximum(close, 10.0)
    high = close + np.abs(np.random.randn(n_bars)) * 1.5
    low = close - np.abs(np.random.randn(n_bars)) * 1.5
    low = np.maximum(low, 1.0)
    open_ = close + np.random.randn(n_bars) * 0.5
    open_ = np.maximum(open_, 1.0)
    # Ensure high >= open and high >= close, low <= open and low <= close
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low, 'Close': close
    }, index=dates)


# ---------------------------------------------------------------------------
# bar_permute tests
# ---------------------------------------------------------------------------

class TestBarPermute:
    def test_output_shape_matches_input(self):
        df = make_ohlc_df(50)
        perm = get_permutation(df, seed=1)
        assert perm.shape == df.shape
        assert (perm.index == df.index).all()

    def test_output_columns_match_input_case(self):
        df = make_ohlc_df(50)
        perm = get_permutation(df, seed=1)
        assert list(perm.columns) == list(df.columns)

    def test_lowercase_columns_also_work(self):
        df = make_ohlc_df(50).rename(columns=str.lower)
        perm = get_permutation(df, seed=1)
        assert list(perm.columns) == ['open', 'high', 'low', 'close']

    def test_start_index_preserves_prefix(self):
        df = make_ohlc_df(50)
        start = 10
        perm = get_permutation(df, start_index=start, seed=1)
        pd.testing.assert_frame_equal(perm.iloc[:start], df.iloc[:start])

    def test_permuted_data_differs_from_original(self):
        df = make_ohlc_df(100)
        perm = get_permutation(df, seed=1)
        assert not np.allclose(df['Close'].values, perm['Close'].values)

    def test_reproducibility_with_seed(self):
        df = make_ohlc_df(50)
        perm1 = get_permutation(df, seed=42)
        perm2 = get_permutation(df, seed=42)
        pd.testing.assert_frame_equal(perm1, perm2)

    def test_all_prices_positive(self):
        df = make_ohlc_df(100)
        perm = get_permutation(df, seed=1)
        assert (perm > 0).all().all()

    def test_nothing_to_permute_returns_copy(self):
        df = make_ohlc_df(5)
        perm = get_permutation(df, start_index=4, seed=1)
        pd.testing.assert_frame_equal(perm, df)


# ---------------------------------------------------------------------------
# Profit factor tests
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_profit_factor_in_results(self):
        """Uptrend then downtrend should produce trades with profit_factor."""
        flat = [100.0] * 25
        rising = [100.0 + i * 5 for i in range(1, 16)]
        falling = [175.0 - i * 10 for i in range(1, 16)]
        prices = flat + rising + falling
        dates = pd.bdate_range(start="2023-01-01", periods=len(prices))
        data = pd.DataFrame({'Close': prices}, index=dates)

        result = run_backtest_on_data(data, slippage_pct=0.0, commission_pct=0.0, quiet=True)
        if result is not None and result['trade_count'] > 0:
            assert 'profit_factor' in result
            assert result['profit_factor'] >= 0

    def test_no_trades_returns_none(self):
        """Flat data producing no trades should return None."""
        prices = [100.0] * 50
        dates = pd.bdate_range(start="2023-01-01", periods=50)
        data = pd.DataFrame({'Close': prices}, index=dates)
        result = run_backtest_on_data(data, quiet=True)
        assert result is None

    def test_gross_profits_and_losses_tracked(self):
        """Results should include gross_profits and gross_losses."""
        flat = [100.0] * 25
        rising = [100.0 + i * 5 for i in range(1, 16)]
        falling = [175.0 - i * 10 for i in range(1, 16)]
        prices = flat + rising + falling
        dates = pd.bdate_range(start="2023-01-01", periods=len(prices))
        data = pd.DataFrame({'Close': prices}, index=dates)

        result = run_backtest_on_data(data, slippage_pct=0.0, commission_pct=0.0, quiet=True)
        if result is not None:
            assert 'gross_profits' in result
            assert 'gross_losses' in result


# ---------------------------------------------------------------------------
# Optimization tests
# ---------------------------------------------------------------------------

class TestOptimization:
    def test_optimize_returns_valid_params(self):
        df = make_ohlc_df(100)
        sw_range = range(3, 8)
        lw_range = range(10, 25, 5)
        best_params, best_metric = optimize_strategy_fast(df, sw_range, lw_range)
        assert best_params['short_window'] in sw_range
        assert best_params['long_window'] in lw_range
        assert best_params['short_window'] < best_params['long_window']

    def test_optimize_short_less_than_long(self):
        df = make_ohlc_df(80)
        best_params, _ = optimize_strategy_fast(df, range(3, 10), range(5, 20, 3))
        assert best_params['short_window'] < best_params['long_window']

    def test_optimize_on_permuted_data(self):
        """Optimizer should work on permuted data too."""
        df = make_ohlc_df(100)
        perm = get_permutation(df, seed=7)
        best_params, best_metric = optimize_strategy_fast(perm, range(3, 8), range(10, 25, 5))
        assert best_params['short_window'] < best_params['long_window']
        assert isinstance(best_metric, float)
