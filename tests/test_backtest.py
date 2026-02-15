import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.backtest import run_backtest, run_backtest_on_data
from src.bot import LONG_WINDOW
from tests.conftest import make_ohlcv, make_flat_ohlcv


def _make_price_df(prices):
    """Build a DataFrame that looks like yfinance output with a 'Close' column."""
    dates = pd.bdate_range(start="2023-01-01", periods=len(prices))
    df = pd.DataFrame({"Close": prices}, index=dates)
    return df


class TestSlippageAndCommission:
    @patch("src.backtest.yf.download")
    def test_pnl_reflects_costs(self, mock_download):
        """With a clear uptrend → buy then sell, costs should reduce PnL."""
        # Build a price series:
        #   - LONG_WINDOW flat prices to prime the bot
        #   - Then a clear crossover up (buy), then crossover down (sell)
        flat = [100.0] * LONG_WINDOW
        # Rising prices → short MA crosses above long MA → buy signal
        rising = [100.0 + i * 5 for i in range(1, 11)]  # 105..150
        # Falling prices → short MA crosses below long MA → sell signal
        falling = [150.0 - i * 10 for i in range(1, 11)]  # 140..60
        prices = flat + rising + falling
        mock_download.return_value = _make_price_df(prices)

        # --- Run with ZERO costs ---
        result_free = run_backtest(
            "TEST", 1, "2023-01-01", "2023-12-31",
            slippage_pct=0.0, commission_pct=0.0,
        )

        # --- Run with costs ---
        result_cost = run_backtest(
            "TEST", 1, "2023-01-01", "2023-12-31",
            slippage_pct=0.001, commission_pct=0.001,
        )

        # Both should have executed at least one trade
        assert result_free is not None and result_free['trade_count'] >= 1
        assert result_cost is not None and result_cost['trade_count'] >= 1

        # PnL with costs should be strictly less than PnL without costs
        assert result_cost['pnl'] < result_free['pnl']
        assert result_cost['total_commission'] > 0

    @patch("src.backtest.yf.download")
    def test_no_trades_on_flat_data(self, mock_download):
        """Perfectly flat prices should produce no crossover and no trades."""
        prices = [100.0] * (LONG_WINDOW + 30)
        mock_download.return_value = _make_price_df(prices)

        result = run_backtest("TEST", 1, "2023-01-01", "2023-12-31")
        # With flat data, short MA == long MA, so no crossover → no trades
        assert result is None  # run_backtest returns None when no trades

    @patch("src.backtest.yf.download")
    def test_end_to_end_known_data(self, mock_download):
        """Construct a clear crossover pattern and verify buy/sell happen."""
        # Flat priming, then sharp rise, then sharp fall
        flat = [100.0] * LONG_WINDOW
        up = [100.0 + i * 10 for i in range(1, 16)]   # 110..250
        down = [250.0 - i * 15 for i in range(1, 16)]  # 235..40
        prices = flat + up + down
        mock_download.return_value = _make_price_df(prices)

        result = run_backtest(
            "TEST", 1, "2023-01-01", "2023-12-31",
            slippage_pct=0.0, commission_pct=0.0,
        )

        assert result is not None
        assert result['trade_count'] >= 1
        # Verify trades list alternates buy/sell
        trade_types = [t['type'] for t in result['trades']]
        for i in range(len(trade_types) - 1):
            if trade_types[i] == 'buy':
                assert trade_types[i + 1] == 'sell'


class TestEquityCurve:
    def test_equity_curve_in_results(self):
        data = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
        )
        if result is not None:
            assert 'equity_curve' in result
            assert 'equity_dates' in result
            assert 'initial_capital' in result
            # Equity curve length = number of simulation bars
            warmup = 20  # long_window
            expected_len = len(data) - warmup
            assert len(result['equity_curve']) == expected_len

    def test_flat_data_constant_equity(self):
        data = make_flat_ohlcv(n=100, price=100.0)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
            initial_capital=10000.0,
        )
        # Flat data = no trades = None
        assert result is None

    def test_custom_initial_capital(self):
        data = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
            initial_capital=50000.0,
        )
        if result is not None:
            assert result['initial_capital'] == 50000.0


class TestFillModel:
    def test_default_close_model(self):
        """Default fill model should be 'close' and work as before."""
        data = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.001, commission_pct=0.001, quiet=True,
        )
        # Should succeed without error
        if result is not None:
            assert result['trade_count'] >= 1

    def test_invalid_fill_model_raises(self):
        data = make_ohlcv(n=100)
        with pytest.raises(ValueError, match="Invalid fill_model"):
            run_backtest_on_data(
                data, short_window=5, long_window=20,
                quiet=True, fill_model="invalid",
            )

    def test_next_open_model_runs(self):
        """next_open fill model should run without errors on OHLCV data."""
        data = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.001, commission_pct=0.001, quiet=True,
            fill_model="next_open",
        )
        # May produce trades or not, but should not error

    def test_next_open_requires_open_column(self):
        """next_open with Close-only data should return None."""
        dates = pd.bdate_range(start="2023-01-01", periods=100)
        data = pd.DataFrame({"Close": np.linspace(100, 150, 100)}, index=dates)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            quiet=True, fill_model="next_open",
        )
        assert result is None

    def test_next_open_fills_differ_from_close(self):
        """next_open fills should use Open prices, producing different results."""
        data = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        result_close = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
            fill_model="close",
        )
        result_open = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
            fill_model="next_open",
        )
        # If both produce trades, PnL should typically differ
        if result_close is not None and result_open is not None:
            if result_close['trade_count'] > 0 and result_open['trade_count'] > 0:
                # At least fill prices should differ
                close_fills = [t['price'] for t in result_close['trades']]
                open_fills = [t['price'] for t in result_open['trades']]
                assert close_fills != open_fills

    def test_vwap_slippage_model_runs(self):
        """vwap_slippage fill model should run without errors."""
        data = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.001, commission_pct=0.001, quiet=True,
            fill_model="vwap_slippage",
        )
        # Should not error

    def test_vwap_slippage_differs_from_close(self):
        """vwap_slippage should produce different fills due to volume scaling."""
        # Create data with highly variable volume to ensure slippage differs
        rng = np.random.RandomState(42)
        n = 200
        dates = pd.bdate_range(start="2023-01-01", periods=n)
        closes = [100.0]
        for i in range(1, n):
            closes.append(closes[-1] + 1.0 + rng.randn() * 5.0)
        closes = np.array(closes)
        highs = closes + rng.uniform(1, 5, n)
        lows = closes - rng.uniform(1, 5, n)
        opens = closes + rng.randn(n) * 2
        # Extreme volume variation: some bars very low, some very high
        volume = np.ones(n) * 5000.0
        volume[::3] = 100.0   # every 3rd bar has very low volume
        volume[::5] = 50000.0  # every 5th bar has very high volume
        data = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volume},
            index=dates,
        )

        result_close = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.01, commission_pct=0.0, quiet=True,
            fill_model="close",
        )
        result_vwap = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.01, commission_pct=0.0, quiet=True,
            fill_model="vwap_slippage",
        )
        if result_close is not None and result_vwap is not None:
            # Fill prices should differ due to volume-scaled slippage
            close_fills = [t['price'] for t in result_close['trades']]
            vwap_fills = [t['price'] for t in result_vwap['trades']]
            assert close_fills != vwap_fills
