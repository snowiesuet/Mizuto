import pytest
import pandas as pd
from unittest.mock import patch
from src.backtest import run_backtest
from src.bot import LONG_WINDOW


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
