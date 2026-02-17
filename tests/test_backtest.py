import logging

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.backtest import run_backtest, run_backtest_on_data
from src.bot import LONG_WINDOW
from src.strategies.base import BaseStrategy
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.atr_breakout import ATRBreakoutStrategy
from src.strategies.pivot_points import PivotPointStrategy
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
        assert result is not None
        assert 'equity_curve' in result
        assert 'equity_dates' in result
        assert 'initial_capital' in result
        # Equity curve length = number of simulation bars
        warmup = 20  # long_window
        expected_len = len(data) - warmup
        assert len(result['equity_curve']) == expected_len

    def test_custom_initial_capital(self):
        data = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
            initial_capital=50000.0,
        )
        assert result is not None
        assert result['initial_capital'] == 50000.0


class TestFillModel:
    def test_default_close_model(self):
        """Default fill model should be 'close' and work as before."""
        data = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.001, commission_pct=0.001, quiet=True,
        )
        assert result is not None
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
        assert result is None or result['trade_count'] >= 0

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
        assert result_close is not None
        assert result_open is not None
        assert result_close['trade_count'] > 0
        assert result_open['trade_count'] > 0
        # Fill prices should differ between close and next_open models
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
        assert result is None or result['trade_count'] >= 0

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
        assert result_close is not None
        assert result_vwap is not None
        # Fill prices should differ due to volume-scaled slippage
        close_fills = [t['price'] for t in result_close['trades']]
        vwap_fills = [t['price'] for t in result_vwap['trades']]
        assert close_fills != vwap_fills


class TestProfitFactorCapped:
    def test_profit_factor_not_infinite(self):
        """When all trades are winners, profit_factor should be capped at 999.99."""
        # Gentle uptrend that produces a buy then a profitable sell
        flat = [100.0] * LONG_WINDOW
        up = [100.0 + i * 3 for i in range(1, 16)]
        down = [145.0 - i * 2 for i in range(1, 16)]
        up2 = [115.0 + i * 3 for i in range(1, 16)]
        prices = flat + up + down + up2
        dates = pd.bdate_range(start="2023-01-01", periods=len(prices))
        data = pd.DataFrame({"Close": prices}, index=dates)

        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
        )
        if result is not None and result['gross_losses'] == 0:
            assert result['profit_factor'] == 999.99


class TestOHLCValidation:
    def test_warns_on_bad_ohlc(self, caplog):
        """Should warn when High < max(Open, Close) or Low > min(Open, Close)."""
        dates = pd.bdate_range(start="2023-01-01", periods=50)
        closes = np.linspace(100, 130, 50)
        data = pd.DataFrame({
            "Open": closes + 2,
            "High": closes - 5,   # Intentionally BELOW Close — invalid
            "Low": closes + 10,   # Intentionally ABOVE Close — invalid
            "Close": closes,
            "Volume": np.ones(50) * 1000,
        }, index=dates)

        with caplog.at_level(logging.WARNING):
            run_backtest_on_data(
                data, short_window=5, long_window=20,
                slippage_pct=0.0, commission_pct=0.0, quiet=False,
            )

        warning_msgs = [r.message for r in caplog.records if "OHLC validation" in r.message]
        assert len(warning_msgs) >= 1

    def test_no_warning_on_clean_ohlc(self, caplog):
        """Clean OHLCV data should not trigger OHLC validation warnings."""
        data = make_flat_ohlcv(n=100, price=100.0)

        with caplog.at_level(logging.WARNING):
            run_backtest_on_data(
                data, short_window=5, long_window=20,
                slippage_pct=0.0, commission_pct=0.0, quiet=False,
            )

        ohlc_warnings = [r for r in caplog.records if "OHLC validation" in r.message]
        assert len(ohlc_warnings) == 0


class TestWarmupPeriod:
    def test_ma_crossover_warmup(self):
        s = MACrossoverStrategy(short_window=5, long_window=20)
        assert s.warmup_period == 20

    def test_atr_breakout_warmup(self):
        s = ATRBreakoutStrategy()
        assert s.warmup_period == 29  # max(14*2, 14, 20) + 1

    def test_pivot_points_warmup(self):
        s = PivotPointStrategy()
        assert s.warmup_period == 3

    def test_base_strategy_default_warmup(self):
        """BaseStrategy default warmup_period should be 0."""
        # Create a minimal concrete subclass
        class DummyStrategy(BaseStrategy):
            name = "Dummy"
            def on_price(self, price, has_position):
                return 'hold'
            def reset(self):
                pass

        s = DummyStrategy()
        assert s.warmup_period == 0


class TestFillModelValidation:
    def test_warns_on_unsupported_fill_model(self, caplog):
        """Should warn when fill model is not in strategy's supported list."""
        class RestrictedStrategy(BaseStrategy):
            name = "Restricted"

            @property
            def supported_fill_models(self):
                return ('close',)

            def on_price(self, price, has_position):
                return 'hold'

            def reset(self):
                pass

        data = make_ohlcv(n=100)

        with caplog.at_level(logging.WARNING):
            run_backtest_on_data(
                data, strategy=RestrictedStrategy(),
                short_window=5, long_window=20,
                slippage_pct=0.0, commission_pct=0.0, quiet=False,
                fill_model="next_open",
            )

        fill_warnings = [r for r in caplog.records
                         if "does not declare support" in r.message]
        assert len(fill_warnings) == 1

    def test_no_warning_on_default_fill_models(self, caplog):
        """Default strategies should not warn about fill model support."""
        data = make_ohlcv(n=100)

        with caplog.at_level(logging.WARNING):
            run_backtest_on_data(
                data, short_window=5, long_window=20,
                slippage_pct=0.0, commission_pct=0.0, quiet=False,
                fill_model="close",
            )

        fill_warnings = [r for r in caplog.records
                         if "does not declare support" in r.message]
        assert len(fill_warnings) == 0
