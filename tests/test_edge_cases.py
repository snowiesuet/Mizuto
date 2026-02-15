"""Edge case and adversarial input tests."""

import numpy as np
import pandas as pd
import pytest

from src.bot import TradingBot
from src.strategies.atr_breakout import ATRBreakoutStrategy


# ---------------------------------------------------------------------------
# load_historical_data edge cases
# ---------------------------------------------------------------------------


class TestLoadHistoricalData:
    def test_empty_dataframe(self):
        """Empty DataFrame → no crash, empty price history."""
        bot = TradingBot("TEST", 1)
        bot.load_historical_data(data=pd.DataFrame())
        assert bot.strategy.price_history == []

    def test_missing_close_column(self):
        """DataFrame without 'Close' column → logs warning, empty history."""
        df = pd.DataFrame({"Open": [100, 101], "High": [105, 106]})
        bot = TradingBot("TEST", 1)
        bot.load_historical_data(data=df)
        assert bot.strategy.price_history == []

    def test_nan_only_close(self):
        """Close column with only NaN → empty history after dropna."""
        dates = pd.bdate_range(start="2023-01-01", periods=5)
        df = pd.DataFrame({"Close": [np.nan] * 5}, index=dates)
        bot = TradingBot("TEST", 1)
        bot.load_historical_data(data=df)
        assert bot.strategy.price_history == []

    def test_ohlcv_strategy_gets_full_dataframe(self):
        """OHLCV strategy receives the full DataFrame via load_ohlcv_history."""
        dates = pd.bdate_range(start="2023-01-01", periods=20)
        df = pd.DataFrame({
            "Open": range(20), "High": range(1, 21),
            "Low": range(20), "Close": range(20), "Volume": [1000] * 20,
        }, index=dates)
        strategy = ATRBreakoutStrategy()
        bot = TradingBot("TEST", 1, strategy=strategy)
        bot.load_historical_data(data=df)
        assert len(strategy._closes) == 20

    def test_partial_nan_close(self):
        """Close with some NaN values → only valid prices loaded."""
        dates = pd.bdate_range(start="2023-01-01", periods=5)
        df = pd.DataFrame({"Close": [100.0, np.nan, 102.0, np.nan, 104.0]}, index=dates)
        bot = TradingBot("TEST", 1)
        bot.load_historical_data(data=df)
        assert len(bot.strategy.price_history) == 3
        assert bot.strategy.price_history == [100.0, 102.0, 104.0]
