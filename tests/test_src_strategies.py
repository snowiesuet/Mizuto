"""Tests for src/ indicators, ATR Breakout, Pivot Points, and short-position support."""

import pytest
import numpy as np
import pandas as pd

from src.indicators import (
    compute_atr,
    compute_adx,
    compute_rolling_max,
    compute_rolling_min,
    compute_pivot_points,
)
from src.strategies.atr_breakout import ATRBreakoutStrategy
from src.strategies.pivot_points import PivotPointStrategy
from src.bot import TradingBot
from src.backtest import run_backtest_on_data
from tests.conftest import make_ohlcv


# ===========================================================================
# Indicator tests
# ===========================================================================

class TestIndicators:
    def test_compute_pivot_points_known_values(self):
        """Floor pivot formula: PP = (H+L+C)/3."""
        pivots = compute_pivot_points(110.0, 90.0, 100.0)
        assert pivots['PP'] == pytest.approx(100.0)
        assert pivots['S1'] == pytest.approx(2 * 100 - 110)  # 90
        assert pivots['R1'] == pytest.approx(2 * 100 - 90)   # 110
        assert pivots['S2'] == pytest.approx(100 - (110 - 90))  # 80
        assert pivots['R2'] == pytest.approx(100 + (110 - 90))  # 120

    def test_compute_atr_not_nan(self):
        """ATR should be a positive number with enough data."""
        df = make_ohlcv(n=50)
        val = compute_atr(df['High'], df['Low'], df['Close'], 14)
        assert not np.isnan(val)
        assert val > 0

    def test_compute_adx_returns_three_floats(self):
        df = make_ohlcv(n=50)
        adx_val, plus_di, minus_di = compute_adx(df['High'], df['Low'], df['Close'], 14)
        assert not np.isnan(adx_val)
        assert not np.isnan(plus_di)
        assert not np.isnan(minus_di)

    def test_rolling_max_min(self):
        series = [1, 3, 2, 5, 4]
        assert compute_rolling_max(series, 3) == 5.0
        assert compute_rolling_min(series, 3) == 2.0


# ===========================================================================
# ATR Breakout strategy unit tests
# ===========================================================================

class TestATRBreakoutStrategy:
    def test_requires_ohlcv(self):
        s = ATRBreakoutStrategy()
        assert s.requires_ohlcv is True

    def test_hold_with_insufficient_data(self):
        s = ATRBreakoutStrategy()
        bar = {'Open': 100, 'High': 105, 'Low': 95, 'Close': 100, 'Volume': 1000}
        assert s.on_bar(bar, False) == 'hold'

    def test_reset_clears_state(self):
        s = ATRBreakoutStrategy()
        # Feed some data
        for i in range(60):
            bar = {'Open': 100+i, 'High': 105+i, 'Low': 95+i, 'Close': 100+i, 'Volume': 1000}
            s.on_bar(bar, False)
        s.reset()
        assert s._highs == []
        assert s._lows == []
        assert s._closes == []
        assert s._sl_price is None
        assert s._position_type is None

    def test_load_ohlcv_history(self):
        s = ATRBreakoutStrategy()
        df = make_ohlcv(n=40)
        s.load_ohlcv_history(df)
        assert len(s._closes) == 40
        assert len(s._highs) == 40



# ===========================================================================
# Pivot Point strategy unit tests
# ===========================================================================

class TestPivotPointStrategy:
    def test_requires_ohlcv(self):
        s = PivotPointStrategy()
        assert s.requires_ohlcv is True

    def test_hold_on_first_bar(self):
        s = PivotPointStrategy()
        bar = {'Open': 100, 'High': 110, 'Low': 90, 'Close': 100, 'Volume': 1000}
        assert s.on_bar(bar, False) == 'hold'

    def test_s1_bounce_long_entry(self):
        """Previous close below S1, current crosses above → buy."""
        s = PivotPointStrategy()
        # First bar sets the previous HLC for pivot calculation
        bar1 = {'Open': 100, 'High': 110, 'Low': 90, 'Close': 100, 'Volume': 1000}
        s.on_bar(bar1, False)

        # Pivots from bar1: PP=(110+90+100)/3=100, S1=2*100-110=90
        # Set prev_bar_close to be below S1 (90)
        s._prev_bar_close = 85.0

        # Current bar close at 92 (above S1=90 with prev below S1) → buy
        bar2 = {'Open': 88, 'High': 95, 'Low': 85, 'Close': 92, 'Volume': 1000}
        signal = s.on_bar(bar2, False)
        assert signal == 'buy'
        assert s._position_type == 'long'

    def test_r1_rejection_short_entry(self):
        """Previous close above R1, current crosses below → short."""
        s = PivotPointStrategy()
        bar1 = {'Open': 100, 'High': 110, 'Low': 90, 'Close': 100, 'Volume': 1000}
        s.on_bar(bar1, False)

        # R1 = 2*100 - 90 = 110
        s._prev_bar_close = 115.0

        # Current close at 108 (below R1=110 with prev above) → short
        bar2 = {'Open': 112, 'High': 115, 'Low': 107, 'Close': 108, 'Volume': 1000}
        signal = s.on_bar(bar2, False)
        assert signal == 'short'
        assert s._position_type == 'short'

    def test_reset(self):
        s = PivotPointStrategy()
        bar1 = {'Open': 100, 'High': 110, 'Low': 90, 'Close': 100, 'Volume': 1000}
        s.on_bar(bar1, False)
        s.reset()
        assert s._prev_high is None
        assert s._prev_bar_close is None
        assert s._position_type is None

    def test_load_ohlcv_history(self):
        s = PivotPointStrategy()
        df = make_ohlcv(n=10)
        s.load_ohlcv_history(df)
        assert s._prev_high is not None
        assert s._prev_close is not None

    def test_s2_r2_disabled_by_default(self):
        """With use_s2_r2=False, S2 bounce should not trigger entry."""
        s = PivotPointStrategy(use_s2_r2=False)
        bar1 = {'Open': 100, 'High': 110, 'Low': 90, 'Close': 100, 'Volume': 1000}
        s.on_bar(bar1, False)
        # S2 = PP - (H-L) = 100 - 20 = 80
        s._prev_bar_close = 75.0  # below S2
        bar2 = {'Open': 78, 'High': 82, 'Low': 75, 'Close': 82, 'Volume': 1000}
        signal = s.on_bar(bar2, False)
        # S1=90, prev_close=75 < S1 but also price=82 < S1=90, so no S1 bounce
        # S2=80, prev_close=75 < S2 and price=82 > S2, but use_s2_r2=False
        assert signal == 'hold'

    def test_s2_bounce_when_enabled(self):
        """With use_s2_r2=True, S2 bounce should trigger buy."""
        s = PivotPointStrategy(use_s2_r2=True)
        bar1 = {'Open': 100, 'High': 110, 'Low': 90, 'Close': 100, 'Volume': 1000}
        s.on_bar(bar1, False)
        # S2 = 80, S1 = 90
        s._prev_bar_close = 75.0  # below S2
        bar2 = {'Open': 78, 'High': 82, 'Low': 75, 'Close': 82, 'Volume': 1000}
        signal = s.on_bar(bar2, False)
        # prev_close=75 < S2=80 and price=82 >= S2=80 → S2 bounce (buy)
        # But also prev_close=75 < S1=90 and price=82 < S1=90 → no S1 bounce
        assert signal == 'buy'


# ===========================================================================
# Bot short-position support
# ===========================================================================

class TestBotShortPosition:
    def test_position_type_default_none(self):
        bot = TradingBot("TEST", 1)
        assert bot.position_type is None

    def test_entry_long_sets_position_type(self):
        bot = TradingBot("TEST", 1)
        bot._handle_position_entry(100.0, position_type='long')
        assert bot.position_type == 'long'

    def test_entry_short_sets_position_type(self):
        bot = TradingBot("TEST", 1)
        bot._handle_position_entry(100.0, position_type='short')
        assert bot.position_type == 'short'

    def test_exit_resets_position_type(self):
        bot = TradingBot("TEST", 1)
        bot._handle_position_entry(100.0, position_type='short')
        bot._handle_position_exit()
        assert bot.position_type is None

    def test_trailing_stop_short(self):
        """Trailing stop for short: triggers when price rises above stop."""
        bot = TradingBot("TEST", 1, trailing_stop_pct=0.05,
                         strategy=ATRBreakoutStrategy())
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='short')
        # Stop should be at 100 * 1.05 = 105
        assert bot.stop_loss_price == pytest.approx(105.0)

        # Price drops to 90 → updates lowest and trailing stop
        bar = {'Open': 90, 'High': 91, 'Low': 89, 'Close': 90, 'Volume': 1000}
        # Manually update tracking (as _run_strategy_logic_bar would)
        bot.highest_price = 90.0  # tracks lowest for shorts
        bot.stop_loss_price = 90.0 * 1.05  # 94.5

        # Price rises to 95 → above stop of 94.5 → sell
        bar2 = {'Open': 95, 'High': 96, 'Low': 94, 'Close': 95, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar2)
        assert signal == ('sell', 'trailing_sl_hit')

    def test_backward_compat_entry_defaults_to_long(self):
        """Calling _handle_position_entry without position_type defaults to 'long'."""
        bot = TradingBot("TEST", 1)
        bot._handle_position_entry(100.0)
        assert bot.position_type == 'long'


# ===========================================================================
# Backtest integration — smoke tests with synthetic data
# ===========================================================================

class TestBacktestIntegration:
    def test_ma_crossover_backward_compat(self):
        """Default backtest (no strategy param) works unchanged."""
        flat = [100.0] * 20
        up = [100.0 + i * 5 for i in range(1, 11)]
        down = [150.0 - i * 10 for i in range(1, 11)]
        prices = flat + up + down
        dates = pd.bdate_range(start="2023-01-01", periods=len(prices))
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = run_backtest_on_data(df, slippage_pct=0, commission_pct=0, quiet=True)
        assert result is not None
        assert result['trade_count'] >= 1

    def test_atr_breakout_smoke(self):
        """ATR Breakout strategy runs through the backtest pipeline without error."""
        df = make_ohlcv(n=200, base=100, volatility=8, trend=1.0, seed=123)
        strategy = ATRBreakoutStrategy()
        result = run_backtest_on_data(
            df, strategy=strategy, slippage_pct=0, commission_pct=0, quiet=True,
        )
        # May or may not produce trades depending on data — just check no crash
        # and if trades exist, they're structured correctly
        if result is not None:
            assert result['trade_count'] >= 0
            trade_types = [t['type'] for t in result['trades']]
            # Every entry should eventually be followed by a sell
            assert trade_types.count('sell') == result['trade_count']

    def test_pivot_points_smoke(self):
        """Pivot Points strategy runs through the backtest pipeline without error."""
        df = make_ohlcv(n=100, base=100, volatility=10, trend=0, seed=456)
        strategy = PivotPointStrategy()
        result = run_backtest_on_data(
            df, strategy=strategy, slippage_pct=0, commission_pct=0, quiet=True,
        )
        if result is not None:
            assert result['trade_count'] >= 0

    def test_pivot_points_with_s2_r2_smoke(self):
        """Pivot Points with S2/R2 enabled runs without error."""
        df = make_ohlcv(n=100, base=100, volatility=10, trend=0, seed=789)
        strategy = PivotPointStrategy(use_s2_r2=True)
        result = run_backtest_on_data(
            df, strategy=strategy, slippage_pct=0, commission_pct=0, quiet=True,
        )
        # Just checking it runs
        assert result is None or result['trade_count'] >= 0

    def test_ohlcv_validation_error(self):
        """Strategy requiring OHLCV should fail gracefully with Close-only data."""
        prices = [100.0] * 50
        dates = pd.bdate_range(start="2023-01-01", periods=50)
        df = pd.DataFrame({"Close": prices}, index=dates)

        strategy = ATRBreakoutStrategy()
        result = run_backtest_on_data(
            df, strategy=strategy, slippage_pct=0, commission_pct=0, quiet=True,
        )
        assert result is None

class TestBaseStrategyOHLCVDefaults:
    """Test that BaseStrategy OHLCV defaults work for non-OHLCV strategies."""

    def test_on_bar_delegates_to_on_price(self):
        from src.strategies.ma_crossover import MACrossoverStrategy
        s = MACrossoverStrategy(short_window=3, long_window=5)
        s.price_history = [10.0, 10.0, 10.0, 10.0]
        bar = {'Open': 50, 'High': 55, 'Low': 45, 'Close': 50.0, 'Volume': 1000}
        signal = s.on_bar(bar, False)
        # Should delegate to on_price(50.0, False) → buy (short > long)
        assert signal == 'buy'

    def test_load_ohlcv_history_extracts_close(self):
        from src.strategies.ma_crossover import MACrossoverStrategy
        s = MACrossoverStrategy(short_window=3, long_window=5)
        df = make_ohlcv(n=10)
        s.load_ohlcv_history(df)
        assert len(s.price_history) > 0

    def test_requires_ohlcv_false_by_default(self):
        from src.strategies.ma_crossover import MACrossoverStrategy
        s = MACrossoverStrategy()
        assert s.requires_ohlcv is False
