"""Tests for Phase 2: Trade Analytics & Attribution.

Covers exit reason tracking, trade duration, enhanced stats,
expectancy, Calmar ratio, and _unpack_signal helper.
"""

import pytest
import numpy as np
import pandas as pd

from src.backtest import run_backtest_on_data, _unpack_signal
from src.metrics import compute_calmar_ratio
from src.strategies.atr_breakout import ATRBreakoutStrategy
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.pivot_points import PivotPointStrategy
from src.bot import LONG_WINDOW
from tests.conftest import make_ohlcv


# ===========================================================================
# _unpack_signal helper
# ===========================================================================

class TestUnpackSignal:
    def test_bare_string(self):
        sig, reason = _unpack_signal('hold')
        assert sig == 'hold'
        assert reason is None

    def test_tuple(self):
        sig, reason = _unpack_signal(('sell', 'sl_hit'))
        assert sig == 'sell'
        assert reason == 'sl_hit'

    def test_buy_string(self):
        sig, reason = _unpack_signal('buy')
        assert sig == 'buy'
        assert reason is None

    def test_sell_tuple_signal_reversal(self):
        sig, reason = _unpack_signal(('sell', 'signal_reversal'))
        assert sig == 'sell'
        assert reason == 'signal_reversal'


# ===========================================================================
# Strategy exit reasons
# ===========================================================================

class TestMACrossoverExitReason:
    def test_sell_returns_signal_reversal(self):
        s = MACrossoverStrategy(short_window=3, long_window=5)
        s.price_history = [50.0, 50.0, 50.0, 50.0]
        signal = s.on_price(5.0, has_position=True)
        assert signal == ('sell', 'signal_reversal')

    def test_buy_remains_string(self):
        s = MACrossoverStrategy(short_window=3, long_window=5)
        s.price_history = [10.0, 10.0, 10.0, 10.0]
        signal = s.on_price(50.0, has_position=False)
        assert signal == 'buy'

    def test_hold_remains_string(self):
        s = MACrossoverStrategy(short_window=3, long_window=5)
        s.price_history = [10.0, 10.0, 10.0, 10.0]
        signal = s.on_price(50.0, has_position=True)
        assert signal == 'hold'


class TestATRBreakoutExitReasons:
    def _make_strategy_with_position(self, position_type, sl, tp):
        s = ATRBreakoutStrategy()
        s._position_type = position_type
        s._sl_price = sl
        s._tp_price = tp
        for i in range(s.warmup_period + 1):
            s._highs.append(100 + i * 0.1)
            s._lows.append(99 + i * 0.1)
            s._closes.append(100 + i * 0.1)
        return s

    def test_long_sl_hit(self):
        s = self._make_strategy_with_position('long', sl=90.0, tp=120.0)
        bar = {'Open': 89, 'High': 91, 'Low': 88, 'Close': 89, 'Volume': 1000}
        assert s.on_bar(bar, True, 'long') == ('sell', 'sl_hit')

    def test_long_tp_hit(self):
        s = self._make_strategy_with_position('long', sl=90.0, tp=105.0)
        bar = {'Open': 106, 'High': 107, 'Low': 105, 'Close': 106, 'Volume': 1000}
        assert s.on_bar(bar, True, 'long') == ('sell', 'tp_hit')

    def test_short_sl_hit(self):
        s = self._make_strategy_with_position('short', sl=110.0, tp=80.0)
        bar = {'Open': 111, 'High': 112, 'Low': 110, 'Close': 111, 'Volume': 1000}
        assert s.on_bar(bar, True, 'short') == ('sell', 'sl_hit')

    def test_short_tp_hit(self):
        s = self._make_strategy_with_position('short', sl=110.0, tp=95.0)
        bar = {'Open': 94, 'High': 95, 'Low': 93, 'Close': 94, 'Volume': 1000}
        assert s.on_bar(bar, True, 'short') == ('sell', 'tp_hit')


class TestPivotPointExitReasons:
    def _make_positioned_strategy(self, position_type, sl, tp):
        s = PivotPointStrategy()
        s._position_type = position_type
        s._sl_price = sl
        s._tp_price = tp
        s._prev_high = 110
        s._prev_low = 90
        s._prev_close = 100
        s._prev_bar_close = 100
        return s

    def test_long_sl_hit(self):
        s = self._make_positioned_strategy('long', sl=80.0, tp=100.0)
        bar = {'Open': 79, 'High': 80, 'Low': 78, 'Close': 79, 'Volume': 1000}
        assert s.on_bar(bar, True, 'long') == ('sell', 'sl_hit')

    def test_long_tp_hit(self):
        s = self._make_positioned_strategy('long', sl=80.0, tp=95.0)
        bar = {'Open': 96, 'High': 97, 'Low': 95, 'Close': 96, 'Volume': 1000}
        assert s.on_bar(bar, True, 'long') == ('sell', 'tp_hit')

    def test_short_sl_hit(self):
        s = self._make_positioned_strategy('short', sl=120.0, tp=100.0)
        bar = {'Open': 121, 'High': 122, 'Low': 120, 'Close': 121, 'Volume': 1000}
        assert s.on_bar(bar, True, 'short') == ('sell', 'sl_hit')

    def test_short_tp_hit(self):
        s = self._make_positioned_strategy('short', sl=120.0, tp=105.0)
        bar = {'Open': 104, 'High': 105, 'Low': 103, 'Close': 104, 'Volume': 1000}
        assert s.on_bar(bar, True, 'short') == ('sell', 'tp_hit')


# ===========================================================================
# Trade dicts — exit_reason, bars_held, entry_price
# ===========================================================================

class TestTradeDict:
    def _run_simple_backtest(self):
        """Run a backtest that produces at least one round trip."""
        flat = [100.0] * LONG_WINDOW
        up = [100.0 + i * 5 for i in range(1, 11)]
        down = [150.0 - i * 10 for i in range(1, 11)]
        prices = flat + up + down
        dates = pd.bdate_range(start="2023-01-01", periods=len(prices))
        df = pd.DataFrame({"Close": prices}, index=dates)
        return run_backtest_on_data(
            df, slippage_pct=0, commission_pct=0, quiet=True,
        )

    def test_sell_trade_has_exit_reason(self):
        result = self._run_simple_backtest()
        assert result is not None
        sell_trades = [t for t in result['trades'] if t['type'] == 'sell']
        assert len(sell_trades) >= 1
        for t in sell_trades:
            assert 'exit_reason' in t
            assert t['exit_reason'] in ('signal_reversal', 'sl_hit', 'tp_hit',
                                         'trailing_sl_hit', 'fixed_sl_hit',
                                         'end_of_data', None)

    def test_sell_trade_has_bars_held(self):
        result = self._run_simple_backtest()
        assert result is not None
        sell_trades = [t for t in result['trades'] if t['type'] == 'sell']
        for t in sell_trades:
            assert 'bars_held' in t
            assert isinstance(t['bars_held'], int)
            assert t['bars_held'] >= 0

    def test_sell_trade_has_entry_price(self):
        result = self._run_simple_backtest()
        assert result is not None
        sell_trades = [t for t in result['trades'] if t['type'] == 'sell']
        for t in sell_trades:
            assert 'entry_price' in t
            assert t['entry_price'] > 0


class TestForceCloseExitReason:
    def test_end_of_data_reason(self):
        """Force-close at end of backtest should have exit_reason='end_of_data'."""
        flat = [100.0] * LONG_WINDOW
        rising = [100.0 + i * 5 for i in range(1, 31)]
        prices = flat + rising
        dates = pd.bdate_range(start="2023-01-01", periods=len(prices))
        data = pd.DataFrame({"Close": prices}, index=dates)

        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
        )
        assert result is not None
        assert result['trade_count'] >= 1
        assert result['pnl'] > 0
        last_trade = result['trades'][-1]
        assert last_trade['type'] == 'sell'
        assert last_trade['closed_position'] == 'long'
        assert last_trade['exit_reason'] == 'end_of_data'
        assert last_trade['bars_held'] >= 0


# ===========================================================================
# Enhanced trade statistics
# ===========================================================================

class TestEnhancedStats:
    def _make_result_with_trades(self):
        """Run a backtest that produces multiple trades for stats testing."""
        data = make_ohlcv(n=200, base=100, volatility=5, trend=0.5, seed=42)
        return run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0, commission_pct=0, quiet=True,
        )

    def test_result_has_consecutive_wins(self):
        result = self._make_result_with_trades()
        if result is not None:
            assert 'consecutive_wins' in result
            assert isinstance(result['consecutive_wins'], int)
            assert result['consecutive_wins'] >= 0

    def test_result_has_consecutive_losses(self):
        result = self._make_result_with_trades()
        if result is not None:
            assert 'consecutive_losses' in result
            assert isinstance(result['consecutive_losses'], int)
            assert result['consecutive_losses'] >= 0

    def test_result_has_largest_win_loss(self):
        result = self._make_result_with_trades()
        if result is not None:
            assert 'largest_win' in result
            assert 'largest_loss' in result

    def test_result_has_avg_bars_held(self):
        result = self._make_result_with_trades()
        if result is not None:
            assert 'avg_bars_held' in result
            assert result['avg_bars_held'] >= 0

    def test_result_has_expectancy(self):
        result = self._make_result_with_trades()
        if result is not None:
            assert 'expectancy' in result
            assert isinstance(result['expectancy'], float)

    def test_result_has_exit_reason_counts(self):
        result = self._make_result_with_trades()
        if result is not None:
            assert 'exit_reason_counts' in result
            assert isinstance(result['exit_reason_counts'], dict)

    def test_expectancy_formula(self):
        """Verify expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)."""
        result = self._make_result_with_trades()
        if result is not None and result['trade_count'] > 0:
            tc = result['trade_count']
            win_rate = result['wins'] / tc
            loss_rate = result['losses'] / tc
            avg_win = (result['gross_profits'] / result['wins']) if result['wins'] > 0 else 0.0
            avg_loss = (result['gross_losses'] / result['losses']) if result['losses'] > 0 else 0.0
            expected = (avg_win * win_rate) - (avg_loss * loss_rate)
            assert result['expectancy'] == pytest.approx(expected)


# ===========================================================================
# Calmar ratio
# ===========================================================================

class TestCalmarRatio:
    def test_calmar_ratio_basic(self):
        """Calmar ratio with known values (equity with a drawdown)."""
        dates = pd.bdate_range(start="2023-01-01", periods=252)
        # Up to 11000, dip to 10500, then recover to 11500
        up = np.linspace(10000, 11000, 100)
        dip = np.linspace(11000, 10500, 50)
        recover = np.linspace(10500, 11500, 102)
        equity = np.concatenate([up, dip, recover]).tolist()
        ratio = compute_calmar_ratio(equity, dates.tolist())
        assert ratio > 0
        assert not np.isinf(ratio)

    def test_calmar_ratio_no_drawdown(self):
        """Straight up equity → infinite Calmar."""
        dates = pd.bdate_range(start="2023-01-01", periods=50)
        equity = list(range(10000, 10050))
        ratio = compute_calmar_ratio(equity, dates.tolist())
        assert ratio == float('inf')

    def test_calmar_ratio_negative_cagr_no_drawdown(self):
        """If CAGR ≤ 0 and no drawdown, return 0."""
        dates = pd.bdate_range(start="2023-01-01", periods=10)
        equity = [100.0] * 10  # flat → CAGR = 0
        ratio = compute_calmar_ratio(equity, dates.tolist())
        assert ratio == 0.0

    def test_calmar_ratio_insufficient_data(self):
        dates = pd.bdate_range(start="2023-01-01", periods=1)
        equity = [10000.0]
        ratio = compute_calmar_ratio(equity, dates.tolist())
        assert ratio == 0.0

    def test_calmar_in_backtest_results(self):
        """Calmar ratio should appear in backtest results."""
        data = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0, commission_pct=0, quiet=True,
        )
        if result is not None:
            assert 'calmar_ratio' in result


# ===========================================================================
# next_open fill model with enriched signals
# ===========================================================================

class TestNextOpenWithExitReasons:
    def test_next_open_trade_dicts_have_exit_reason(self):
        """next_open fill model should still produce exit_reason in trade dicts."""
        data = make_ohlcv(n=200, volatility=5.0, trend=1.0, seed=42)
        result = run_backtest_on_data(
            data, short_window=5, long_window=20,
            slippage_pct=0, commission_pct=0, quiet=True,
            fill_model="next_open",
        )
        if result is not None:
            sell_trades = [t for t in result['trades'] if t['type'] == 'sell']
            for t in sell_trades:
                assert 'exit_reason' in t
                assert 'bars_held' in t
                assert 'entry_price' in t
