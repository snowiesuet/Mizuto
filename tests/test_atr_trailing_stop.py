"""Tests for ATR-based trailing stops and breakeven stops (Step 3.4)."""

import pytest
from src.bot import TradingBot
from src.strategies.atr_breakout import ATRBreakoutStrategy


def _make_atr_bot(trailing_stop_atr=None, breakeven_threshold=None,
                  trailing_stop_pct=None, stop_loss_pct=None):
    """Create a TradingBot with ATR strategy for bar-level testing."""
    strategy = ATRBreakoutStrategy()
    bot = TradingBot("TEST", 1, strategy=strategy,
                     trailing_stop_atr=trailing_stop_atr,
                     breakeven_threshold=breakeven_threshold,
                     trailing_stop_pct=trailing_stop_pct,
                     stop_loss_pct=stop_loss_pct)
    # Seed enough history for warmup
    for i in range(strategy.warmup_period + 1):
        strategy._highs.append(100 + i * 0.1)
        strategy._lows.append(99 + i * 0.1)
        strategy._closes.append(100 + i * 0.1)
    return bot


class TestATRTrailingStopLong:
    def test_stop_tracks_highest_minus_atr(self):
        bot = _make_atr_bot(trailing_stop_atr=2.0)
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='long')
        bot._current_atr = 5.0  # stop distance = 10

        # Price rises to 115 -> stop at 105
        bar = {'Open': 114, 'High': 116, 'Low': 113, 'Close': 115, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar)
        assert bot.stop_loss_price == pytest.approx(105.0)
        assert signal != ('sell', 'trailing_sl_hit')

    def test_triggers_on_breach(self):
        bot = _make_atr_bot(trailing_stop_atr=2.0)
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='long')
        bot._current_atr = 5.0

        # Price rises to 115
        bar1 = {'Open': 114, 'High': 116, 'Low': 113, 'Close': 115, 'Volume': 1000}
        bot._run_strategy_logic_bar(bar1)

        # Price drops to 104 -> below 105 -> sell
        bar2 = {'Open': 106, 'High': 107, 'Low': 103, 'Close': 104, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar2)
        assert signal == ('sell', 'trailing_sl_hit')


class TestATRTrailingStopShort:
    def test_stop_tracks_lowest_plus_atr(self):
        bot = _make_atr_bot(trailing_stop_atr=2.0)
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='short')
        bot._current_atr = 5.0  # stop distance = 10

        # Price drops to 85 -> stop at 95
        bar = {'Open': 86, 'High': 87, 'Low': 84, 'Close': 85, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar)
        assert bot.stop_loss_price == pytest.approx(95.0)
        assert signal != ('sell', 'trailing_sl_hit')

    def test_triggers_on_breach(self):
        bot = _make_atr_bot(trailing_stop_atr=2.0)
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='short')
        bot._current_atr = 5.0

        # Price drops to 85
        bar1 = {'Open': 86, 'High': 87, 'Low': 84, 'Close': 85, 'Volume': 1000}
        bot._run_strategy_logic_bar(bar1)

        # Price rises to 96 -> above 95 -> sell
        bar2 = {'Open': 94, 'High': 97, 'Low': 93, 'Close': 96, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar2)
        assert signal == ('sell', 'trailing_sl_hit')


class TestATROverridesPct:
    def test_atr_takes_precedence(self):
        """When both ATR and pct trailing stops are set, ATR takes precedence."""
        bot = _make_atr_bot(trailing_stop_atr=2.0, trailing_stop_pct=0.01)
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='long')
        bot._current_atr = 5.0
        # ATR stop distance = 10. Pct stop distance = 1.
        # If ATR prevails, price at 95 should NOT trigger (95 > 90 stop)
        bar = {'Open': 96, 'High': 97, 'Low': 94, 'Close': 95, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar)
        assert signal != ('sell', 'trailing_sl_hit')

    def test_warning_logged(self, caplog):
        """Both set should log a warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            _make_atr_bot(trailing_stop_atr=2.0, trailing_stop_pct=0.05)
        assert any("ATR trailing stop takes precedence" in r.message for r in caplog.records)


class TestBreakevenStopLong:
    def test_activates_at_threshold(self):
        bot = _make_atr_bot(breakeven_threshold=0.05, stop_loss_pct=0.10)
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='long')

        # Price reaches 106 -> 6% profit > 5% threshold -> breakeven
        bar = {'Open': 105, 'High': 107, 'Low': 104, 'Close': 106, 'Volume': 1000}
        bot._run_strategy_logic_bar(bar)
        assert bot.breakeven_activated is True
        assert bot.stop_loss_price == pytest.approx(100.0)

    def test_not_activated_below_threshold(self):
        bot = _make_atr_bot(breakeven_threshold=0.05, stop_loss_pct=0.10)
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='long')

        # Price only 103 -> 3% < 5% -> no breakeven
        bar = {'Open': 102, 'High': 104, 'Low': 101, 'Close': 103, 'Volume': 1000}
        bot._run_strategy_logic_bar(bar)
        assert bot.breakeven_activated is False


class TestBreakevenStopShort:
    def test_activates_at_threshold(self):
        bot = _make_atr_bot(breakeven_threshold=0.05, stop_loss_pct=0.10)
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='short')

        # Price drops to 94 -> 6% profit -> breakeven
        bar = {'Open': 95, 'High': 96, 'Low': 93, 'Close': 94, 'Volume': 1000}
        bot._run_strategy_logic_bar(bar)
        assert bot.breakeven_activated is True
        assert bot.stop_loss_price == pytest.approx(100.0)


class TestBreakevenReset:
    def test_reset_on_exit(self):
        bot = _make_atr_bot(breakeven_threshold=0.05)
        bot.breakeven_activated = True
        bot._handle_position_exit()
        assert bot.breakeven_activated is False

    def test_reset_on_new_entry(self):
        bot = _make_atr_bot(breakeven_threshold=0.05)
        bot.breakeven_activated = True
        bot._handle_position_entry(100.0, position_type='long')
        assert bot.breakeven_activated is False
