import pytest
from unittest.mock import patch, MagicMock
from src.bot import TradingBot
from src.strategies.atr_breakout import ATRBreakoutStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bot(short=5, long=20, trailing_stop_pct=None, stop_loss_pct=None, strategy=None):
    """Create a TradingBot with deterministic settings (no network calls)."""
    return TradingBot(
        symbol="TEST",
        trade_amount=1,
        short_window=short,
        long_window=long,
        trailing_stop_pct=trailing_stop_pct,
        stop_loss_pct=stop_loss_pct,
        strategy=strategy,
    )


def feed_prices(bot, prices):
    """Feed a list of prices into the bot's strategy history without triggering strategy."""
    bot.strategy.price_history = list(prices[-bot.long_window:])


# ---------------------------------------------------------------------------
# Moving-average calculation
# ---------------------------------------------------------------------------

class TestMovingAverages:
    def test_insufficient_data_returns_none(self):
        bot = make_bot(short=3, long=5)
        bot.strategy.price_history = [1.0, 2.0, 3.0]  # only 3, need 5
        short_ma, long_ma = bot.strategy.calculate_moving_averages()
        assert short_ma is None
        assert long_ma is None

    def test_known_values(self):
        bot = make_bot(short=3, long=5)
        bot.strategy.price_history = [10.0, 20.0, 30.0, 40.0, 50.0]
        short_ma, long_ma = bot.strategy.calculate_moving_averages()
        # short MA (last 3): mean(30, 40, 50) = 40
        assert short_ma == pytest.approx(40.0)
        # long MA (last 5): mean(10, 20, 30, 40, 50) = 30
        assert long_ma == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

class TestSignals:
    def test_buy_signal_when_short_above_long(self):
        """Short MA > Long MA and no position → buy."""
        bot = make_bot(short=3, long=5)
        # Build history so that short MA > long MA after adding next price
        # History: [10, 10, 10, 10, 10] → long MA = 10
        # After appending 30: [10, 10, 10, 10, 30] → short MA(last 3) = mean(10,10,30) ≈ 16.67
        # Actually we need a clearer crossover. Use prices that make it obvious.
        bot.strategy.price_history = [10.0, 10.0, 10.0, 10.0]  # 4 prices, need 5 total
        bot.has_position = False
        # Append a high price to push short MA above long MA
        signal = bot._run_strategy_logic(50.0)
        # Now history = [10, 10, 10, 10, 50] (5 items)
        # short MA (last 3): mean(10, 10, 50) = 23.33
        # long MA (last 5): mean(10, 10, 10, 10, 50) = 18
        # short > long → buy
        assert signal == 'buy'

    def test_sell_signal_when_short_below_long(self):
        """Short MA < Long MA and holding → sell."""
        bot = make_bot(short=3, long=5)
        bot.strategy.price_history = [50.0, 50.0, 50.0, 50.0]  # 4 prices
        bot.has_position = True
        bot.entry_price = 50.0
        bot.highest_price = 50.0
        # Append a low price
        signal = bot._run_strategy_logic(5.0)
        # history = [50, 50, 50, 50, 5]
        # short MA (last 3): mean(50, 50, 5) = 35
        # long MA (last 5): mean(50, 50, 50, 50, 5) = 41
        # short < long → sell
        assert signal == ('sell', 'signal_reversal')

    def test_hold_when_no_crossover_no_position(self):
        """Short MA > Long MA but already no position and already generated buy before → hold.
        Actually: if no position and short > long it's buy. Let's test hold: short < long, no position."""
        bot = make_bot(short=3, long=5)
        # Prices trending down → short < long, no position → hold
        bot.strategy.price_history = [50.0, 40.0, 30.0, 20.0]
        bot.has_position = False
        signal = bot._run_strategy_logic(10.0)
        # history = [50, 40, 30, 20, 10]
        # short MA = mean(30, 20, 10) = 20
        # long MA = mean(50, 40, 30, 20, 10) = 30
        # short < long, no position → hold
        assert signal == 'hold'

    def test_hold_when_holding_and_short_above_long(self):
        """Holding and short > long → hold (no sell signal)."""
        bot = make_bot(short=3, long=5)
        bot.strategy.price_history = [10.0, 10.0, 10.0, 10.0]
        bot.has_position = True
        bot.entry_price = 10.0
        bot.highest_price = 10.0
        signal = bot._run_strategy_logic(50.0)
        # short MA > long MA, has position → hold
        assert signal == 'hold'


# ---------------------------------------------------------------------------
# Stop-loss
# ---------------------------------------------------------------------------

class TestTrailingStopLoss:
    def test_trailing_stop_triggers_sell(self):
        bot = make_bot(short=3, long=5, trailing_stop_pct=0.05)
        # Build history with high prices so MAs won't trigger a sell by crossover
        bot.strategy.price_history = [100.0, 100.0, 100.0, 100.0]
        bot.has_position = True
        bot._handle_position_entry(100.0)

        # Price rises to 110 → updates highest and trailing stop
        signal = bot._run_strategy_logic(110.0)
        assert signal == 'hold'
        assert bot.highest_price == 110.0
        assert bot.stop_loss_price == pytest.approx(110.0 * 0.95)

        # Price drops to just above the stop → hold
        signal = bot._run_strategy_logic(104.6)
        assert signal == 'hold'

        # Price drops below the trailing stop (110 * 0.95 = 104.5) → sell
        signal = bot._run_strategy_logic(104.0)
        assert signal == ('sell', 'trailing_sl_hit')


class TestFixedStopLoss:
    def test_fixed_stop_triggers_sell(self):
        bot = make_bot(short=3, long=5, stop_loss_pct=0.10)
        # Use rising history so short MA stays above long MA (no MA crossover sell)
        bot.strategy.price_history = [80.0, 90.0, 100.0, 110.0]
        bot.has_position = True
        bot._handle_position_entry(100.0)
        # Fixed stop = 100 * 0.90 = 90.0

        # Price at 95 → above 90 stop, short MA > long MA → hold
        signal = bot._run_strategy_logic(95.0)
        assert signal == 'hold'

        # Price at 89 → below 90 stop → sell
        signal = bot._run_strategy_logic(89.0)
        assert signal == ('sell', 'fixed_sl_hit')


# ---------------------------------------------------------------------------
# Position entry / exit
# ---------------------------------------------------------------------------

class TestPositionTracking:
    def test_handle_position_entry_sets_fields(self):
        bot = make_bot(trailing_stop_pct=0.05)
        bot._handle_position_entry(200.0)
        assert bot.entry_price == 200.0
        assert bot.highest_price == 200.0
        assert bot.stop_loss_price == pytest.approx(200.0 * 0.95)

    def test_handle_position_entry_without_trailing(self):
        bot = make_bot()
        bot._handle_position_entry(200.0)
        assert bot.entry_price == 200.0
        assert bot.highest_price == 200.0
        assert bot.stop_loss_price is None

    def test_handle_position_exit_resets_fields(self):
        bot = make_bot(trailing_stop_pct=0.05)
        bot._handle_position_entry(200.0)
        bot._handle_position_exit()
        assert bot.entry_price is None
        assert bot.highest_price is None
        assert bot.stop_loss_price is None


# ---------------------------------------------------------------------------
# Bar-level stop-loss logic (_run_strategy_logic_bar)
# ---------------------------------------------------------------------------

def _make_atr_bot(trailing_stop_pct=None, stop_loss_pct=None):
    """Create a bot with ATRBreakoutStrategy for bar-level tests."""
    strategy = ATRBreakoutStrategy()
    bot = make_bot(trailing_stop_pct=trailing_stop_pct, stop_loss_pct=stop_loss_pct, strategy=strategy)
    # Seed strategy with enough warmup data so it won't interfere
    for i in range(strategy.warmup_period + 1):
        strategy._highs.append(100 + i * 0.1)
        strategy._lows.append(99 + i * 0.1)
        strategy._closes.append(100 + i * 0.1)
    return bot


class TestBarTrailingStopLong:
    def test_trailing_stop_long_triggers_sell(self):
        """Long position: price rises then drops below trailing stop → sell."""
        bot = _make_atr_bot(trailing_stop_pct=0.05)
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='long')

        # Price rises to 110 → updates highest and trailing stop
        bar = {'Open': 109, 'High': 111, 'Low': 108, 'Close': 110, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar)
        assert bot.highest_price == 110.0
        assert bot.stop_loss_price == pytest.approx(110.0 * 0.95)
        assert signal != 'sell'

        # Price drops below trailing stop (110 * 0.95 = 104.5) → sell
        bar2 = {'Open': 105, 'High': 106, 'Low': 103, 'Close': 104, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar2)
        assert signal == ('sell', 'trailing_sl_hit')


class TestBarTrailingStopShort:
    def test_trailing_stop_short_triggers_sell(self):
        """Short position: price drops then rises above trailing stop → sell."""
        bot = _make_atr_bot(trailing_stop_pct=0.05)
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='short')
        # Initial stop at 100 * 1.05 = 105

        # Price drops to 90 → updates lowest and trailing stop
        bar = {'Open': 91, 'High': 92, 'Low': 89, 'Close': 90, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar)
        assert bot.highest_price == 90.0  # tracks lowest for shorts
        assert bot.stop_loss_price == pytest.approx(90.0 * 1.05)
        assert signal != 'sell'

        # Price rises above stop (90 * 1.05 = 94.5) → sell
        bar2 = {'Open': 94, 'High': 96, 'Low': 93, 'Close': 95, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar2)
        assert signal == ('sell', 'trailing_sl_hit')


class TestBarFixedStop:
    def test_fixed_stop_long_triggers_sell(self):
        """Long position: price drops below fixed stop → sell."""
        bot = _make_atr_bot(stop_loss_pct=0.10)
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='long')
        # Fixed stop = 100 * 0.90 = 90

        bar = {'Open': 89, 'High': 91, 'Low': 88, 'Close': 89, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar)
        assert signal == ('sell', 'fixed_sl_hit')

    def test_fixed_stop_short_triggers_sell(self):
        """Short position: price rises above fixed stop → sell."""
        bot = _make_atr_bot(stop_loss_pct=0.10)
        bot.has_position = True
        bot._handle_position_entry(100.0, position_type='short')
        # Fixed stop = 100 * 1.10 = 110

        bar = {'Open': 111, 'High': 112, 'Low': 110, 'Close': 111, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar)
        assert signal == ('sell', 'fixed_sl_hit')


class TestBarDelegatesToStrategy:
    def test_no_position_delegates_to_strategy(self):
        """Without a position, bar logic delegates to strategy.on_bar()."""
        bot = _make_atr_bot(trailing_stop_pct=0.05)
        bot.has_position = False

        bar = {'Open': 100, 'High': 105, 'Low': 95, 'Close': 100, 'Volume': 1000}
        signal = bot._run_strategy_logic_bar(bar)
        # Should return whatever the strategy returns (hold/buy/short), not 'sell'
        assert signal in ('hold', 'buy', 'short')


# ---------------------------------------------------------------------------
# Live trading path (run_strategy + exchange)
# ---------------------------------------------------------------------------


class TestRunStrategy:
    @patch("src.bot.place_order", return_value=True)
    @patch("src.bot.fetch_price", return_value=50.0)
    def test_buy_signal_places_order(self, mock_fetch, mock_order):
        """Buy signal → place_order('buy') is called, has_position becomes True."""
        bot = make_bot(short=3, long=5)
        bot.strategy.price_history = [10.0, 10.0, 10.0, 10.0]
        bot.has_position = False

        bot.run_strategy()

        mock_order.assert_called_once_with("TEST", "buy", 1)
        assert bot.has_position is True

    @patch("src.bot.place_order", return_value=True)
    @patch("src.bot.fetch_price", return_value=5.0)
    def test_sell_signal_places_order(self, mock_fetch, mock_order):
        """Sell signal → place_order('sell') is called, has_position becomes False."""
        bot = make_bot(short=3, long=5)
        bot.strategy.price_history = [50.0, 50.0, 50.0, 50.0]
        bot.has_position = True
        bot.entry_price = 50.0
        bot.highest_price = 50.0

        bot.run_strategy()

        mock_order.assert_called_once_with("TEST", "sell", 1)
        assert bot.has_position is False

    @patch("src.bot.place_order")
    @patch("src.bot.fetch_price", return_value=None)
    def test_none_price_skips_cycle(self, mock_fetch, mock_order):
        """fetch_price returning None → no order placed, no crash."""
        bot = make_bot(short=3, long=5)
        bot.run_strategy()
        mock_order.assert_not_called()


class TestExchange:
    def test_fetch_price_returns_float(self):
        from src.exchange import fetch_price
        price = fetch_price("TEST")
        assert isinstance(price, float)
        assert 95 <= price <= 105  # 100 ± 5

    def test_place_order_returns_true(self):
        from src.exchange import place_order
        assert place_order("TEST", "buy", 1) is True
        assert place_order("TEST", "sell", 1) is True
