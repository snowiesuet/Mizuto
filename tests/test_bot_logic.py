import pytest
from src.bot import TradingBot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bot(short=5, long=20, trailing_stop_pct=None, stop_loss_pct=None):
    """Create a TradingBot with deterministic settings (no network calls)."""
    return TradingBot(
        symbol="TEST",
        trade_amount=1,
        short_window=short,
        long_window=long,
        trailing_stop_pct=trailing_stop_pct,
        stop_loss_pct=stop_loss_pct,
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
        assert signal == 'sell'

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
        assert signal == 'sell'


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
        assert signal == 'sell'


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
