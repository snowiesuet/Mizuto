import logging
from src.exchange import fetch_price, place_order
import pandas as pd
import yfinance as yf

from src.strategies.ma_crossover import MACrossoverStrategy

SHORT_WINDOW = 5         # Short-term moving average window (kept for backward compat imports)
LONG_WINDOW = 20         # Long-term moving average window

class TradingBot:
    def __init__(self, symbol, trade_amount, short_window=SHORT_WINDOW, long_window=LONG_WINDOW,
                 trailing_stop_pct=None, stop_loss_pct=None, strategy=None):
        logging.info("TradingBot initialized from bot_logic.py - DEBUG TEST")
        self.symbol = symbol
        self.trade_amount = trade_amount
        self.short_window = short_window
        self.long_window = long_window
        self.has_position = False  # Tracks if we currently hold the asset
        self.position_type = None  # 'long', 'short', or None

        # Strategy (defaults to MA Crossover for backward compatibility)
        if strategy is not None:
            self.strategy = strategy
        else:
            self.strategy = MACrossoverStrategy(short_window=short_window, long_window=long_window)

        # Stop-loss parameters (infrastructure — stays in TradingBot)
        self.trailing_stop_pct = trailing_stop_pct  # Trailing stop percentage (e.g., 0.05 for 5%)
        self.stop_loss_pct = stop_loss_pct  # Fixed stop loss percentage
        self.entry_price = None  # Price at which position was entered
        self.highest_price = None  # Highest price since entering position (for trailing stop)
        self.stop_loss_price = None  # Current stop loss price

    def load_historical_data(self, data=None):
        """
        Initializes price history from yfinance data or a provided DataFrame.
        """
        if data is None:
            logging.info(f"Fetching historical data for {self.symbol}...")
            try:
                data = yf.download(tickers=self.symbol, period="180d", interval="1d", auto_adjust=True)
            except Exception as e:
                logging.error(f"Failed to fetch historical data: {e}")
                data = pd.DataFrame()
        else:
            logging.info("Loading historical data from provided DataFrame.")

        logging.debug(f"Raw data for price history initialization:\n{data.head()}")
        logging.debug(f"Data shape: {data.shape}")

        if data.empty:
            logging.warning("No valid historical data provided or retrieved. Starting with an empty price history.")
            return

        # OHLCV strategies get the full DataFrame
        if self.strategy.requires_ohlcv:
            self.strategy.load_ohlcv_history(data)
            return

        if 'Close' in data.columns:
            close_prices = data['Close'].dropna().values.flatten().tolist()
            logging.debug(f"Close prices after dropping NaN and converting to list: {close_prices[:5]}")
            logging.debug(f"Number of Close prices: {len(close_prices)}")

            if close_prices:
                self.strategy.load_price_history(close_prices)
                if len(close_prices) < self.long_window:
                    logging.warning(f"Only {len(close_prices)} prices available, but {self.long_window} are required.")
            else:
                logging.warning("The 'Close' column is empty after dropping NaN values. Starting with an empty price history.")
        else:
            logging.warning("'Close' column not found in the historical data. Starting with an empty price history.")

    def _run_strategy_logic(self, current_price):
        """
        Contains the core logic for the trading strategy.
        This method is designed to be called by both live trading and backtesting.
        """
        current_price = float(current_price)

        # --- Check for trailing stop-loss if we have a position ---
        if self.has_position and self.trailing_stop_pct is not None:
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
                self.stop_loss_price = self.highest_price * (1 - self.trailing_stop_pct)
                logging.debug(f"Updated trailing stop: highest_price={self.highest_price:.2f}, stop_loss_price={self.stop_loss_price:.2f}")

            if current_price <= self.stop_loss_price:
                logging.info(f"Trailing stop-loss triggered! Price {current_price:.2f} <= Stop {self.stop_loss_price:.2f}")
                return 'sell'

        # --- Check for fixed stop-loss if we have a position ---
        if self.has_position and self.stop_loss_pct is not None and self.entry_price is not None:
            fixed_stop_price = self.entry_price * (1 - self.stop_loss_pct)
            if current_price <= fixed_stop_price:
                logging.info(f"Fixed stop-loss triggered! Price {current_price:.2f} <= Stop {fixed_stop_price:.2f}")
                return 'sell'

        # --- Delegate signal generation to the strategy ---
        return self.strategy.on_price(current_price, self.has_position)

    def _run_strategy_logic_bar(self, bar: dict):
        """Core logic for strategies that need full OHLCV bars.

        Performs bot-level stop-loss checks (supporting both long and short
        positions) then delegates to ``strategy.on_bar()``.
        """
        current_price = float(bar['Close'])

        # --- Bot-level trailing stop-loss ---
        if self.has_position and self.trailing_stop_pct is not None:
            if self.position_type == 'short':
                # For shorts, track lowest price and stop triggers on rise
                if self.highest_price is None or current_price < self.highest_price:
                    self.highest_price = current_price
                    self.stop_loss_price = self.highest_price * (1 + self.trailing_stop_pct)
                if current_price >= self.stop_loss_price:
                    logging.info(f"Trailing stop-loss triggered (short)! Price {current_price:.2f} >= Stop {self.stop_loss_price:.2f}")
                    return 'sell'
            else:
                # Long position (original behavior)
                if self.highest_price is None or current_price > self.highest_price:
                    self.highest_price = current_price
                    self.stop_loss_price = self.highest_price * (1 - self.trailing_stop_pct)
                if current_price <= self.stop_loss_price:
                    logging.info(f"Trailing stop-loss triggered! Price {current_price:.2f} <= Stop {self.stop_loss_price:.2f}")
                    return 'sell'

        # --- Bot-level fixed stop-loss ---
        if self.has_position and self.stop_loss_pct is not None and self.entry_price is not None:
            if self.position_type == 'short':
                fixed_stop_price = self.entry_price * (1 + self.stop_loss_pct)
                if current_price >= fixed_stop_price:
                    logging.info(f"Fixed stop-loss triggered (short)! Price {current_price:.2f} >= Stop {fixed_stop_price:.2f}")
                    return 'sell'
            else:
                fixed_stop_price = self.entry_price * (1 - self.stop_loss_pct)
                if current_price <= fixed_stop_price:
                    logging.info(f"Fixed stop-loss triggered! Price {current_price:.2f} <= Stop {fixed_stop_price:.2f}")
                    return 'sell'

        # --- Delegate to strategy ---
        return self.strategy.on_bar(bar, self.has_position, self.position_type)

    def _handle_position_entry(self, entry_price, position_type='long'):
        """
        Updates position tracking variables when entering a position.
        Call this method after a successful buy or short order.
        """
        self.entry_price = float(entry_price)
        self.highest_price = float(entry_price)
        self.position_type = position_type
        if self.trailing_stop_pct is not None:
            if position_type == 'short':
                self.stop_loss_price = self.entry_price * (1 + self.trailing_stop_pct)
            else:
                self.stop_loss_price = self.entry_price * (1 - self.trailing_stop_pct)
        logging.info(f"Position entered at {self.entry_price:.2f} ({position_type})")

    def _handle_position_exit(self):
        """
        Resets position tracking variables when exiting a position.
        Call this method after a successful sell order.
        """
        self.entry_price = None
        self.highest_price = None
        self.stop_loss_price = None
        self.position_type = None
        logging.info("Position exited - stop-loss tracking reset")

    def run_strategy(self):
        """
        Executes the trading strategy for live trading.
        """
        logging.info("Running trading strategy...")
        current_price = fetch_price(self.symbol)

        if current_price is None:
            logging.warning("Could not retrieve price. Skipping this cycle.")
            return

        signal = self._run_strategy_logic(current_price)

        if signal == 'buy':
            if place_order(self.symbol, 'buy', self.trade_amount):
                self.has_position = True
                self._handle_position_entry(current_price)
        elif signal == 'sell':
            if place_order(self.symbol, 'sell', self.trade_amount):
                self.has_position = False
                self._handle_position_exit()

