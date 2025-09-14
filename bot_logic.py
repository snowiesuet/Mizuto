import logging
from exchange import fetch_price, place_order  # Import place_order here
import pandas as pd
import yfinance as yf

SHORT_WINDOW = 5         # Short-term moving average window
LONG_WINDOW = 20         # Long-term moving average window

class TradingBot:
    def __init__(self, symbol, trade_amount, short_window=SHORT_WINDOW, long_window=LONG_WINDOW):
        logging.info("TradingBot initialized from bot_logic.py - DEBUG TEST")
        self.symbol = symbol
        self.trade_amount = trade_amount
        self.short_window = short_window
        self.long_window = long_window
        self.has_position = False  # Tracks if we currently hold the asset
        self.price_history = []  # Stores recent prices for calculating moving averages

        # Historical data is now loaded manually, not in the constructor.

    def load_historical_data(self, data=None):
        """
        Initializes price history from yfinance data or a provided DataFrame.
        """
        if data is None:
            logging.info(f"Fetching historical data for {self.symbol}...")
            try:
                # Adjust period and interval based on Yahoo Finance limitations
                data = yf.download(tickers=self.symbol, period="180d", interval="1d", auto_adjust=True)
            except Exception as e:
                logging.error(f"Failed to fetch historical data: {e}")
                data = pd.DataFrame() # Ensure data is an empty DataFrame on failure
        else:
            logging.info("Loading historical data from provided DataFrame.")

        # Log the raw data for debugging
        logging.debug(f"Raw data for price history initialization:\n{data.head()}")
        logging.debug(f"Data shape: {data.shape}")

        if data.empty:
            logging.warning("No valid historical data provided or retrieved. Starting with an empty price history.")
            return

        # Ensure the 'Close' column exists and is valid
        if 'Close' in data.columns:
            # Use .values.flatten().tolist() to get a flat list of numerical values
            close_prices = data['Close'].dropna().values.flatten().tolist()
            logging.debug(f"Close prices after dropping NaN and converting to list: {close_prices[:5]}") # Log first 5
            logging.debug(f"Number of Close prices: {len(close_prices)}")

            if close_prices:
                # Convert 'Close' column to a list of numeric values
                if len(close_prices) < self.long_window:
                    self.price_history = list(close_prices)
                else:
                    self.price_history = list(close_prices)[-self.long_window:]
                logging.info(f"Loaded {len(self.price_history)} historical prices into price history.")
                logging.debug(f"Contents of self.price_history after slicing: {self.price_history}")
                # If fewer than long_window prices are available, log a warning
                if len(self.price_history) < self.long_window:
                    logging.warning(f"Only {len(self.price_history)} prices available, but {self.long_window} are required.")
            else:
                logging.warning("The 'Close' column is empty after dropping NaN values. Starting with an empty price history.")
        else:
            logging.warning("'Close' column not found in the historical data. Starting with an empty price history.")

    def calculate_moving_averages(self):
        """
        Calculates the short-term and long-term moving averages.
        """
        if len(self.price_history) < self.long_window:
            logging.warning("Not enough data in price history to calculate moving averages.")
            return None, None  # Not enough data to calculate moving averages

        prices = pd.Series(self.price_history)
        short_ma = prices.rolling(window=self.short_window).mean().iloc[-1]
        long_ma = prices.rolling(window=self.long_window).mean().iloc[-1]
        return float(short_ma), float(long_ma)

    def _run_strategy_logic(self, current_price):
        """
        Contains the core logic for the trading strategy.
        This method is designed to be called by both live trading and backtesting.
        """
        # Convert current_price to float if it's not already
        current_price = float(current_price)
        
        # Add the current price to the price history
        self.price_history.append(current_price)
        if len(self.price_history) > self.long_window:
            self.price_history.pop(0)  # Keep the history size manageable

        # Calculate moving averages
        short_ma, long_ma = self.calculate_moving_averages()
        if short_ma is None or long_ma is None:
            logging.info("Not enough data to calculate moving averages. Waiting for more data...")
            return 'hold' # Return hold signal

        logging.info(f"Current Price: {current_price:.2f}, Short MA: {short_ma:.2f}, Long MA: {long_ma:.2f}")

        # --- Strategy Logic ---
        if not self.has_position and short_ma > long_ma:
            # Buy signal: Short MA crosses above Long MA
            logging.info(f"Signal: Buy")
            return 'buy'

        elif self.has_position and short_ma < long_ma:
            # Sell signal: Short MA crosses below Long MA
            logging.info(f"Signal: Sell")
            return 'sell'
        else:
            logging.info("Signal: Hold")
            return 'hold'

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
        elif signal == 'sell':
            if place_order(self.symbol, 'sell', self.trade_amount):
                self.has_position = False


def configure_logging():
    """
    Configures logging for the application.
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')