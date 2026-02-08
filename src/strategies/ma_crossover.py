import logging

import pandas as pd

from src.strategies.base import BaseStrategy

SHORT_WINDOW = 5
LONG_WINDOW = 20


class MACrossoverStrategy(BaseStrategy):
    """Moving-average crossover strategy.

    Generates a *buy* signal when the short MA crosses above the long MA and
    there is no open position, and a *sell* signal when the short MA crosses
    below the long MA while a position is held.
    """

    name = "MA Crossover"

    def __init__(self, short_window: int = SHORT_WINDOW, long_window: int = LONG_WINDOW):
        self.short_window = short_window
        self.long_window = long_window
        self.price_history: list[float] = []

    # --- BaseStrategy interface ---------------------------------------------------

    def on_price(self, price: float, has_position: bool) -> str:
        price = float(price)

        self.price_history.append(price)
        if len(self.price_history) > self.long_window:
            self.price_history.pop(0)

        short_ma, long_ma = self.calculate_moving_averages()
        if short_ma is None or long_ma is None:
            logging.info("Not enough data to calculate moving averages. Waiting for more data...")
            return 'hold'

        logging.info(f"Current Price: {price:.2f}, Short MA: {short_ma:.2f}, Long MA: {long_ma:.2f}")

        if not has_position and short_ma > long_ma:
            logging.info("Signal: Buy")
            return 'buy'
        elif has_position and short_ma < long_ma:
            logging.info("Signal: Sell (MA crossover)")
            return 'sell'
        else:
            logging.info("Signal: Hold")
            return 'hold'

    def reset(self) -> None:
        self.price_history = []

    def load_price_history(self, prices: list[float]) -> None:
        if len(prices) < self.long_window:
            self.price_history = list(prices)
        else:
            self.price_history = list(prices)[-self.long_window:]
        logging.info(f"Strategy loaded {len(self.price_history)} historical prices.")

    # --- Helpers ------------------------------------------------------------------

    def calculate_moving_averages(self):
        if len(self.price_history) < self.long_window:
            logging.warning("Not enough data in price history to calculate moving averages.")
            return None, None

        prices = pd.Series(self.price_history)
        short_ma = prices.rolling(window=self.short_window).mean().iloc[-1]
        long_ma = prices.rolling(window=self.long_window).mean().iloc[-1]
        return float(short_ma), float(long_ma)
