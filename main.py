import time
import logging
from bot_logic import TradingBot
from utils import configure_logging

def main():
    """
    Main function to configure and run the bot.
    """
    # --- Configuration ---
    SYMBOL = 'BTC-USD'  # Correct ticker for Bitcoin/USD on Yahoo Finance
    SHORT_WINDOW = 10         # Short-term moving average window
    LONG_WINDOW = 50          # Long-term moving average window
    TRADE_AMOUNT = 0.01       # Amount of the asset to trade
    RUN_INTERVAL_SECONDS = 60 # How often to run the strategy

    configure_logging()
    bot = TradingBot(
        symbol=SYMBOL,
        trade_amount=TRADE_AMOUNT,
        short_window=SHORT_WINDOW,
        long_window=LONG_WINDOW
    )

    bot.load_historical_data()  # Manually load historical data

    logging.info(f"Starting trading bot for {SYMBOL}...")
    while True:
        try:
            bot.run_strategy()
            logging.info(f"Sleeping for {RUN_INTERVAL_SECONDS} seconds...")
            time.sleep(RUN_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logging.info("Bot stopped by user.")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}")
            time.sleep(RUN_INTERVAL_SECONDS)  # Wait before retrying


if __name__ == "__main__":
    main()