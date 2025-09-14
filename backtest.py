import logging
import pandas as pd
import yfinance as yf
from bot_logic import TradingBot, LONG_WINDOW

def configure_logging():
    """
    Configures logging for the backtesting application.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_backtest(symbol, trade_amount, start_date, end_date):
    """
    Runs a backtest of the trading strategy over a given period.
    """
    configure_logging()
    logging.info(f"Starting backtest for {symbol} from {start_date} to {end_date}...")

    # --- 1. Data Fetching ---
    # Fetch all data for the backtest period at once.
    try:
        data = yf.download(tickers=symbol, start=start_date, end=end_date, interval="1d", auto_adjust=True)
        if data.empty:
            logging.error("No data fetched for the given date range. Aborting backtest.")
            return
    except Exception as e:
        logging.error(f"Failed to fetch historical data for backtest: {e}")
        return

    # --- 2. Initialization ---
    # Initialize the trading bot
    bot = TradingBot(symbol, trade_amount)

    # Prime the bot's price history with the initial window of data.
    # The simulation will start after this initial period.
    initial_history_data = data.head(LONG_WINDOW)
    bot.load_historical_data(data=initial_history_data)
    
    # The rest of the data is used for the simulation loop
    simulation_data = data.iloc[LONG_WINDOW:]

    # --- 3. Backtesting Simulation ---
    trades = []
    
    # Simulate day-by-day
    for index, row in simulation_data.iterrows():
        current_price = float(row['Close'])  # Convert to float explicitly
        
        # Run the strategy logic with the historical price
        signal = bot._run_strategy_logic(current_price)

        # --- Simulate Trade Execution ---
        if signal == 'buy' and not bot.has_position:
            # Simulate buying
            bot.has_position = True
            trades.append({'date': index.date(), 'type': 'buy', 'price': current_price})
            logging.info(f"Simulating BUY at {current_price:.2f} on {index.date()}")

        elif signal == 'sell' and bot.has_position:
            # Simulate selling
            bot.has_position = False
            trades.append({'date': index.date(), 'type': 'sell', 'price': current_price})
            logging.info(f"Simulating SELL at {current_price:.2f} on {index.date()}")

    logging.info("Backtest simulation finished.")
    
    # --- 4. Analyze Results ---
    if not trades:
        logging.info("No trades were executed during the backtest.")
        return

    pnl = 0
    wins = 0
    losses = 0
    last_buy_price = 0
    trade_count = 0

    for i in range(len(trades)):
        trade = trades[i]
        if trade['type'] == 'buy':
            if last_buy_price == 0: # Only record the first buy in a sequence
                last_buy_price = trade['price']
        elif trade['type'] == 'sell' and last_buy_price > 0:
            profit = trade['price'] - last_buy_price
            pnl += profit
            if profit > 0:
                wins += 1
            else:
                losses += 1
            last_buy_price = 0 # Reset after a sell
            trade_count +=1


    logging.info("--- Backtest Results ---")
    logging.info(f"Total PnL: {pnl:.2f}")
    logging.info(f"Completed Trades (Buy/Sell pairs): {trade_count}")
    if trade_count > 0:
        logging.info(f"Win/Loss Ratio: {wins}/{losses}")
        logging.info(f"Win Rate: {(wins/trade_count)*100:.2f}%")


if __name__ == "__main__":
    # --- Configuration ---
    TEST_SYMBOL = "BTC-USD"
    TRADE_AMOUNT = 1
    START_DATE = "2023-01-01"
    END_DATE = "2023-12-31"

    run_backtest(TEST_SYMBOL, TRADE_AMOUNT, START_DATE, END_DATE)

