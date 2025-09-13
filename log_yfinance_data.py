import logging
import yfinance as yf

def configure_logging():
    """
    Configures logging for the application.
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_and_log_data(symbol, period="60d", interval="1h"):
    """
    Fetches historical data from yfinance and logs the results.
    """
    logging.info(f"Fetching historical data for {symbol} with period={period} and interval={interval}...")
    try:
        # Fetch historical data
        data = yf.download(tickers=symbol, period=period, interval=interval, auto_adjust=True)
        
        # Log the raw data
        logging.debug(f"Raw data fetched from yfinance:\n{data.head()}")
        logging.debug(f"Data shape: {data.shape}")

        if data.empty:
            logging.warning("No valid historical data retrieved.")
        else:
            # Log the 'Close' column
            if 'Close' in data.columns:
                close_prices = data['Close'].dropna()
                logging.info(f"Number of Close prices: {len(close_prices)}")
                logging.debug(f"Close prices:\n{close_prices}")
            else:
                logging.warning("'Close' column not found in the data.")
    except Exception as e:
        logging.error(f"Failed to fetch historical data: {e}")

if __name__ == "__main__":
    configure_logging()
    # Replace 'BTC-USD' with the ticker you want to test
    fetch_and_log_data(symbol="BTC-USD", period="60d", interval="1h")