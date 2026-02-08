import logging
import pandas as pd
import yfinance as yf
from src.bot import TradingBot, LONG_WINDOW, SHORT_WINDOW
from src.utils import configure_logging


def run_backtest_on_data(data, symbol="SYN", trade_amount=1.0,
                         short_window=SHORT_WINDOW, long_window=LONG_WINDOW,
                         trailing_stop_pct=None, stop_loss_pct=None,
                         slippage_pct=0.001, commission_pct=0.001, quiet=False):
    """
    Run a backtest on a provided DataFrame (no yfinance download).

    Args:
        data: DataFrame with at least a 'Close' column, indexed by date.
        symbol: Symbol name (for logging only).
        trade_amount: Units per trade.
        short_window: Short MA window.
        long_window: Long MA window.
        trailing_stop_pct: Trailing stop percentage (None to disable).
        stop_loss_pct: Fixed stop-loss percentage (None to disable).
        slippage_pct: Slippage fraction per trade.
        commission_pct: Commission fraction per trade.
        quiet: If True, suppress logging output (useful for batch runs).

    Returns:
        Dict with keys: 'pnl', 'profit_factor', 'total_commission',
                        'trade_count', 'wins', 'losses', 'trades',
                        'gross_profits', 'gross_losses'
        Returns None if no trades executed.
    """
    prev_level = None
    if quiet:
        prev_level = logging.root.level
        logging.root.setLevel(logging.WARNING)

    try:
        return _run_backtest_on_data_inner(
            data, symbol, trade_amount, short_window, long_window,
            trailing_stop_pct, stop_loss_pct, slippage_pct, commission_pct,
        )
    finally:
        if quiet and prev_level is not None:
            logging.root.setLevel(prev_level)


def _run_backtest_on_data_inner(data, symbol, trade_amount, short_window, long_window,
                                trailing_stop_pct, stop_loss_pct, slippage_pct, commission_pct):
    """Inner implementation — separated so quiet-mode logging restore always runs."""

    if data.empty:
        logging.warning("Empty DataFrame passed to run_backtest_on_data.")
        return None

    # --- Initialization ---
    bot = TradingBot(symbol, trade_amount,
                     short_window=short_window, long_window=long_window,
                     trailing_stop_pct=trailing_stop_pct, stop_loss_pct=stop_loss_pct)

    initial_history_data = data.head(long_window)
    bot.load_historical_data(data=initial_history_data)

    simulation_data = data.iloc[long_window:]

    # --- Backtesting Simulation ---
    trades = []

    for index, row in simulation_data.iterrows():
        current_price = float(row['Close'])
        signal = bot._run_strategy_logic(current_price)

        if signal == 'buy' and not bot.has_position:
            fill_price = current_price * (1 + slippage_pct)
            bot.has_position = True
            bot._handle_position_entry(fill_price)
            trades.append({'date': index.date() if hasattr(index, 'date') else index,
                           'type': 'buy', 'price': fill_price})
            logging.info(f"Simulating BUY at {fill_price:.2f} (market {current_price:.2f}) on {index}")

        elif signal == 'sell' and bot.has_position:
            fill_price = current_price * (1 - slippage_pct)
            bot.has_position = False
            bot._handle_position_exit()
            trades.append({'date': index.date() if hasattr(index, 'date') else index,
                           'type': 'sell', 'price': fill_price})
            logging.info(f"Simulating SELL at {fill_price:.2f} (market {current_price:.2f}) on {index}")

    logging.info("Backtest simulation finished.")

    # --- Analyze Results ---
    if not trades:
        logging.info("No trades were executed during the backtest.")
        return None

    pnl = 0
    total_commission = 0
    wins = 0
    losses = 0
    gross_profits = 0.0
    gross_losses = 0.0
    last_buy_price = 0
    trade_count = 0

    for trade in trades:
        commission = trade['price'] * commission_pct
        total_commission += commission

        if trade['type'] == 'buy':
            if last_buy_price == 0:
                last_buy_price = trade['price']
        elif trade['type'] == 'sell' and last_buy_price > 0:
            buy_commission = last_buy_price * commission_pct
            sell_commission = trade['price'] * commission_pct
            profit = trade['price'] - last_buy_price - buy_commission - sell_commission
            pnl += profit
            if profit > 0:
                wins += 1
                gross_profits += profit
            else:
                losses += 1
                gross_losses += abs(profit)
            last_buy_price = 0
            trade_count += 1

    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

    logging.info("--- Backtest Results ---")
    logging.info(f"Total PnL (after costs): {pnl:.2f}")
    logging.info(f"Profit Factor: {profit_factor:.4f}")
    logging.info(f"Total Commission Paid: {total_commission:.2f}")
    logging.info(f"Completed Trades (Buy/Sell pairs): {trade_count}")
    if trade_count > 0:
        logging.info(f"Win/Loss Ratio: {wins}/{losses}")
        logging.info(f"Win Rate: {(wins/trade_count)*100:.2f}%")

    return {
        'pnl': pnl,
        'profit_factor': profit_factor,
        'total_commission': total_commission,
        'trade_count': trade_count,
        'wins': wins,
        'losses': losses,
        'gross_profits': gross_profits,
        'gross_losses': gross_losses,
        'trades': trades,
    }


def run_backtest(symbol, trade_amount, start_date, end_date, trailing_stop_pct=None, stop_loss_pct=None,
                 slippage_pct=0.001, commission_pct=0.001):
    """
    Runs a backtest of the trading strategy over a given period.

    Args:
        symbol: Trading symbol (e.g., "BTC-USD")
        trade_amount: Amount to trade
        start_date: Start date for backtest
        end_date: End date for backtest
        trailing_stop_pct: Trailing stop-loss percentage (e.g., 0.05 for 5%)
        stop_loss_pct: Fixed stop-loss percentage (e.g., 0.10 for 10%)
        slippage_pct: Simulated slippage as a fraction (default 0.001 = 0.1%).
                      Buy price is adjusted up, sell price adjusted down.
        commission_pct: Commission fee as a fraction (default 0.001 = 0.1%).
                        Deducted from each trade.
    """
    configure_logging()
    logging.info(f"Starting backtest for {symbol} from {start_date} to {end_date}...")
    logging.info(f"Slippage: {slippage_pct*100:.2f}%, Commission: {commission_pct*100:.2f}%")
    if trailing_stop_pct:
        logging.info(f"Using trailing stop-loss: {trailing_stop_pct*100}%")
    if stop_loss_pct:
        logging.info(f"Using fixed stop-loss: {stop_loss_pct*100}%")

    # --- Data Fetching ---
    try:
        data = yf.download(tickers=symbol, start=start_date, end=end_date, interval="1d", auto_adjust=True)
        if data.empty:
            logging.error("No data fetched for the given date range. Aborting backtest.")
            return
    except Exception as e:
        logging.error(f"Failed to fetch historical data for backtest: {e}")
        return

    return run_backtest_on_data(
        data=data, symbol=symbol, trade_amount=trade_amount,
        trailing_stop_pct=trailing_stop_pct, stop_loss_pct=stop_loss_pct,
        slippage_pct=slippage_pct, commission_pct=commission_pct,
    )


if __name__ == "__main__":
    # --- Configuration ---
    TEST_SYMBOL = "BTC-USD"
    TRADE_AMOUNT = 1
    START_DATE = "2023-01-01"
    END_DATE = "2023-12-31"

    # Stop-loss configuration (set to None to disable)
    TRAILING_STOP_PCT = 0.05  # 5% trailing stop-loss
    STOP_LOSS_PCT = None      # Fixed stop-loss (disabled)

    run_backtest(TEST_SYMBOL, TRADE_AMOUNT, START_DATE, END_DATE,
                trailing_stop_pct=TRAILING_STOP_PCT, stop_loss_pct=STOP_LOSS_PCT)
