import logging
import pandas as pd
import yfinance as yf
from src.bot import TradingBot, LONG_WINDOW, SHORT_WINDOW
from src.utils import configure_logging


def run_backtest_on_data(data, symbol="SYN", trade_amount=1.0,
                         short_window=SHORT_WINDOW, long_window=LONG_WINDOW,
                         trailing_stop_pct=None, stop_loss_pct=None,
                         slippage_pct=0.001, commission_pct=0.001, quiet=False,
                         strategy=None):
    """
    Run a backtest on a provided DataFrame (no yfinance download).

    Args:
        data: DataFrame with at least a 'Close' column, indexed by date.
              OHLCV strategies require 'Open', 'High', 'Low', 'Close' columns.
        symbol: Symbol name (for logging only).
        trade_amount: Units per trade.
        short_window: Short MA window.
        long_window: Long MA window.
        trailing_stop_pct: Trailing stop percentage (None to disable).
        stop_loss_pct: Fixed stop-loss percentage (None to disable).
        slippage_pct: Slippage fraction per trade.
        commission_pct: Commission fraction per trade.
        quiet: If True, suppress logging output (useful for batch runs).
        strategy: A BaseStrategy instance (default None = MACrossover).

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
            strategy=strategy,
        )
    finally:
        if quiet and prev_level is not None:
            logging.root.setLevel(prev_level)


def _run_backtest_on_data_inner(data, symbol, trade_amount, short_window, long_window,
                                trailing_stop_pct, stop_loss_pct, slippage_pct, commission_pct,
                                strategy=None):
    """Inner implementation — separated so quiet-mode logging restore always runs."""

    if data.empty:
        logging.warning("Empty DataFrame passed to run_backtest_on_data.")
        return None

    # --- Initialization ---
    bot = TradingBot(symbol, trade_amount,
                     short_window=short_window, long_window=long_window,
                     trailing_stop_pct=trailing_stop_pct, stop_loss_pct=stop_loss_pct,
                     strategy=strategy)

    uses_ohlcv = bot.strategy.requires_ohlcv

    # Validate OHLCV columns when strategy requires them
    if uses_ohlcv:
        required_cols = {'Open', 'High', 'Low', 'Close'}
        missing = required_cols - set(data.columns)
        if missing:
            logging.error(f"Strategy requires OHLCV data but columns missing: {missing}")
            return None

    # Dynamic warmup period
    warmup = getattr(bot.strategy, 'warmup_period', long_window)
    warmup = max(warmup, long_window) if not uses_ohlcv else warmup

    initial_history_data = data.head(warmup)
    bot.load_historical_data(data=initial_history_data)

    simulation_data = data.iloc[warmup:]

    # --- Backtesting Simulation ---
    trades = []

    for index, row in simulation_data.iterrows():
        current_price = float(row['Close'])

        # Dispatch to bar-based or price-based logic
        if uses_ohlcv:
            bar = {
                'Open': float(row['Open']),
                'High': float(row['High']),
                'Low': float(row['Low']),
                'Close': current_price,
                'Volume': float(row['Volume']) if 'Volume' in row.index else 0,
            }
            signal = bot._run_strategy_logic_bar(bar)
        else:
            signal = bot._run_strategy_logic(current_price)

        if signal == 'buy' and not bot.has_position:
            fill_price = current_price * (1 + slippage_pct)
            bot.has_position = True
            bot._handle_position_entry(fill_price, position_type='long')
            trades.append({'date': index.date() if hasattr(index, 'date') else index,
                           'type': 'buy', 'price': fill_price})
            logging.info(f"Simulating BUY at {fill_price:.2f} (market {current_price:.2f}) on {index}")

        elif signal == 'short' and not bot.has_position:
            fill_price = current_price * (1 - slippage_pct)  # slippage unfavorable for short entry
            bot.has_position = True
            bot._handle_position_entry(fill_price, position_type='short')
            trades.append({'date': index.date() if hasattr(index, 'date') else index,
                           'type': 'short', 'price': fill_price})
            logging.info(f"Simulating SHORT at {fill_price:.2f} (market {current_price:.2f}) on {index}")

        elif signal == 'sell' and bot.has_position:
            # Slippage direction depends on position type
            if bot.position_type == 'short':
                fill_price = current_price * (1 + slippage_pct)  # unfavorable for short exit (buying back)
            else:
                fill_price = current_price * (1 - slippage_pct)  # unfavorable for long exit (selling)
            closing_position_type = bot.position_type
            bot.has_position = False
            bot._handle_position_exit()
            trades.append({'date': index.date() if hasattr(index, 'date') else index,
                           'type': 'sell', 'price': fill_price,
                           'closed_position': closing_position_type})
            logging.info(f"Simulating SELL ({closing_position_type}) at {fill_price:.2f} (market {current_price:.2f}) on {index}")

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
    last_entry_price = 0
    last_entry_type = None  # 'buy' or 'short'
    trade_count = 0

    for trade in trades:
        commission = trade['price'] * commission_pct
        total_commission += commission

        if trade['type'] in ('buy', 'short'):
            if last_entry_price == 0:
                last_entry_price = trade['price']
                last_entry_type = trade['type']
        elif trade['type'] == 'sell' and last_entry_price > 0:
            entry_commission = last_entry_price * commission_pct
            exit_commission = trade['price'] * commission_pct

            if last_entry_type == 'short':
                # Short PnL: entry_price - exit_price - commissions
                profit = last_entry_price - trade['price'] - entry_commission - exit_commission
            else:
                # Long PnL: exit_price - entry_price - commissions
                profit = trade['price'] - last_entry_price - entry_commission - exit_commission

            pnl += profit
            if profit > 0:
                wins += 1
                gross_profits += profit
            else:
                losses += 1
                gross_losses += abs(profit)
            last_entry_price = 0
            last_entry_type = None
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
