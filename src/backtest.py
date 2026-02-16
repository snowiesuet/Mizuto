import logging

import numpy as np
import pandas as pd
import yfinance as yf

from src.bot import TradingBot, LONG_WINDOW, SHORT_WINDOW
from src.utils import configure_logging


def _unpack_signal(raw):
    """Normalize strategy/bot return into (signal_type, exit_reason).

    Accepts bare strings (``'sell'``) or tuples (``('sell', 'sl_hit')``).
    """
    if isinstance(raw, tuple):
        return raw[0], raw[1]
    return raw, None


def _validate_ohlc_structure(data):
    """Warn on OHLC bars where High/Low don't bracket Open/Close.

    Does NOT reject data — some real data sources have minor violations.
    Returns the count of invalid bars for testing.
    """
    if not {'Open', 'High', 'Low', 'Close'}.issubset(data.columns):
        return 0

    high = data['High'].values
    low = data['Low'].values
    open_ = data['Open'].values
    close = data['Close'].values

    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)

    n_high = int(np.sum(high < bar_max))
    n_low = int(np.sum(low > bar_min))

    if n_high > 0:
        logging.warning(
            f"OHLC validation: {n_high} bar(s) have High < max(Open, Close)"
        )
    if n_low > 0:
        logging.warning(
            f"OHLC validation: {n_low} bar(s) have Low > min(Open, Close)"
        )

    return n_high + n_low


def run_backtest_on_data(data, symbol="SYN", trade_amount=1.0,
                         short_window=SHORT_WINDOW, long_window=LONG_WINDOW,
                         trailing_stop_pct=None, stop_loss_pct=None,
                         slippage_pct=0.001, commission_pct=0.001, quiet=False,
                         strategy=None, initial_capital=10000.0,
                         fill_model="close"):
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
        initial_capital: Starting portfolio value for equity curve tracking.
        fill_model: Order fill model - 'close' (default), 'next_open', or 'vwap_slippage'.

    Returns:
        Dict with keys: 'pnl', 'profit_factor', 'total_commission',
                        'trade_count', 'wins', 'losses', 'trades',
                        'gross_profits', 'gross_losses',
                        'equity_curve', 'equity_dates', 'initial_capital',
                        plus risk-adjusted metrics from src.metrics.
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
            strategy=strategy, initial_capital=initial_capital,
            fill_model=fill_model,
        )
    finally:
        if quiet and prev_level is not None:
            logging.root.setLevel(prev_level)


def _compute_fill_price(base_price, signal_type, slippage_pct, fill_model,
                        bar=None, median_volume=None):
    """Compute fill price based on fill model.

    Args:
        base_price: The base price for the fill (Close or next Open depending on model).
        signal_type: 'buy', 'short', or 'sell_long', 'sell_short'.
        slippage_pct: Base slippage percentage.
        fill_model: 'close', 'next_open', or 'vwap_slippage'.
        bar: Current bar dict (needed for vwap_slippage volume).
        median_volume: Median volume across dataset (for vwap_slippage).

    Returns:
        Fill price as float.
    """
    effective_slippage = slippage_pct

    if fill_model == 'vwap_slippage' and bar is not None and median_volume is not None:
        volume = bar.get('Volume', 0)
        if volume > 0:
            effective_slippage = slippage_pct * max(median_volume / volume, 1.0)

    if signal_type == 'buy':
        return base_price * (1 + effective_slippage)
    elif signal_type == 'short':
        return base_price * (1 - effective_slippage)
    elif signal_type == 'sell_short':
        return base_price * (1 + effective_slippage)
    else:  # sell_long
        return base_price * (1 - effective_slippage)


def _run_backtest_on_data_inner(data, symbol, trade_amount, short_window, long_window,
                                trailing_stop_pct, stop_loss_pct, slippage_pct, commission_pct,
                                strategy=None, initial_capital=10000.0,
                                fill_model="close"):
    """Inner implementation — separated so quiet-mode logging restore always runs."""

    if data.empty:
        logging.warning("Empty DataFrame passed to run_backtest_on_data.")
        return None

    valid_fill_models = ('close', 'next_open', 'vwap_slippage')
    if fill_model not in valid_fill_models:
        raise ValueError(f"Invalid fill_model '{fill_model}'. Must be one of {valid_fill_models}")

    # --- Initialization ---
    bot = TradingBot(symbol, trade_amount,
                     short_window=short_window, long_window=long_window,
                     trailing_stop_pct=trailing_stop_pct, stop_loss_pct=stop_loss_pct,
                     strategy=strategy)

    uses_ohlcv = bot.strategy.requires_ohlcv

    # Validate fill model compatibility with strategy
    if fill_model not in bot.strategy.supported_fill_models:
        logging.warning(
            f"Strategy '{bot.strategy.name}' does not declare support for "
            f"fill_model='{fill_model}'. Supported: {bot.strategy.supported_fill_models}. "
            f"Proceeding anyway."
        )

    # Validate OHLCV columns when strategy requires them
    if uses_ohlcv:
        required_cols = {'Open', 'High', 'Low', 'Close'}
        missing = required_cols - set(data.columns)
        if missing:
            logging.error(f"Strategy requires OHLCV data but columns missing: {missing}")
            return None

    # Validate OHLC structure (warn-only, never rejects data)
    if {'Open', 'High', 'Low', 'Close'}.issubset(data.columns):
        _validate_ohlc_structure(data)

    # Validate fill model requirements
    if fill_model == 'next_open' and 'Open' not in data.columns:
        logging.error("fill_model='next_open' requires 'Open' column in data.")
        return None

    # Dynamic warmup period — always enforce minimum of long_window
    warmup = max(bot.strategy.warmup_period, long_window)

    initial_history_data = data.head(warmup)
    bot.load_historical_data(data=initial_history_data)

    simulation_data = data.iloc[warmup:]

    # --- Equity Curve Tracking ---
    cash = initial_capital
    equity_curve = []
    equity_dates = []

    # Pre-compute median volume for vwap_slippage model
    median_volume = None
    if fill_model == 'vwap_slippage' and 'Volume' in simulation_data.columns:
        median_volume = float(simulation_data['Volume'].median())

    # --- Backtesting Simulation ---
    trades = []
    pending_signal = None  # For next_open fill model: (signal_type, exit_reason, bar)
    entry_bar_idx = None   # Bar index when position was opened (for bars_held)

    sim_rows = list(simulation_data.iterrows())
    for i, (index, row) in enumerate(sim_rows):
        current_price = float(row['Close'])

        # --- Execute pending signal from previous bar (next_open model) ---
        if fill_model == 'next_open' and pending_signal is not None:
            pending_sig, pending_exit_reason, pending_bar = pending_signal
            pending_signal = None
            open_price = float(row['Open'])

            if pending_sig == 'buy' and not bot.has_position:
                fill_price = _compute_fill_price(open_price, 'buy', slippage_pct, fill_model)
                bot.has_position = True
                bot._handle_position_entry(fill_price, position_type='long')
                entry_bar_idx = i
                trades.append({'date': index.date() if hasattr(index, 'date') else index,
                               'type': 'buy', 'price': fill_price})
                logging.info(f"Simulating BUY at {fill_price:.2f} (open {open_price:.2f}) on {index}")

            elif pending_sig == 'short' and not bot.has_position:
                fill_price = _compute_fill_price(open_price, 'short', slippage_pct, fill_model)
                bot.has_position = True
                bot._handle_position_entry(fill_price, position_type='short')
                entry_bar_idx = i
                trades.append({'date': index.date() if hasattr(index, 'date') else index,
                               'type': 'short', 'price': fill_price})
                logging.info(f"Simulating SHORT at {fill_price:.2f} (open {open_price:.2f}) on {index}")

            elif pending_sig == 'sell' and bot.has_position:
                sig_type = 'sell_short' if bot.position_type == 'short' else 'sell_long'
                fill_price = _compute_fill_price(open_price, sig_type, slippage_pct, fill_model)
                closing_position_type = bot.position_type
                closing_entry_price = bot.entry_price
                bars_held = (i - entry_bar_idx) if entry_bar_idx is not None else 0
                bot.has_position = False
                bot._handle_position_exit()
                # Update cash with realized PnL
                if closing_position_type == 'long':
                    realized = (fill_price - closing_entry_price) * trade_amount
                elif closing_position_type == 'short':
                    realized = (closing_entry_price - fill_price) * trade_amount
                else:
                    realized = 0.0
                entry_comm = closing_entry_price * commission_pct
                exit_comm = fill_price * commission_pct
                cash += realized - entry_comm - exit_comm
                trades.append({'date': index.date() if hasattr(index, 'date') else index,
                               'type': 'sell', 'price': fill_price,
                               'closed_position': closing_position_type,
                               'exit_reason': pending_exit_reason,
                               'bars_held': bars_held,
                               'entry_price': closing_entry_price})
                entry_bar_idx = None
                logging.info(f"Simulating SELL ({closing_position_type}) at {fill_price:.2f} (open {open_price:.2f}) on {index}")

            else:
                # Signal could not be executed — log the drop
                if pending_sig in ('buy', 'short') and bot.has_position:
                    logging.warning(
                        f"Dropped pending '{pending_sig}' signal on {index}: "
                        f"already in {bot.position_type} position"
                    )
                elif pending_sig == 'sell' and not bot.has_position:
                    logging.warning(
                        f"Dropped pending 'sell' signal on {index}: "
                        f"no open position"
                    )

        # --- Generate signal for current bar ---
        # Build bar dict for fill model (volume-based slippage) even if strategy is price-only
        bar = None
        if 'Volume' in row.index:
            bar = {'Volume': float(row['Volume'])}

        # Dispatch to bar-based or price-based logic
        if uses_ohlcv:
            bar = {
                'Open': float(row['Open']),
                'High': float(row['High']),
                'Low': float(row['Low']),
                'Close': current_price,
                'Volume': float(row['Volume']) if 'Volume' in row.index else 0,
            }
            raw_signal = bot._run_strategy_logic_bar(bar)
        else:
            raw_signal = bot._run_strategy_logic(current_price)

        signal, exit_reason = _unpack_signal(raw_signal)

        if fill_model == 'next_open':
            # Defer execution to next bar
            if signal in ('buy', 'short', 'sell'):
                pending_signal = (signal, exit_reason, bar)
        else:
            # Execute immediately (close and vwap_slippage models)
            if signal == 'buy' and not bot.has_position:
                fill_price = _compute_fill_price(current_price, 'buy', slippage_pct,
                                                  fill_model, bar=bar, median_volume=median_volume)
                bot.has_position = True
                bot._handle_position_entry(fill_price, position_type='long')
                entry_bar_idx = i
                trades.append({'date': index.date() if hasattr(index, 'date') else index,
                               'type': 'buy', 'price': fill_price})
                logging.info(f"Simulating BUY at {fill_price:.2f} (market {current_price:.2f}) on {index}")

            elif signal == 'short' and not bot.has_position:
                fill_price = _compute_fill_price(current_price, 'short', slippage_pct,
                                                  fill_model, bar=bar, median_volume=median_volume)
                bot.has_position = True
                bot._handle_position_entry(fill_price, position_type='short')
                entry_bar_idx = i
                trades.append({'date': index.date() if hasattr(index, 'date') else index,
                               'type': 'short', 'price': fill_price})
                logging.info(f"Simulating SHORT at {fill_price:.2f} (market {current_price:.2f}) on {index}")

            elif signal == 'sell' and bot.has_position:
                sig_type = 'sell_short' if bot.position_type == 'short' else 'sell_long'
                fill_price = _compute_fill_price(current_price, sig_type, slippage_pct,
                                                  fill_model, bar=bar, median_volume=median_volume)
                closing_position_type = bot.position_type
                closing_entry_price = bot.entry_price
                bars_held = (i - entry_bar_idx) if entry_bar_idx is not None else 0
                bot.has_position = False
                bot._handle_position_exit()
                # Update cash with realized PnL
                if closing_position_type == 'long':
                    realized = (fill_price - closing_entry_price) * trade_amount
                elif closing_position_type == 'short':
                    realized = (closing_entry_price - fill_price) * trade_amount
                else:
                    realized = 0.0
                entry_comm = closing_entry_price * commission_pct
                exit_comm = fill_price * commission_pct
                cash += realized - entry_comm - exit_comm
                trades.append({'date': index.date() if hasattr(index, 'date') else index,
                               'type': 'sell', 'price': fill_price,
                               'closed_position': closing_position_type,
                               'exit_reason': exit_reason,
                               'bars_held': bars_held,
                               'entry_price': closing_entry_price})
                entry_bar_idx = None
                logging.info(f"Simulating SELL ({closing_position_type}) at {fill_price:.2f} (market {current_price:.2f}) on {index}")

        # --- Track equity (mark-to-market) ---
        if bot.position_type == 'long':
            unrealized = (current_price - bot.entry_price) * trade_amount
            portfolio_value = cash + unrealized
        elif bot.position_type == 'short':
            unrealized = (bot.entry_price - current_price) * trade_amount
            portfolio_value = cash + unrealized
        else:
            portfolio_value = cash

        equity_curve.append(portfolio_value)
        equity_dates.append(index)

    logging.info("Backtest simulation finished.")

    # --- Force-close any open position at end of backtest ---
    if bot.has_position and bot.entry_price is not None and len(simulation_data) > 0:
        last_price = float(simulation_data.iloc[-1]['Close'])
        last_date = simulation_data.index[-1]
        closing_position_type = bot.position_type
        closing_entry_price = bot.entry_price
        bars_held = (len(sim_rows) - 1 - entry_bar_idx) if entry_bar_idx is not None else 0

        if closing_position_type == 'long':
            realized = (last_price - closing_entry_price) * trade_amount
        elif closing_position_type == 'short':
            realized = (closing_entry_price - last_price) * trade_amount
        else:
            realized = 0.0

        entry_comm = closing_entry_price * commission_pct
        exit_comm = last_price * commission_pct
        cash += realized - entry_comm - exit_comm

        bot.has_position = False
        bot._handle_position_exit()

        trades.append({
            'date': last_date.date() if hasattr(last_date, 'date') else last_date,
            'type': 'sell',
            'price': last_price,
            'closed_position': closing_position_type,
            'exit_reason': 'end_of_data',
            'bars_held': bars_held,
            'entry_price': closing_entry_price,
        })
        entry_bar_idx = None
        logging.info(
            f"Force-closed {closing_position_type} position at {last_price:.2f} "
            f"(end of backtest)"
        )

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
    round_trip_profits = []  # per-trade profit for enhanced stats
    round_trip_bars = []     # per-trade bars held
    exit_reason_counts = {}  # exit_reason → count

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
            round_trip_profits.append(profit)
            if profit > 0:
                wins += 1
                gross_profits += profit
            else:
                losses += 1
                gross_losses += abs(profit)
            last_entry_price = 0
            last_entry_type = None
            trade_count += 1

            # Track bars held
            bh = trade.get('bars_held')
            if bh is not None:
                round_trip_bars.append(bh)

            # Track exit reasons
            reason = trade.get('exit_reason')
            if reason is not None:
                exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1

    profit_factor = gross_profits / gross_losses if gross_losses > 0 else 999.99

    # --- Enhanced trade statistics ---
    # Consecutive wins / losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    for p in round_trip_profits:
        if p > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)

    largest_win = max(round_trip_profits) if round_trip_profits else 0.0
    largest_loss = min(round_trip_profits) if round_trip_profits else 0.0
    avg_bars_held = (sum(round_trip_bars) / len(round_trip_bars)) if round_trip_bars else 0.0

    # Expectancy: (avg_win × win_rate) - (avg_loss × loss_rate)
    if trade_count > 0:
        win_rate = wins / trade_count
        loss_rate = losses / trade_count
        avg_win = (gross_profits / wins) if wins > 0 else 0.0
        avg_loss = (gross_losses / losses) if losses > 0 else 0.0
        expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)
    else:
        win_rate = 0.0
        loss_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        expectancy = 0.0

    logging.info("--- Backtest Results ---")
    logging.info(f"Total PnL (after costs): {pnl:.2f}")
    logging.info(f"Profit Factor: {profit_factor:.4f}")
    logging.info(f"Total Commission Paid: {total_commission:.2f}")
    logging.info(f"Completed Trades (Buy/Sell pairs): {trade_count}")
    if trade_count > 0:
        logging.info(f"Win/Loss Ratio: {wins}/{losses}")
        logging.info(f"Win Rate: {win_rate*100:.2f}%")
        logging.info(f"Expectancy: {expectancy:.2f}")
        logging.info(f"Avg Bars Held: {avg_bars_held:.1f}")

    result = {
        'pnl': pnl,
        'profit_factor': profit_factor,
        'total_commission': total_commission,
        'trade_count': trade_count,
        'wins': wins,
        'losses': losses,
        'gross_profits': gross_profits,
        'gross_losses': gross_losses,
        'trades': trades,
        'equity_curve': equity_curve,
        'equity_dates': equity_dates,
        'initial_capital': initial_capital,
        'consecutive_wins': max_consecutive_wins,
        'consecutive_losses': max_consecutive_losses,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'avg_bars_held': avg_bars_held,
        'expectancy': expectancy,
        'exit_reason_counts': exit_reason_counts,
    }

    # Compute risk-adjusted metrics if equity curve is available
    try:
        from src.metrics import compute_all_metrics
        metrics = compute_all_metrics(equity_curve, equity_dates,
                                       simulation_data, initial_capital)
        result.update(metrics)
    except ImportError:
        pass  # metrics module not yet available

    return result


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
        # yfinance returns MultiIndex columns for single tickers; flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
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
