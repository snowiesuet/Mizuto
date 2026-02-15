"""
Multi-asset backtesting support.

Run backtests across multiple symbols sequentially and aggregate results.
Not portfolio-level (no cross-asset correlation or allocation).
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from src.backtest import run_backtest_on_data
from src.bot import SHORT_WINDOW, LONG_WINDOW


def run_multi_backtest_on_data(
    datasets,
    trade_amount=1.0,
    short_window=SHORT_WINDOW,
    long_window=LONG_WINDOW,
    trailing_stop_pct=None,
    stop_loss_pct=None,
    slippage_pct=0.001,
    commission_pct=0.001,
    strategy=None,
    initial_capital=10000.0,
):
    """
    Run backtests on multiple symbols from pre-loaded DataFrames.

    Args:
        datasets: Dict mapping symbol name -> DataFrame with 'Close' column.
        trade_amount: Units per trade.
        short_window: Short MA window.
        long_window: Long MA window.
        trailing_stop_pct: Trailing stop percentage (None to disable).
        stop_loss_pct: Fixed stop-loss percentage (None to disable).
        slippage_pct: Slippage fraction per trade.
        commission_pct: Commission fraction per trade.
        strategy: A BaseStrategy instance (reset between symbols).
        initial_capital: Starting capital per symbol.

    Returns:
        Dict with:
          'per_symbol': dict mapping symbol -> individual backtest result
          'summary': DataFrame with one row per symbol
          'aggregate': dict with aggregated metrics
    """
    per_symbol = {}

    for symbol, data in datasets.items():
        if strategy is not None:
            strategy.reset()

        result = run_backtest_on_data(
            data,
            symbol=symbol,
            trade_amount=trade_amount,
            short_window=short_window,
            long_window=long_window,
            trailing_stop_pct=trailing_stop_pct,
            stop_loss_pct=stop_loss_pct,
            slippage_pct=slippage_pct,
            commission_pct=commission_pct,
            quiet=True,
            strategy=strategy,
            initial_capital=initial_capital,
        )
        per_symbol[symbol] = result

    # Build summary DataFrame
    rows = []
    for symbol, result in per_symbol.items():
        if result is None:
            rows.append({
                'symbol': symbol,
                'pnl': None,
                'trade_count': 0,
                'win_rate': None,
                'profit_factor': None,
                'sharpe_ratio': None,
                'max_drawdown_pct': None,
            })
        else:
            tc = result['trade_count']
            rows.append({
                'symbol': symbol,
                'pnl': result['pnl'],
                'trade_count': tc,
                'win_rate': result['wins'] / tc if tc > 0 else 0.0,
                'profit_factor': result['profit_factor'],
                'sharpe_ratio': result.get('sharpe_ratio'),
                'max_drawdown_pct': result.get('max_drawdown_pct'),
            })

    summary = pd.DataFrame(rows)

    # Aggregate metrics across symbols with trades
    valid = {k: v for k, v in per_symbol.items() if v is not None}
    if valid:
        sharpe_values = [v.get('sharpe_ratio', 0) for v in valid.values()
                         if v.get('sharpe_ratio') is not None]
        aggregate = {
            'total_pnl': sum(v['pnl'] for v in valid.values()),
            'mean_sharpe': float(np.mean(sharpe_values)) if sharpe_values else None,
            'total_trades': sum(v['trade_count'] for v in valid.values()),
            'symbols_profitable': sum(1 for v in valid.values() if v['pnl'] > 0),
            'symbols_tested': len(per_symbol),
        }
    else:
        aggregate = {
            'total_pnl': 0.0,
            'mean_sharpe': None,
            'total_trades': 0,
            'symbols_profitable': 0,
            'symbols_tested': len(per_symbol),
        }

    return {
        'per_symbol': per_symbol,
        'summary': summary,
        'aggregate': aggregate,
    }


def run_multi_backtest(
    symbols,
    start_date,
    end_date,
    trade_amount=1.0,
    short_window=SHORT_WINDOW,
    long_window=LONG_WINDOW,
    trailing_stop_pct=None,
    stop_loss_pct=None,
    slippage_pct=0.001,
    commission_pct=0.001,
    strategy=None,
    initial_capital=10000.0,
):
    """
    Run backtests on multiple symbols, downloading data from yfinance.

    Args:
        symbols: List of ticker symbols (e.g., ['BTC-USD', 'ETH-USD']).
        start_date: Start date string (e.g., '2023-01-01').
        end_date: End date string (e.g., '2023-12-31').
        (other args same as run_multi_backtest_on_data)

    Returns:
        Same as run_multi_backtest_on_data.
    """
    datasets = {}
    for symbol in symbols:
        try:
            data = yf.download(
                tickers=symbol, start=start_date, end=end_date,
                interval="1d", auto_adjust=True,
            )
            if not data.empty:
                datasets[symbol] = data
            else:
                logging.warning(f"No data for {symbol}, skipping.")
        except Exception as e:
            logging.warning(f"Failed to download {symbol}: {e}")

    return run_multi_backtest_on_data(
        datasets,
        trade_amount=trade_amount,
        short_window=short_window,
        long_window=long_window,
        trailing_stop_pct=trailing_stop_pct,
        stop_loss_pct=stop_loss_pct,
        slippage_pct=slippage_pct,
        commission_pct=commission_pct,
        strategy=strategy,
        initial_capital=initial_capital,
    )
