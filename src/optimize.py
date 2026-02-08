"""
Parameter optimization for Mizuto trading strategy.
Grid search over MA windows and stop-loss parameters.
"""

import itertools
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.backtest import run_backtest_on_data


def optimize_strategy(
    data: pd.DataFrame,
    short_window_range: range = range(3, 15),
    long_window_range: range = range(10, 50, 5),
    trailing_stop_pcts: List[Optional[float]] = [None],
    stop_loss_pcts: List[Optional[float]] = [None],
    metric: str = 'profit_factor',
    slippage_pct: float = 0.001,
    commission_pct: float = 0.001,
) -> Tuple[Dict, float]:
    """
    Grid search over strategy parameters to find the best combination.

    Args:
        data: OHLC DataFrame with 'Close' column.
        short_window_range: Range of short MA windows to test.
        long_window_range: Range of long MA windows to test.
        trailing_stop_pcts: List of trailing stop percentages to test (include None for disabled).
        stop_loss_pcts: List of fixed stop-loss percentages to test (include None for disabled).
        metric: Which metric to optimize ('profit_factor' or 'pnl').
        slippage_pct: Slippage fraction.
        commission_pct: Commission fraction.

    Returns:
        Tuple of (best_params_dict, best_metric_value).
        best_params_dict has keys: 'short_window', 'long_window',
                                    'trailing_stop_pct', 'stop_loss_pct'
    """
    param_grid = [
        (sw, lw, tsp, slp)
        for sw, lw, tsp, slp in itertools.product(
            short_window_range, long_window_range, trailing_stop_pcts, stop_loss_pcts
        )
        if sw < lw
    ]

    best_metric = -float('inf')
    best_params = None

    for sw, lw, tsp, slp in param_grid:
        result = run_backtest_on_data(
            data,
            short_window=sw,
            long_window=lw,
            trailing_stop_pct=tsp,
            stop_loss_pct=slp,
            slippage_pct=slippage_pct,
            commission_pct=commission_pct,
            quiet=True,
        )

        if result is None:
            metric_val = 0.0
        else:
            metric_val = result.get(metric, 0.0)
            # Treat infinite profit factor (no losses) as 0 for comparison
            if metric_val == float('inf'):
                metric_val = 0.0

        if metric_val > best_metric:
            best_metric = metric_val
            best_params = {
                'short_window': sw,
                'long_window': lw,
                'trailing_stop_pct': tsp,
                'stop_loss_pct': slp,
            }

    if best_params is None:
        # No valid parameter combos (shouldn't happen, but be safe)
        best_params = {
            'short_window': short_window_range[0],
            'long_window': long_window_range[0],
            'trailing_stop_pct': trailing_stop_pcts[0],
            'stop_loss_pct': stop_loss_pcts[0],
        }
        best_metric = 0.0

    return best_params, best_metric


def optimize_strategy_fast(
    data: pd.DataFrame,
    short_window_range: range = range(3, 15),
    long_window_range: range = range(10, 50, 5),
    metric: str = 'profit_factor',
    slippage_pct: float = 0.001,
    commission_pct: float = 0.001,
) -> Tuple[Dict, float]:
    """
    Simplified optimization over MA windows only (no stop-loss grid).
    Faster for MCPT where optimization runs hundreds of times.

    Args:
        data: OHLC DataFrame with 'Close' column.
        short_window_range: Range of short MA windows to test.
        long_window_range: Range of long MA windows to test.
        metric: Which metric to optimize ('profit_factor' or 'pnl').
        slippage_pct: Slippage fraction.
        commission_pct: Commission fraction.

    Returns:
        Tuple of (best_params_dict, best_metric_value).
    """
    return optimize_strategy(
        data,
        short_window_range=short_window_range,
        long_window_range=long_window_range,
        trailing_stop_pcts=[None],
        stop_loss_pcts=[None],
        metric=metric,
        slippage_pct=slippage_pct,
        commission_pct=commission_pct,
    )
