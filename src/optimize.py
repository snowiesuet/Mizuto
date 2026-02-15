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
            # Treat capped profit factor (no losses) as 0 for comparison
            if metric_val >= 999.99:
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


def walk_forward_optimize(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    short_window_range: range = range(3, 15),
    long_window_range: range = range(10, 50, 5),
    trailing_stop_pcts: List[Optional[float]] = [None],
    stop_loss_pcts: List[Optional[float]] = [None],
    metric: str = 'profit_factor',
    slippage_pct: float = 0.001,
    commission_pct: float = 0.001,
    strategy=None,
) -> Dict:
    """
    Walk-forward optimization: optimize on training set, evaluate on test set.

    Splits data temporally into train/test, finds best parameters on the
    training portion, then evaluates those parameters out-of-sample on the
    test portion. This tests whether optimized parameters generalize.

    Args:
        data: OHLC DataFrame with 'Close' column.
        train_ratio: Fraction of data for training (default 0.7).
        short_window_range: Range of short MA windows to test.
        long_window_range: Range of long MA windows to test.
        trailing_stop_pcts: Trailing stop percentages to test.
        stop_loss_pcts: Fixed stop-loss percentages to test.
        metric: Metric to optimize ('profit_factor' or 'pnl').
        slippage_pct: Slippage fraction.
        commission_pct: Commission fraction.
        strategy: Optional BaseStrategy instance.

    Returns:
        Dict with keys:
          'train_params': best params from training set
          'train_metric': best metric on training set
          'test_result': full backtest result on test set (includes risk metrics)
          'test_metric': metric value on test set
          'train_dates': (start, end) tuple
          'test_dates': (start, end) tuple
          'overfit_ratio': train_metric / test_metric (>1 suggests overfitting)
    """
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    if train_data.empty or test_data.empty:
        raise ValueError(
            f"Train/test split produced empty set. "
            f"Data has {len(data)} rows, split at {split_idx}."
        )

    # Optimize on training data
    best_params, train_metric_val = optimize_strategy(
        train_data,
        short_window_range=short_window_range,
        long_window_range=long_window_range,
        trailing_stop_pcts=trailing_stop_pcts,
        stop_loss_pcts=stop_loss_pcts,
        metric=metric,
        slippage_pct=slippage_pct,
        commission_pct=commission_pct,
    )

    # Evaluate on test data with best params
    test_result = run_backtest_on_data(
        test_data,
        short_window=best_params['short_window'],
        long_window=best_params['long_window'],
        trailing_stop_pct=best_params['trailing_stop_pct'],
        stop_loss_pct=best_params['stop_loss_pct'],
        slippage_pct=slippage_pct,
        commission_pct=commission_pct,
        quiet=True,
        strategy=strategy,
    )

    test_metric_val = 0.0
    if test_result is not None:
        test_metric_val = test_result.get(metric, 0.0)
        if test_metric_val >= 999.99:
            test_metric_val = 0.0

    # Compute overfit ratio
    if test_metric_val != 0:
        overfit_ratio = train_metric_val / test_metric_val
    else:
        overfit_ratio = float('inf') if train_metric_val > 0 else 1.0

    train_dates = (train_data.index[0], train_data.index[-1])
    test_dates = (test_data.index[0], test_data.index[-1])

    return {
        'train_params': best_params,
        'train_metric': train_metric_val,
        'test_result': test_result,
        'test_metric': test_metric_val,
        'train_dates': train_dates,
        'test_dates': test_dates,
        'overfit_ratio': overfit_ratio,
    }


def rolling_walk_forward(
    data: pd.DataFrame,
    n_windows: int = 5,
    train_ratio: float = 0.7,
    short_window_range: range = range(3, 15),
    long_window_range: range = range(10, 50, 5),
    trailing_stop_pcts: List[Optional[float]] = [None],
    stop_loss_pcts: List[Optional[float]] = [None],
    metric: str = 'profit_factor',
    slippage_pct: float = 0.001,
    commission_pct: float = 0.001,
    strategy=None,
) -> Dict:
    """
    Rolling walk-forward optimization across multiple windows.

    Divides the test period into n_windows equal segments. For each window,
    uses a fixed-length training period immediately before it and optimizes
    parameters on that training set, then evaluates on the test window.

    Args:
        data: OHLC DataFrame with 'Close' column.
        n_windows: Number of walk-forward windows (default 5).
        train_ratio: Fraction of total data used for training in first window.
        short_window_range: Range of short MA windows.
        long_window_range: Range of long MA windows.
        trailing_stop_pcts: Trailing stop percentages.
        stop_loss_pcts: Fixed stop-loss percentages.
        metric: Metric to optimize.
        slippage_pct: Slippage fraction.
        commission_pct: Commission fraction.
        strategy: Optional BaseStrategy instance.

    Returns:
        Dict with keys:
          'windows': list of per-window results (each like walk_forward_optimize)
          'aggregate_test_metric': mean of test metrics across windows
          'aggregate_train_metric': mean of train metrics
          'aggregate_overfit_ratio': mean overfit ratio
    """
    total_len = len(data)
    train_window_size = int(total_len * train_ratio)
    remaining = total_len - train_window_size
    test_window_size = remaining // n_windows

    if test_window_size < 1:
        raise ValueError(
            f"Not enough data for {n_windows} windows. "
            f"Data has {total_len} rows, train window is {train_window_size}."
        )

    windows = []
    for i in range(n_windows):
        test_start = train_window_size + i * test_window_size
        test_end = test_start + test_window_size
        if i == n_windows - 1:
            test_end = total_len  # Last window absorbs remainder
        train_start = max(0, test_start - train_window_size)

        window_data = data.iloc[train_start:test_end]
        window_train_ratio = (test_start - train_start) / (test_end - train_start)

        try:
            result = walk_forward_optimize(
                window_data,
                train_ratio=window_train_ratio,
                short_window_range=short_window_range,
                long_window_range=long_window_range,
                trailing_stop_pcts=trailing_stop_pcts,
                stop_loss_pcts=stop_loss_pcts,
                metric=metric,
                slippage_pct=slippage_pct,
                commission_pct=commission_pct,
                strategy=strategy,
            )
            windows.append(result)
        except ValueError as e:
            logging.warning(f"Window {i} skipped: {e}")

    if not windows:
        raise ValueError("All walk-forward windows failed.")

    test_metrics = [w['test_metric'] for w in windows]
    train_metrics = [w['train_metric'] for w in windows]
    overfit_ratios = [w['overfit_ratio'] for w in windows
                      if w['overfit_ratio'] != float('inf')]

    return {
        'windows': windows,
        'aggregate_test_metric': sum(test_metrics) / len(test_metrics),
        'aggregate_train_metric': sum(train_metrics) / len(train_metrics),
        'aggregate_overfit_ratio': (
            sum(overfit_ratios) / len(overfit_ratios)
            if overfit_ratios else float('inf')
        ),
    }
