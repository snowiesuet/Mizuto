"""
Monte Carlo Permutation Testing (MCPT) for Mizuto Trading Strategy.

Tests whether the strategy's performance is statistically significant
by comparing it against performance on shuffled (permuted) price data.

Usage:
    python mcpt_validation.py
    python mcpt_validation.py --symbol BTC-USD --start 2023-01-01 --end 2023-12-31
    python mcpt_validation.py --walkforward --train-ratio 0.7
    python mcpt_validation.py --permutations 500 --save-plots

References:
    - neurotrader888/mcpt (MIT license) for bar permutation methodology
    - White, H. (2000). "A Reality Check for Data Snooping"
"""

import argparse
import logging
import sys
import os
from typing import Dict, List, Optional

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.bar_permute import get_permutation
from src.backtest import run_backtest_on_data
from src.data_loader import load_nq
from src.optimize import optimize_strategy_fast


def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download OHLC data from yfinance.

    Returns:
        DataFrame with columns Open, High, Low, Close indexed by date.

    Raises:
        ValueError: If no data is returned.
    """
    data = yf.download(tickers=symbol, start=start_date, end=end_date,
                       interval="1d", auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data fetched for {symbol} from {start_date} to {end_date}")

    # Flatten MultiIndex columns if present (yfinance sometimes returns these)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    return data


def run_insample_mcpt(
    data: pd.DataFrame,
    n_permutations: int = 200,
    short_window_range: range = range(3, 15),
    long_window_range: range = range(10, 50, 5),
    metric: str = 'profit_factor',
    slippage_pct: float = 0.001,
    commission_pct: float = 0.001,
) -> Dict:
    """
    Run in-sample MCPT.

    1. Optimize strategy on real data -> best_real_metric
    2. For each permutation:
       a. Permute the OHLC data
       b. Optimize strategy on permuted data -> best_perm_metric
    3. p-value = count(best_perm_metric >= best_real_metric) / n_permutations

    Returns:
        Dict with 'real_params', 'real_metric', 'permuted_metrics', 'p_value',
        'n_permutations'.
    """
    # Step 1: Optimize on real data
    print("Optimizing strategy on real data...")
    best_params, best_real_metric = optimize_strategy_fast(
        data, short_window_range, long_window_range, metric,
        slippage_pct=slippage_pct, commission_pct=commission_pct,
    )
    print(f"Real data - Best {metric}: {best_real_metric:.4f}, Params: {best_params}")

    # Step 2: Run permutations
    perm_better_count = 1  # Include real result itself (standard convention)
    permuted_metrics = []

    print(f"\nRunning {n_permutations} permutations...")
    for i in tqdm(range(1, n_permutations), desc="MCPT Permutations"):
        perm_data = get_permutation(data, start_index=0, seed=i)
        _, best_perm_metric = optimize_strategy_fast(
            perm_data, short_window_range, long_window_range, metric,
            slippage_pct=slippage_pct, commission_pct=commission_pct,
        )

        if best_perm_metric >= best_real_metric:
            perm_better_count += 1

        permuted_metrics.append(best_perm_metric)

    p_value = perm_better_count / n_permutations

    return {
        'real_params': best_params,
        'real_metric': best_real_metric,
        'permuted_metrics': permuted_metrics,
        'p_value': p_value,
        'n_permutations': n_permutations,
    }


def run_walkforward_mcpt(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    n_permutations: int = 200,
    short_window_range: range = range(3, 15),
    long_window_range: range = range(10, 50, 5),
    metric: str = 'profit_factor',
    slippage_pct: float = 0.001,
    commission_pct: float = 0.001,
) -> Dict:
    """
    Run walk-forward (out-of-sample) MCPT.

    1. Split data into train/test by train_ratio
    2. Optimize on train data -> best params
    3. Evaluate best params on test data -> real_oos_metric
    4. For each permutation:
       a. Permute data (start_index = split point, preserving train data)
       b. Optimize on train portion of permuted data -> perm best params
       c. Evaluate perm best params on permuted test data -> perm_oos_metric
    5. p-value = count(perm_oos_metric >= real_oos_metric) / n_permutations

    Returns:
        Dict with 'real_params', 'real_oos_metric', 'permuted_oos_metrics',
        'p_value', 'train_end_date', 'n_permutations'.
    """
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    print(f"Train period: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} bars)")
    print(f"Test period:  {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data)} bars)")

    # Step 1: Optimize on real train data
    print("\nOptimizing on training data...")
    best_params, train_metric = optimize_strategy_fast(
        train_data, short_window_range, long_window_range, metric,
        slippage_pct=slippage_pct, commission_pct=commission_pct,
    )
    print(f"Train - Best {metric}: {train_metric:.4f}, Params: {best_params}")

    # Step 2: Evaluate on real test data with best params
    real_oos_result = run_backtest_on_data(
        test_data,
        short_window=best_params['short_window'],
        long_window=best_params['long_window'],
        slippage_pct=slippage_pct,
        commission_pct=commission_pct,
        quiet=True,
    )
    real_oos_metric = real_oos_result.get(metric, 0.0) if real_oos_result else 0.0
    if real_oos_metric == float('inf'):
        real_oos_metric = 0.0
    print(f"Test (OOS) - {metric}: {real_oos_metric:.4f}")

    # Step 3: Permutation loop
    perm_better_count = 1
    permuted_oos_metrics = []

    print(f"\nRunning {n_permutations} walk-forward permutations...")
    for i in tqdm(range(1, n_permutations), desc="Walk-Forward MCPT"):
        perm_data = get_permutation(data, start_index=split_idx, seed=i)
        perm_train = perm_data.iloc[:split_idx]
        perm_test = perm_data.iloc[split_idx:]

        # Optimize on permuted train data
        perm_params, _ = optimize_strategy_fast(
            perm_train, short_window_range, long_window_range, metric,
            slippage_pct=slippage_pct, commission_pct=commission_pct,
        )

        # Evaluate on permuted test data
        perm_oos_result = run_backtest_on_data(
            perm_test,
            short_window=perm_params['short_window'],
            long_window=perm_params['long_window'],
            slippage_pct=slippage_pct,
            commission_pct=commission_pct,
            quiet=True,
        )
        perm_oos_metric = perm_oos_result.get(metric, 0.0) if perm_oos_result else 0.0
        if perm_oos_metric == float('inf'):
            perm_oos_metric = 0.0

        if perm_oos_metric >= real_oos_metric:
            perm_better_count += 1

        permuted_oos_metrics.append(perm_oos_metric)

    p_value = perm_better_count / n_permutations

    return {
        'real_params': best_params,
        'real_oos_metric': real_oos_metric,
        'permuted_oos_metrics': permuted_oos_metrics,
        'p_value': p_value,
        'train_end_date': train_data.index[-1].date(),
        'n_permutations': n_permutations,
    }


def plot_mcpt_results(
    real_metric: float,
    permuted_metrics: List[float],
    p_value: float,
    title: str = "MCPT Results",
    metric_name: str = "Profit Factor",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot histogram of permuted metrics with the real metric as a vertical line.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(permuted_metrics, bins=30, color='steelblue', alpha=0.7,
             edgecolor='black', label='Permutations')
    plt.axvline(real_metric, color='red', linewidth=2, linestyle='--',
                label=f'Real ({real_metric:.4f})')
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    plt.title(f"{title}\np-value = {p_value:.4f}")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Permutation Testing for Mizuto Trading Strategy"
    )
    parser.add_argument('--symbol', default='BTC-USD',
                        help='Trading symbol (default: BTC-USD)')
    parser.add_argument('--start', default='2023-01-01',
                        help='Start date (default: 2023-01-01)')
    parser.add_argument('--end', default='2023-12-31',
                        help='End date (default: 2023-12-31)')
    parser.add_argument('--permutations', type=int, default=200,
                        help='Number of permutations (default: 200)')
    parser.add_argument('--metric', choices=['profit_factor', 'pnl'],
                        default='profit_factor',
                        help='Metric to optimize (default: profit_factor)')
    parser.add_argument('--walkforward', action='store_true',
                        help='Also run walk-forward MCPT')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train/test split ratio for walk-forward (default: 0.7)')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to files instead of showing')
    parser.add_argument('--nq', action='store_true',
                        help='Use local NQ futures data instead of yfinance')
    parser.add_argument('--timeframe', default='1d', choices=['1d', '1h', '5m'],
                        help='NQ data timeframe (default: 1d, only used with --nq)')
    parser.add_argument('--years', nargs='+', type=int, default=None,
                        help='NQ year filter (e.g. --years 2023 2024, only used with --nq)')

    args = parser.parse_args()

    # Suppress noisy logging during batch runs
    logging.basicConfig(level=logging.WARNING)

    print("=" * 60)
    print("MIZUTO - Monte Carlo Permutation Testing")
    print("=" * 60)

    # Load data from local NQ CSVs or yfinance
    if args.nq:
        symbol = f"NQ-{args.timeframe}"
        data = load_nq(timeframe=args.timeframe, years=args.years)
        year_str = f" years={args.years}" if args.years else " all years"
        print(f"Source: Local NQ data ({args.timeframe},{year_str})")
        print(f"Period: {data.index[0]} to {data.index[-1]}")
    else:
        symbol = args.symbol
        data = fetch_data(args.symbol, args.start, args.end)
        print(f"Symbol: {args.symbol}")
        print(f"Period: {args.start} to {args.end}")

    print(f"Permutations: {args.permutations}")
    print(f"Metric: {args.metric}")
    print(f"Loaded {len(data)} bars of data")
    print()

    # --- In-sample MCPT ---
    print("-" * 40)
    print("IN-SAMPLE MCPT")
    print("-" * 40)
    insample_results = run_insample_mcpt(
        data,
        n_permutations=args.permutations,
        metric=args.metric,
    )

    print(f"\n--- In-Sample Results ---")
    print(f"Best params: {insample_results['real_params']}")
    print(f"Real {args.metric}: {insample_results['real_metric']:.4f}")
    print(f"MCPT p-value: {insample_results['p_value']:.4f}")
    if insample_results['p_value'] < 0.05:
        print(">> SIGNIFICANT at 5% level -- strategy likely captures real patterns")
    else:
        print(">> NOT significant at 5% level -- results may be due to data-snooping")

    save_path = "mcpt_insample.png" if args.save_plots else None
    plot_mcpt_results(
        insample_results['real_metric'],
        insample_results['permuted_metrics'],
        insample_results['p_value'],
        title=f"In-Sample MCPT: {symbol}",
        metric_name=args.metric.replace('_', ' ').title(),
        save_path=save_path,
    )

    # --- Walk-forward MCPT (optional) ---
    if args.walkforward:
        print("\n" + "-" * 40)
        print("WALK-FORWARD MCPT")
        print("-" * 40)
        wf_results = run_walkforward_mcpt(
            data,
            train_ratio=args.train_ratio,
            n_permutations=args.permutations,
            metric=args.metric,
        )

        print(f"\n--- Walk-Forward Results ---")
        print(f"Best params (from train): {wf_results['real_params']}")
        print(f"Train/test split at: {wf_results['train_end_date']}")
        print(f"OOS {args.metric}: {wf_results['real_oos_metric']:.4f}")
        print(f"MCPT p-value: {wf_results['p_value']:.4f}")
        if wf_results['p_value'] < 0.05:
            print(">> SIGNIFICANT at 5% level")
        else:
            print(">> NOT significant at 5% level")

        save_path = "mcpt_walkforward.png" if args.save_plots else None
        plot_mcpt_results(
            wf_results['real_oos_metric'],
            wf_results['permuted_oos_metrics'],
            wf_results['p_value'],
            title=f"Walk-Forward MCPT: {symbol}",
            metric_name=args.metric.replace('_', ' ').title(),
            save_path=save_path,
        )


if __name__ == '__main__':
    main()
