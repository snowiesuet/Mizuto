"""Run backtests on local NQ futures data.

Usage:
    python scripts/backtest_nq.py                   # 1d, all years
    python scripts/backtest_nq.py --timeframe 5m    # 5m, all years
    python scripts/backtest_nq.py --timeframe 1h    # 1h
    python scripts/backtest_nq.py --years 2023 2024 # specific years
"""

import argparse
import logging
import sys
import os

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_nq
from src.backtest import run_backtest_on_data
from src.utils import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Backtest MA crossover on NQ futures data")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "1h", "5m"],
                        help="Data timeframe (default: 1d)")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Filter by year(s), e.g. --years 2023 2024")
    parser.add_argument("--short-window", type=int, default=5,
                        help="Short MA window (default: 5)")
    parser.add_argument("--long-window", type=int, default=20,
                        help="Long MA window (default: 20)")
    parser.add_argument("--trailing-stop", type=float, default=None,
                        help="Trailing stop-loss %% as decimal (e.g. 0.05 for 5%%)")
    parser.add_argument("--stop-loss", type=float, default=None,
                        help="Fixed stop-loss %% as decimal (e.g. 0.10 for 10%%)")
    parser.add_argument("--trade-amount", type=float, default=1.0,
                        help="Units per trade (default: 1)")
    parser.add_argument("--slippage", type=float, default=0.001,
                        help="Slippage fraction (default: 0.001)")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Commission fraction (default: 0.001)")
    args = parser.parse_args()

    configure_logging()

    logging.info(f"Loading NQ {args.timeframe} data...")
    data = load_nq(timeframe=args.timeframe, years=args.years)
    logging.info(f"Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")

    results = run_backtest_on_data(
        data=data,
        symbol=f"NQ-{args.timeframe}",
        trade_amount=args.trade_amount,
        short_window=args.short_window,
        long_window=args.long_window,
        trailing_stop_pct=args.trailing_stop,
        stop_loss_pct=args.stop_loss,
        slippage_pct=args.slippage,
        commission_pct=args.commission,
    )

    if results is None:
        logging.info("No trades executed.")
    else:
        print(f"\n{'='*40}")
        print(f"NQ Backtest Results ({args.timeframe})")
        print(f"{'='*40}")
        print(f"Total PnL:       {results['pnl']:.2f}")
        print(f"Profit Factor:   {results['profit_factor']:.4f}")
        print(f"Trades:          {results['trade_count']}")
        print(f"Wins:            {results['wins']}")
        print(f"Losses:          {results['losses']}")
        if results['trade_count'] > 0:
            print(f"Win Rate:        {(results['wins']/results['trade_count'])*100:.1f}%")
        print(f"Gross Profits:   {results['gross_profits']:.2f}")
        print(f"Gross Losses:    {results['gross_losses']:.2f}")
        print(f"Commission:      {results['total_commission']:.2f}")


if __name__ == "__main__":
    main()
