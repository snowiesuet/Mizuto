"""Run and optimise backtesting.py strategies.

Usage (CLI)::

    python -m bt.runner --strategy ma_crossover --symbol BTC-USD --start 2023-01-01 --end 2023-12-31
    python -m bt.runner --strategy atr_breakout --optimize
    python -m bt.runner --strategy pivot_points --nq --timeframe 1d --plot
"""

import argparse

import pandas as pd
import yfinance as yf
from backtesting import Backtest
from backtesting.lib import FractionalBacktest

from bt.strategies import MACrossoverBT, ATRBreakoutStrategy, PivotPointStrategy

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    "ma_crossover": MACrossoverBT,
    "atr_breakout": ATRBreakoutStrategy,
    "pivot_points": PivotPointStrategy,
}

# Default parameter grids for optimisation
OPTIMIZE_PARAMS = {
    "ma_crossover": dict(
        short_window=range(3, 15),
        long_window=range(15, 50, 5),
    ),
    "atr_breakout": dict(
        adx_threshold=range(20, 35, 5),
        atr_multiplier=[1.0, 1.5, 2.0],
        lookback_period=range(10, 30, 5),
    ),
    "pivot_points": dict(
        use_s2_r2=[False, True],
    ),
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(
    symbol: str = "BTC-USD",
    start: str = "2023-01-01",
    end: str = "2023-12-31",
    nq: bool = False,
    timeframe: str = "1d",
) -> pd.DataFrame:
    """Load OHLCV data — either from yfinance or the local NQ CSV store.

    Returns a DataFrame with columns Open, High, Low, Close, Volume and a
    DatetimeIndex (as required by backtesting.py).
    """
    if nq:
        from src.data_loader import load_nq

        df = load_nq(timeframe=timeframe)
        # Filter to date range if provided
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        return df

    df = yf.download(symbol, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


# ---------------------------------------------------------------------------
# Run / optimise helpers
# ---------------------------------------------------------------------------


def run_bt(
    strategy_name: str,
    data: pd.DataFrame | None = None,
    cash: float = 10_000,
    commission: float = 0.001,
    plot: bool = False,
    fractional: bool = True,
    **kwargs,
):
    """Run a single backtest and return the stats Series.

    Extra *kwargs* are forwarded as strategy parameter overrides (e.g.
    ``short_window=10``).
    """
    strategy_cls = STRATEGIES[strategy_name]

    if kwargs:
        strategy_cls = type(
            strategy_cls.__name__,
            (strategy_cls,),
            kwargs,
        )

    if data is None:
        data = load_data()

    bt_cls = FractionalBacktest if fractional else Backtest
    bt = bt_cls(data, strategy_cls, cash=cash, commission=commission)
    stats = bt.run()

    if plot:
        bt.plot()

    return stats


def optimize_bt(
    strategy_name: str,
    data: pd.DataFrame | None = None,
    cash: float = 10_000,
    commission: float = 0.001,
    maximize: str = "Equity Final [$]",
    fractional: bool = True,
    **override_params,
):
    """Grid-optimise a strategy and return the best stats + param grid.

    *override_params* replace the default grid for matching keys.
    """
    strategy_cls = STRATEGIES[strategy_name]
    params = {**OPTIMIZE_PARAMS.get(strategy_name, {}), **override_params}

    if data is None:
        data = load_data()

    bt_cls = FractionalBacktest if fractional else Backtest
    bt = bt_cls(data, strategy_cls, cash=cash, commission=commission)
    stats = bt.optimize(maximize=maximize, **params)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run backtesting.py strategies")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGIES),
        help="Strategy to run",
    )
    parser.add_argument("--symbol", default="BTC-USD", help="yfinance symbol")
    parser.add_argument("--start", default="2023-01-01", help="Start date")
    parser.add_argument("--end", default="2023-12-31", help="End date")
    parser.add_argument("--cash", type=float, default=10_000)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--optimize", action="store_true", help="Run grid optimisation")
    parser.add_argument("--plot", action="store_true", help="Show interactive plot")
    parser.add_argument("--nq", action="store_true", help="Use local NQ CSV data")
    parser.add_argument("--timeframe", default="1d", help="NQ timeframe (1d/1h/5m)")

    args = parser.parse_args()

    data = load_data(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        nq=args.nq,
        timeframe=args.timeframe,
    )

    if data.empty:
        print("No data loaded — aborting.")
        return

    if args.optimize:
        stats = optimize_bt(
            args.strategy,
            data=data,
            cash=args.cash,
            commission=args.commission,
        )
    else:
        stats = run_bt(
            args.strategy,
            data=data,
            cash=args.cash,
            commission=args.commission,
            plot=args.plot,
        )

    print(stats)


if __name__ == "__main__":
    main()
