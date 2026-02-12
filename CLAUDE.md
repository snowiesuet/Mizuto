# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mizuto is a Python algorithmic trading bot that supports both live (simulated) trading and historical backtesting. It has two backtesting engines: a custom one in `src/` (MA crossover with trailing/fixed stop-loss) and a `backtesting.py`-based framework in `bt/` with multiple strategies (MA crossover, ATR breakout, pivot points). The project is currently in Phase 2 of its roadmap (backtesting complete; ML-based prediction in Phase 3 is next).

## Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Run live trading bot (simulated exchange, polls every 60s)
python main.py

# Run backtest (BTC-USD, configurable dates/stop-loss in src/backtest.py __main__ block)
python -m src.backtest

# Run trailing stop-loss demo/comparison
python scripts/trailing_stop_demo.py

# Run MCPT validation
python scripts/mcpt_validation.py

# Fetch and log yfinance data for debugging
python scripts/log_yfinance_data.py

# Run backtesting.py strategies (bt/ package)
python -m bt --strategy ma_crossover --symbol BTC-USD --start 2023-01-01 --end 2023-12-31
python -m bt --strategy atr_breakout --optimize
python -m bt --strategy pivot_points --plot

# Run tests
pytest tests/
```

There are no linter or build system configured.

## Architecture

The bot has three execution paths:

- **Live path**: `main.py` → `TradingBot.run_strategy()` → `exchange.fetch_price()` / `exchange.place_order()`
- **Custom backtest path**: `src/backtest.py` → `TradingBot._run_strategy_logic()` (called directly, bypassing exchange)
- **backtesting.py path**: `bt/runner.py` → `backtesting.Backtest` with pluggable `bt.strategies` classes

### Package layout

```
src/                  # Library package
├── __init__.py          # Re-exports TradingBot, configure_logging
├── bot.py               # TradingBot class (core decision engine)
├── exchange.py           # Simulated fetch_price / place_order
├── backtest.py           # Historical backtesting via yfinance
├── bar_permute.py        # OHLCV bar permutation for MCPT
├── indicators.py         # Indicator helpers (ATR, ADX, pivots, rolling max/min)
├── optimize.py           # Grid search over MA / stop-loss params
├── utils.py              # configure_logging helper
└── strategies/           # Pluggable strategy classes
    ├── __init__.py
    ├── base.py           # BaseStrategy ABC (with OHLCV support)
    ├── ma_crossover.py   # MACrossoverStrategy
    ├── atr_breakout.py   # ATRBreakoutStrategy (ADX + ATR breakout, long & short)
    └── pivot_points.py   # PivotPointStrategy (floor pivot bounce/rejection)
```

Key modules:
- **`src/bot.py`** — `TradingBot` class: pluggable strategies, trailing/fixed stop-loss, long+short position tracking. `_run_strategy_logic(price)` for close-only strategies; `_run_strategy_logic_bar(bar)` for OHLCV strategies. Signals: `'buy'`/`'sell'`/`'short'`/`'hold'`.
- **`src/exchange.py`** — Simulated `fetch_price()` and `place_order()` (placeholders for a real exchange API like ccxt).
- **`src/backtest.py`** — Downloads historical data via yfinance, primes bot with warmup history, then simulates day-by-day. Supports long and short trades. Pass `strategy=` to use non-default strategies. Computes PnL, win/loss ratio, win rate.
- **`src/bar_permute.py`** — OHLCV bar permutation (`get_permutation`) for Monte Carlo Permutation Testing. Shuffles bars in log-space preserving single-bar statistics.
- **`src/indicators.py`** — Indicator helper functions (`compute_atr`, `compute_adx`, `compute_rolling_max`, `compute_rolling_min`, `compute_pivot_points`) returning scalar values for bar-by-bar use.
- **`src/optimize.py`** — Grid search parameter optimization (`optimize_strategy`, `optimize_strategy_fast`). Currently MA-specific.
- **`src/utils.py`** — `configure_logging()` helper (single canonical definition).
- **`src/strategies/`** — Pluggable strategy classes: `BaseStrategy` ABC (with OHLCV support via `on_bar`/`requires_ohlcv`), `MACrossoverStrategy`, `ATRBreakoutStrategy` (ADX + ATR breakout with long/short and internal SL/TP), `PivotPointStrategy` (floor pivot bounce/rejection with long/short).

```
bt/                   # backtesting.py framework package
├── __init__.py          # Package docstring
├── __main__.py          # python -m bt entrypoint
├── runner.py            # CLI runner, run_bt(), optimize_bt(), load_data()
├── helpers.py           # Shared indicator functions (sma, atr, adx, pivot_points, rolling_max/min) using ta library
└── strategies/          # backtesting.py Strategy subclasses
    ├── __init__.py      # Re-exports all strategies
    ├── ma_crossover.py  # MACrossoverBT (SMA crossover, long only)
    ├── atr_breakout.py  # ATRBreakoutStrategy (ADX + ATR breakout with SL/TP)
    └── pivot_points.py  # PivotPointStrategy (floor pivot bounce/rejection)
```

Key bt modules:
- **`bt/runner.py`** — CLI and programmatic runner. `load_data()` fetches from yfinance or local NQ CSV. `run_bt()` / `optimize_bt()` execute single runs or grid optimization. Strategy registry maps names to classes.
- **`bt/helpers.py`** — Indicator functions (`sma`, `atr`, `adx`, `pivot_points`, `rolling_max`, `rolling_min`) returning numpy arrays for use with `self.I()`. Uses the `ta` library for ATR/ADX.
- **`bt/strategies/`** — Three strategies: `MACrossoverBT` (SMA 5/20 crossover), `ATRBreakoutStrategy` (ADX filter + ATR breakout with ATR-scaled SL/TP), `PivotPointStrategy` (S1/R1 bounce with optional S2/R2 levels).

Other directories:
- **`tests/`** — pytest tests (`test_bt_strategies.py` for bt/ smoke tests, `test_mcpt.py` for MCPT/optimization tests).
- **`scripts/`** — Standalone CLI tools and demos (MCPT validation, NQ backtest, trailing stop demo, yfinance logger).
- **`archive/`** — Original standalone template (`trading_bot_template.py`), not imported.
- **`pinescript_ideas/`** — TradingView PineScript strategy files (not part of the Python bot).

## Key Design Details

- `MACrossoverStrategy` maintains a rolling `price_history` list capped to `long_window` (default 20). Moving averages are computed from this list using pandas rolling.
- Position entry/exit is tracked via `_handle_position_entry()` / `_handle_position_exit()` which manage `entry_price`, `highest_price`, and `stop_loss_price`.
- The backtest engine manages `has_position` externally (checking it before acting on signals) rather than letting the bot auto-execute trades.
- Historical data is loaded via `load_historical_data(data=None)` — pass a DataFrame for backtest, or `None` to fetch from yfinance.
- Dependencies: `pandas`, `yfinance`, `numpy`, `backtesting`, `ta`, `matplotlib`, `tqdm`, `pytest` (see `requirements.txt`).
