# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mizuto is a Python algorithmic trading bot that uses a moving average crossover strategy with optional trailing/fixed stop-loss. It supports both live (simulated) trading and historical backtesting via yfinance data. The project is currently in Phase 2 of its roadmap (backtesting complete; ML-based prediction in Phase 3 is next).

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
python tests/mcpt_validation.py

# Fetch and log yfinance data for debugging
python scripts/log_yfinance_data.py

# Run tests
pytest tests/
```

There are no linter or build system configured.

## Architecture

The bot has two execution paths that share core strategy logic:

- **Live path**: `main.py` → `TradingBot.run_strategy()` → `exchange.fetch_price()` / `exchange.place_order()`
- **Backtest path**: `src/backtest.py` → `TradingBot._run_strategy_logic()` (called directly, bypassing exchange)

### Package layout

```
src/                  # Library package
├── __init__.py          # Re-exports TradingBot, configure_logging
├── bot.py               # TradingBot class (core decision engine)
├── exchange.py           # Simulated fetch_price / place_order
├── backtest.py           # Historical backtesting via yfinance
├── optimize.py           # Grid search over MA / stop-loss params
├── utils.py              # configure_logging helper
└── strategies/           # Pluggable strategy classes
    ├── __init__.py
    ├── base.py           # BaseStrategy ABC
    └── ma_crossover.py   # MACrossoverStrategy
```

Key modules:
- **`src/bot.py`** — `TradingBot` class: MA crossover strategy, trailing/fixed stop-loss, position tracking. The `_run_strategy_logic(price)` method is the core decision engine (returns `'buy'`/`'sell'`/`'hold'`). `run_strategy()` wraps it for live use; backtest calls `_run_strategy_logic` directly.
- **`src/exchange.py`** — Simulated `fetch_price()` and `place_order()` (placeholders for a real exchange API like ccxt).
- **`src/backtest.py`** — Downloads historical data via yfinance, primes bot with `LONG_WINDOW` days of history, then simulates day-by-day. Computes PnL, win/loss ratio, win rate.
- **`src/optimize.py`** — Grid search parameter optimization (`optimize_strategy`, `optimize_strategy_fast`).
- **`src/utils.py`** — `configure_logging()` helper (single canonical definition).
- **`src/strategies/`** — Pluggable strategy classes (`BaseStrategy` ABC, `MACrossoverStrategy`).

Other directories:
- **`tests/`** — pytest tests and MCPT validation script.
- **`scripts/`** — Standalone demo/utility scripts (trailing stop demo, yfinance logger).
- **`archive/`** — Original standalone template (`trading_bot_template.py`), not imported.
- **`pinescript_ideas/`** — TradingView PineScript strategy files (not part of the Python bot).

## Key Design Details

- `MACrossoverStrategy` maintains a rolling `price_history` list capped to `long_window` (default 20). Moving averages are computed from this list using pandas rolling.
- Position entry/exit is tracked via `_handle_position_entry()` / `_handle_position_exit()` which manage `entry_price`, `highest_price`, and `stop_loss_price`.
- The backtest engine manages `has_position` externally (checking it before acting on signals) rather than letting the bot auto-execute trades.
- Historical data is loaded via `load_historical_data(data=None)` — pass a DataFrame for backtest, or `None` to fetch from yfinance.
- Dependencies are minimal: `pandas` and `yfinance` only (see `requirements.txt`).
