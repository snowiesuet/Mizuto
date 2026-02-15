# CLAUDE.md

## Project Overview

Mizuto is a Python algorithmic trading bot with two backtesting engines: custom (`src/`) and `backtesting.py`-based (`bt/`). Currently in Phase 3 (hardening & trade analytics). See [roadmap.md](roadmap.md) for full plan, [backtesting.md](backtesting.md) for system analysis and known bugs.

## Commands

```bash
python main.py              # Live trading (simulated)
python -m src.backtest      # Custom backtest engine
python -m bt --strategy ma_crossover --symbol BTC-USD --start 2023-01-01 --end 2023-12-31
pytest tests/               # Run tests
```

No linter or build system configured. Windows dev environment (`venv\Scripts\activate`).

## Architecture (3 execution paths)

- **Live**: `main.py` → `TradingBot.run_strategy()` → `exchange.fetch_price()` / `place_order()`
- **Custom backtest**: `src/backtest.py` → `TradingBot._run_strategy_logic()` (bypasses exchange)
- **backtesting.py**: `bt/runner.py` → `backtesting.Backtest` with pluggable `bt.strategies`

## Package Layout

- `src/` — Core library: `bot.py` (decision engine), `backtest.py` (simulation), `metrics.py`, `optimize.py`, `indicators.py`, `bar_permute.py` (MCPT), `multi_asset.py`, `strategies/` (BaseStrategy ABC + 3 strategies)
- `bt/` — backtesting.py wrapper: `runner.py` (CLI), `helpers.py` (indicators via `ta`), `strategies/` (3 Strategy subclasses)
- `scripts/` — CLI tools (MCPT validation, NQ backtest, trailing stop demo)
- `tests/` — pytest tests
- `archive/` — Original template, not imported
- `pinescript_ideas/` — TradingView PineScript, not part of Python bot

## Key Design Patterns

- **Strategy signals**: `'buy'` / `'sell'` / `'short'` / `'hold'`. Strategies implement `BaseStrategy` ABC with `on_bar()`/`requires_ohlcv` for OHLCV data.
- **Position tracking**: `TradingBot` manages entry/exit via `_handle_position_entry()` / `_handle_position_exit()`. Backtest manages `has_position` externally.
- **MA crossover**: Rolling `price_history` list capped to `long_window` (default 20). MAs computed via pandas rolling.
- **Historical data**: `load_historical_data(data=None)` — pass DataFrame for backtest, `None` for yfinance fetch.
- **Fill models**: `close`, `next_open`, `vwap_slippage` — configured per backtest run.
