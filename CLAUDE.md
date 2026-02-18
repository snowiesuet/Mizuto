# CLAUDE.md

## Project Overview

Mizuto is a Python algorithmic trading bot with two backtesting engines: custom (`src/`) and `backtesting.py`-based (`bt/`). Phase 3 complete, ready for Phase 4 (ML model). See [roadmap.md](roadmap.md) for full plan, [backtesting.md](backtesting.md) for system analysis.

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

- `src/` — Core library: `bot.py` (decision engine), `backtest.py` (simulation), `metrics.py`, `optimize.py`, `indicators.py`, `bar_permute.py` (MCPT), `multi_asset.py`, `position_sizing.py`, `sensitivity.py`, `strategies/` (BaseStrategy ABC + 3 strategies)
- `bt/` — backtesting.py wrapper: `runner.py` (CLI), `helpers.py` (indicators via `ta`), `strategies/` (3 Strategy subclasses)
- `scripts/` — CLI tools (MCPT validation, NQ backtest, trailing stop demo)
- `tests/` — pytest tests
- `archive/` — Original template, not imported
- `pinescript_ideas/` — TradingView PineScript, not part of Python bot

## Key Design Patterns

- **Strategy signals**: `'buy'` / `'short'` / `'hold'` are bare strings. Sell signals are tuples: `('sell', 'reason')` where reason is `'signal_reversal'`, `'sl_hit'`, `'tp_hit'`, `'trailing_sl_hit'`, or `'fixed_sl_hit'`. Use `_unpack_signal()` in `backtest.py` for backward-compatible unpacking.
- **Position tracking**: `TradingBot` manages entry/exit via `_handle_position_entry()` / `_handle_position_exit()`. Backtest manages `has_position` externally.
- **Trade attribution**: Sell trade dicts include `exit_reason`, `bars_held`, and `entry_price`. Backtest results include `expectancy`, `consecutive_wins/losses`, `largest_win/loss`, `avg_bars_held`, `exit_reason_counts`, `calmar_ratio`.
- **MA crossover**: Rolling `price_history` list capped to `long_window` (default 20). MAs computed via pandas rolling.
- **Historical data**: `load_historical_data(data=None)` — pass DataFrame for backtest, `None` for yfinance fetch.
- **Fill models**: `close`, `next_open`, `vwap_slippage` — configured per backtest run.
- **Position sizing**: `position_sizing='volatility'|'rolling_std'|None` on `run_backtest_on_data()`. Configurable `risk_per_trade`, `sizing_atr_multiplier`, `max_portfolio_risk`. Per-trade amounts stored in trade dicts.
- **ATR trailing stop**: `trailing_stop_atr` param on TradingBot — trails by N × ATR, recalculated each bar. Takes precedence over `trailing_stop_pct`.
- **Breakeven stop**: `breakeven_threshold` param — moves SL to entry price after configurable profit %.
- **Pivot point timeframes**: `PivotPointStrategy(pivot_timeframe=...)` controls higher-timeframe pivot source. `'auto'` (default) infers from data frequency: intraday→daily pivots, daily→weekly, weekly→monthly. Explicit `'D'`/`'W'`/`'M'` or `None` (legacy single-bar). Bar dicts include `'Timestamp'` key for period boundary detection; bars without timestamps fall back to legacy behavior.
- **Timeframes**: `periods_per_year` param flows through backtest → metrics → optimize. `'auto'` triggers `infer_periods_per_year()`.
- **Walk-forward integration**: `walk_forward=True` on `run_backtest_on_data()` delegates to `walk_forward_optimize()`.
- **Sensitivity analysis**: `analyze_sensitivity()` in `src/sensitivity.py` — varies params ±10-20%, reports per-param sensitivity scores.
