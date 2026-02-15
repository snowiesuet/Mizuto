# Mizuto

*mizu (水, fluidity) + to (ト, trade) — smooth and adaptive*

<img width="300" height="300" alt="mizuto" src="https://github.com/user-attachments/assets/6e340722-2c82-41c6-9b05-3c618b16c308" />

A Python algorithmic trading bot supporting live (simulated) trading and historical backtesting. Built with two independent backtesting engines, pluggable strategies, and overfitting defenses (walk-forward analysis, Monte Carlo permutation testing).

Currently in **Phase 2** (backtesting complete). See [roadmap.md](roadmap.md) for the full plan and [backtesting.md](backtesting.md) for a detailed analysis of the backtesting system.

## Features

- **3 strategies**: MA crossover, ATR breakout, pivot points — all support long and short positions
- **Dual backtesting engines**: custom bar-by-bar simulator (`src/`) and [backtesting.py](https://kernc.github.io/backtesting.py/) framework (`bt/`)
- **Risk management**: trailing and fixed stop-loss, commission and slippage modeling
- **3 fill models**: close, next-open, VWAP with volume-scaled slippage
- **Parameter optimization**: grid search, walk-forward analysis, rolling walk-forward
- **Validation**: Monte Carlo permutation testing (MCPT) with bar shuffling in log-space
- **10+ metrics**: Sharpe, Sortino, CAGR, max drawdown, profit factor, equity curves, buy-and-hold comparison
- **Multi-asset**: sequential backtesting across symbols

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows (use source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
```

**Dependencies**: pandas, numpy, yfinance, backtesting, ta, matplotlib, tqdm, pytest

## Usage

### Live Trading (Simulated)

```bash
python main.py
```

Runs the bot with a simulated exchange, polling every 60 seconds.

### Custom Backtest Engine

```bash
# Default BTC-USD backtest (dates/params configurable in src/backtest.py __main__ block)
python -m src.backtest
```

### backtesting.py Framework

```bash
python -m bt --strategy ma_crossover --symbol BTC-USD --start 2023-01-01 --end 2023-12-31
python -m bt --strategy atr_breakout --optimize
python -m bt --strategy pivot_points --plot
```

### Validation & Scripts

```bash
# Monte Carlo permutation testing
python scripts/mcpt_validation.py
python scripts/mcpt_validation.py --symbol BTC-USD --walkforward --save-plots

# NQ futures backtest
python scripts/backtest_nq.py --timeframe 1d --years 2023 2024

# Trailing stop-loss demo
python scripts/trailing_stop_demo.py
```

### Tests

```bash
pytest tests/
```

## Strategies

| Strategy | Signal Logic | Data | Positions | Internal SL/TP |
|----------|-------------|------|-----------|-----------------|
| **MA Crossover** | SMA(5) crosses SMA(20) | Close | Long (short optional) | No |
| **ATR Breakout** | ADX filter + price breaks high/low ± ATR | OHLCV | Long & Short | ATR-scaled |
| **Pivot Points** | S1 bounce (long) / R1 rejection (short) | OHLCV | Long & Short | Pivot-based |

All strategies implement a common `BaseStrategy` ABC and work with both backtesting engines.

## Project Structure

```
src/                  # Core library
├── bot.py               # TradingBot (decision engine, stop-loss, position tracking)
├── backtest.py           # Historical backtesting via yfinance
├── metrics.py            # Sharpe, Sortino, CAGR, max drawdown, profit factor
├── optimize.py           # Grid search, walk-forward, rolling walk-forward
├── bar_permute.py        # OHLCV bar permutation for MCPT
├── indicators.py         # ATR, ADX, pivots, rolling max/min
├── multi_asset.py        # Sequential multi-symbol backtesting
├── exchange.py           # Simulated exchange (fetch_price / place_order)
├── data_loader.py        # NQ futures CSV loader
├── utils.py              # configure_logging helper
└── strategies/           # Pluggable strategy classes
    ├── base.py              # BaseStrategy ABC
    ├── ma_crossover.py      # MACrossoverStrategy
    ├── atr_breakout.py      # ATRBreakoutStrategy
    └── pivot_points.py      # PivotPointStrategy

bt/                   # backtesting.py framework
├── runner.py            # CLI runner, run_bt(), optimize_bt()
├── helpers.py           # Indicator functions (sma, atr, adx, pivots)
└── strategies/          # backtesting.py Strategy subclasses

scripts/              # Standalone CLI tools
├── mcpt_validation.py   # Monte Carlo permutation testing
├── backtest_nq.py       # NQ futures backtest
├── trailing_stop_demo.py
└── log_yfinance_data.py

tests/                # pytest tests
```

## Architecture

The bot has three execution paths:

1. **Live path**: `main.py` → `TradingBot.run_strategy()` → simulated exchange
2. **Custom backtest**: `src/backtest.py` → bar-by-bar simulation with equity tracking
3. **backtesting.py**: `bt/runner.py` → `backtesting.Backtest` with pluggable strategies

```
Historical Data (yfinance)
  │
  ├─→ Custom Engine (src/backtest.py)
  │     ├─→ Warmup period (prime indicators)
  │     ├─→ Bar-by-bar simulation (signal → fill → track)
  │     ├─→ Equity curve + trade log
  │     └─→ Metrics (src/metrics.py)
  │
  └─→ backtesting.py (bt/runner.py)
        ├─→ Strategy subclasses with self.I() indicators
        ├─→ Built-in optimization + plotting
        └─→ Fractional position sizing
```

## License

See repository for license details.
