# Mizuto

*mizu (水, fluidity) + to (ト, trade) — smooth and adaptive*

<img width="300" height="300" alt="mizuto" src="https://github.com/user-attachments/assets/6e340722-2c82-41c6-9b05-3c618b16c308" />

A Python algorithmic trading bot supporting live (simulated) trading and historical backtesting. Features pluggable strategies (MA crossover, ATR breakout, pivot points), trailing/fixed stop-loss, long and short positions, and Monte Carlo permutation testing for strategy validation.

Currently in **Phase 2** (backtesting complete). See [roadmap.md](roadmap.md) for the full development plan.

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Usage

```bash
# Live trading bot (simulated exchange, polls every 60s)
python main.py

# Custom backtest (BTC-USD, configurable in src/backtest.py __main__ block)
python -m src.backtest

# backtesting.py strategies
python -m bt --strategy ma_crossover --symbol BTC-USD --start 2023-01-01 --end 2023-12-31
python -m bt --strategy atr_breakout --optimize
python -m bt --strategy pivot_points --plot

# MCPT validation
python scripts/mcpt_validation.py
python scripts/mcpt_validation.py --symbol BTC-USD --walkforward --save-plots

# NQ futures backtest
python scripts/backtest_nq.py --timeframe 1d --years 2023 2024

# Run tests
pytest tests/
```

## Project Structure

```
src/                  # Core library
├── bot.py               # TradingBot (decision engine, stop-loss, position tracking)
├── backtest.py           # Historical backtesting via yfinance
├── bar_permute.py        # OHLCV bar permutation for MCPT
├── indicators.py         # ATR, ADX, pivots, rolling max/min
├── optimize.py           # Grid search over MA / stop-loss params
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

## License

See repository for license details.
