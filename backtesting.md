# Backtesting System Analysis & Improvement Roadmap

> Generated: 2026-02-15 | Covers both `src/` (custom engine) and `bt/` (backtesting.py framework)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Current Strengths](#current-strengths)
3. [Strategy Summary](#strategy-summary)
4. [Metrics & Statistics](#metrics--statistics)
5. [Known Issues & Bugs](#known-issues--bugs)
6. [Missing Features](#missing-features)
7. [Improvement Roadmap](#improvement-roadmap)

---

## Architecture Overview

The project has **two independent backtesting engines**:

### Custom Engine (`src/`)

```
src/backtest.py    → Day-by-day simulation loop, equity tracking, trade pairing
src/bot.py         → TradingBot: position management, trailing/fixed stop-loss
src/strategies/    → Pluggable strategies (MA Crossover, ATR Breakout, Pivot Points)
src/optimize.py    → Grid search, walk-forward analysis, rolling walk-forward
src/metrics.py     → Sharpe, Sortino, CAGR, max drawdown, profit factor
src/indicators.py  → ATR, ADX, pivot points, rolling max/min
src/bar_permute.py → OHLCV bar permutation for Monte Carlo Permutation Testing
src/multi_asset.py → Sequential backtesting across multiple symbols
```

**Flow**: Load data (yfinance) → warmup period (prime indicators) → simulate bar-by-bar → generate signals → execute fills → track equity → compute metrics.

### backtesting.py Framework (`bt/`)

```
bt/runner.py       → CLI wrapper, run_bt(), optimize_bt(), load_data()
bt/helpers.py      → Indicator functions (SMA, ATR, ADX, pivots) using ta library
bt/strategies/     → 3 Strategy subclasses wrapping backtesting.py
```

Built on the mature `backtesting` library. Provides fractional shares, built-in optimization, and interactive plotting.

---

## Current Strengths

### Well-Designed Areas

| Area | Details |
|------|---------|
| **Modular strategies** | Clean ABC with `requires_ohlcv` property. Easy to add new strategies. |
| **Dual stop-loss** | Both trailing % and fixed % stop-loss supported for long and short. |
| **Three fill models** | `close`, `next_open`, `vwap_slippage` — realistic execution simulation. |
| **Commission & slippage** | Tracked separately, deducted from cash on each trade. |
| **Walk-forward analysis** | Train/test split with overfit ratio detection. Rolling windows too. |
| **MCPT validation** | Bar permutation in log-space preserves single-bar statistics. |
| **Comprehensive metrics** | 10+ metrics including risk-adjusted (Sharpe, Sortino, max drawdown). |
| **Test coverage (~70%)** | 167 tests across 12 files. Fill models, metrics, optimization, and edge cases well tested. |

### Code Quality: 7/10

Well-structured with clear separation of concerns. Strategy pattern is clean and extensible. Main gaps are in edge case handling and state synchronization.

---

## Strategy Summary

| Feature | MA Crossover | ATR Breakout | Pivot Points |
|---------|:---:|:---:|:---:|
| Price-only mode | Yes | No | No |
| OHLCV mode | Yes | Yes | Yes |
| Long positions | Yes | Yes | Yes |
| Short positions | Optional | Yes | Yes |
| Internal SL/TP | No | Yes (ATR-scaled) | Yes (pivot-based) |
| Both engines | Yes | Yes | Yes |
| Cooldown logic | No | Yes (5-bar) | No |

**MA Crossover**: Buy when SMA(5) > SMA(20), sell on reversal. Simple trend-following.

**ATR Breakout**: ADX > threshold + price breaks high/low ± ATR×multiplier. Directional via +DI/-DI. SL = 2×ATR, TP = 3×ATR.

**Pivot Points**: S1 bounce (long) and R1 rejection (short). SL at S2/R2, TP at pivot point.

---

## Metrics & Statistics

### Currently Computed

**Trade-level**: Individual PnL, win/loss, gross profits/losses, win rate, profit factor, exit reason, bars held, entry price.

**Portfolio-level**: Total PnL (after costs), total commission, trade count, equity curve, expectancy.

**Enhanced stats**: Consecutive wins/losses (max streak), largest win/loss, average bars held, exit reason counts.

**Risk-adjusted** (from `src/metrics.py`):
- Sharpe ratio (annualized, 252 trading days)
- Sortino ratio (downside deviation only)
- Max drawdown (%) and drawdown duration (bars)
- CAGR and annualized return
- Calmar ratio (CAGR / |max drawdown|)
- Buy-and-hold comparison

**Exit reason tracking**: Strategies and bot return `('sell', reason)` tuples. Reasons: `'signal_reversal'`, `'sl_hit'`, `'tp_hit'`, `'trailing_sl_hit'`, `'fixed_sl_hit'`, `'end_of_data'`. Backward compatible via `_unpack_signal()` helper.

### Not Yet Computed

- Recovery factor
- Ulcer index
- Tail ratio

---

## Known Issues & Bugs

### Critical (P0)

#### ~~1. Position State Duplication~~ (FIXED)
`position_type_tracker` removed from `backtest.py`. `bot.position_type` is the sole source of truth.

#### ~~2. Open Positions Not Force-Closed at End of Backtest~~ (FIXED)
Force-close logic added for both long and short positions. Tested in `test_open_position_is_force_closed`.

#### ~~3. Strategy-Internal SL/TP Not Validated Against Bot State~~ (FIXED)
Strategies now return `('sell', reason)` tuples with exit reasons (`'sl_hit'`, `'tp_hit'`, `'signal_reversal'`). Bot stop-losses return `('sell', 'trailing_sl_hit')` or `('sell', 'fixed_sl_hit')`. Backtest engine tracks exit reasons per trade via `_unpack_signal()` helper.

### Important (P1)

#### ~~4. Profit Factor = Infinity Handled Inconsistently~~ (FIXED)
Capped at 999.99 when `gross_losses == 0`. Tested in `test_profit_factor_not_infinite`.

#### ~~5. Pending Signal Lost on Conflict~~ (FIXED)
Warning now logged when pending signal conflicts with existing position ("Dropped pending" message).

#### ~~6. Warmup Logic Inconsistent~~ (FIXED)
Unified: always uses `max(strategy.warmup_period, long_window)` for all strategies.

### Minor (P2)

#### 7. Metrics Assume Daily Bars
`periods_per_year=252` is hard-coded. Breaks for intraday or weekly backtests.

#### 8. Buy-and-Hold Comparison Uses Simulation Start
Doesn't account for warmup period consistently. May penalize B&H unfairly if warmup has downtrend.

#### 9. Drawdown Duration Doesn't Track Recovery
Only counts consecutive bars below peak. Doesn't distinguish partial vs. full recovery.

---

## Missing Features

### Essential for Production

| Feature | Why It Matters |
|---------|---------------|
| ~~**Trade attribution**~~ | **Done.** Exit reasons tracked per trade: `sl_hit`, `tp_hit`, `signal_reversal`, `trailing_sl_hit`, `fixed_sl_hit`, `end_of_data`. |
| ~~**Position sizing**~~ | **Done.** Volatility-scaled (`ATR`) and rolling-std sizing in `src/position_sizing.py`. Max portfolio risk cap. |
| ~~**OHLC validation**~~ | **Done.** Warns on invalid bars. Test data generators (`make_ohlcv`, `make_trending_ohlcv`) fixed to produce valid OHLC. |
| ~~**Intraday support**~~ | **Done.** `periods_per_year` param on all functions, `infer_periods_per_year()` auto-detection. |
| ~~**Parameter sensitivity**~~ | **Done.** `src/sensitivity.py` with `analyze_sensitivity()` — varies params ±10-20%, reports stability scores. |
| **Regime detection** | No market regime identification. Strategy may work in trends but fail in chop. |

### Important for Robustness

| Feature | Why It Matters |
|---------|---------------|
| **Stress testing** | No testing against specific market scenarios (crashes, flash crashes, sideways). |
| **Portfolio backtesting** | Only per-symbol. No correlation-aware multi-asset portfolio construction. |
| **Live/backtest alignment** | Backtest has slippage/commission; live exchange.py doesn't. Results will diverge. |
| **Bid-ask spread modeling** | Commission is flat %. Real markets have variable spreads. |
| **Data quality checks** | No outlier detection, no gap handling, no split/dividend adjustment. |

### Nice-to-Have Enhancements

| Feature | Details |
|---------|---------|
| ~~ATR-based trailing stops~~ | **Done.** `trailing_stop_atr` param on TradingBot. Trails by N × ATR, recalculated each bar. |
| ~~Breakeven stops~~ | **Done.** `breakeven_threshold` param. Moves SL to entry after configurable profit %. |
| Early stopping in optimization | Grid search runs all combos. Could skip unpromising regions. |
| Parallel optimization | Single-threaded. `multiprocessing` would speed up significantly. |
| Vectorized indicators | ATR/Pivot create new Series every bar. Rolling buffers would be faster. |

---

## Improvement Roadmap

### Phase 1: Bug Fixes & Hardening ✅

- [x] **Unify position tracking** — `position_type_tracker` removed; `bot.position_type` is sole source of truth
- [x] **Force-close positions at backtest end** — long and short positions force-closed; tested
- [x] **Cap profit factor** — capped at 999.99; tested
- [x] **Validate OHLC structure** — warns on invalid bars at backtest start
- [x] **Reset strategy state between backtests** — `multi_asset.py` calls `strategy.reset()` between symbols
- [x] **Log dropped signals** — warns when pending signal conflicts with existing position
- [x] **Fix `conftest.make_ohlcv`** — High/Low now computed relative to `max(Open, Close)` / `min(Open, Close)`; 0 invalid bars

### Phase 2: Trade Analytics & Attribution ✅

- [x] **Exit reason tracking** — strategies/bot return `('sell', reason)` tuples; trade dicts include `exit_reason`, `bars_held`, `entry_price`
- [x] **Trade duration** — `bars_held` tracked per trade via `entry_bar_idx` in simulation loop
- [x] **Enhanced trade stats** — `consecutive_wins`, `consecutive_losses`, `largest_win`, `largest_loss`, `avg_bars_held`, `exit_reason_counts`
- [x] **Expectancy calculation** — `(avg_win × win_rate) - (avg_loss × loss_rate)` in backtest results
- [x] **Calmar ratio** — `compute_calmar_ratio()` added to `src/metrics.py`, included in `compute_all_metrics()`

### Phase 3: Position Sizing & Risk Management ✅

- [x] **Volatility-scaled sizing** — `src/position_sizing.py` with `volatility_scaled_size()` (ATR-based) and `rolling_std_size()`
- [x] **Max portfolio risk** — `cap_by_max_risk()` limits total exposure as % of equity
- [x] **ATR trailing stops** — `trailing_stop_atr` param on TradingBot, recalculated each bar
- [x] **Breakeven stops** — `breakeven_threshold` param, moves SL to entry after configurable profit %

### Phase 4: Robustness & Validation ✅

- [x] **Parameter sensitivity analysis** — `src/sensitivity.py` with `analyze_sensitivity()`, per-param sensitivity scores
- [ ] **Regime-aware backtesting** — classify market regimes (trend/range/volatile) and report per-regime performance
- [ ] **Stress testing** — test against 2008 crash, COVID crash, flash crash scenarios
- [x] **Configurable timeframes** — `periods_per_year` param + `infer_periods_per_year()` auto-detection (5m, 15m, 1h, daily, weekly)
- [x] **Walk-forward as standard workflow** — `walk_forward=True` param on `run_backtest_on_data()`

### Phase 5: Portfolio & Production

- [ ] **Multi-asset portfolio backtesting** — correlation-aware, capital allocation across symbols
- [ ] **Live/backtest parity** — add slippage and commission to live exchange.py
- [ ] **Bid-ask spread modeling** — variable spreads based on volatility/time-of-day
- [ ] **Data quality pipeline** — outlier detection, gap filling, split adjustment
- [ ] **Performance optimization** — vectorized indicators, parallel grid search, caching

---

## Test Coverage

### Well Tested (265 tests, 19 files)
- Fill models (close, next_open, vwap_slippage) — including model comparison tests
- Slippage & commission math
- Equity curve tracking and initial capital
- Bar permutation (MCPT) — shape, columns, reproducibility, positivity
- All metric calculations (Sharpe, Sortino, CAGR, drawdown, buy-and-hold, Calmar ratio)
- Optimization and walk-forward parameter validation
- Force-close open positions at backtest end (long)
- OHLC validation warnings on bad data
- Cross-engine validation (src/ vs bt/)
- Multi-asset backtest structure
- All 3 strategies: unit tests + smoke tests in both engines
- Trade analytics: exit reasons, bars held, enhanced stats, expectancy, `_unpack_signal()`
- Position sizing models: fixed, volatility-scaled, rolling-std, max risk cap
- ATR trailing stops and breakeven stops (long and short)
- Parameter sensitivity analysis
- Configurable timeframes and periods_per_year passthrough
- Walk-forward integration into `run_backtest_on_data()`
- Robustness: edge cases, overlapping signals, state leakage, empty data

### Known Test Issues
- ~~`conftest.make_ohlcv` produces invalid OHLC~~ **Fixed.** High/Low now bracket `max(Open, Close)` / `min(Open, Close)`. All generators produce valid OHLC data.
- Cross-engine tests use loose tolerances (50% trade count, 25% win rate).

### Needs More Tests
- [ ] `next_open` model edge cases (last bar, gap bars)
- [ ] Real yfinance data (all current tests use synthetic data)
- [ ] Kelly criterion sizing (not yet implemented)

---

## Quick Reference: Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/backtest.py` | ~510 | Main simulation engine (with trade analytics) |
| `src/bot.py` | ~194 | Position & risk management |
| `src/optimize.py` | ~329 | Parameter optimization |
| `src/metrics.py` | ~255 | Risk metrics (incl. Calmar ratio) |
| `src/strategies/base.py` | ~93 | Strategy ABC (sell tuples in return types) |
| `src/strategies/ma_crossover.py` | ~78 | MA crossover strategy |
| `src/strategies/atr_breakout.py` | ~207 | ATR breakout strategy |
| `src/strategies/pivot_points.py` | ~171 | Pivot point strategy |
| `src/multi_asset.py` | ~90 | Multi-symbol backtesting |
| `src/bar_permute.py` | ~70 | MCPT bar permutation |
| `src/position_sizing.py` | ~80 | Position sizing models (volatility, rolling-std, max risk) |
| `src/sensitivity.py` | ~130 | Parameter sensitivity analysis |
| `tests/test_backtest.py` | ~380 | Backtest engine tests |
| `tests/test_trade_analytics.py` | ~340 | Trade analytics & attribution tests |
| `tests/test_metrics.py` | ~175 | Metrics tests |
| `tests/conftest.py` | ~113 | Shared fixtures & data generators |
