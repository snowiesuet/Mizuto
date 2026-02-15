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

**Trade-level**: Individual PnL, win/loss, gross profits/losses, win rate, profit factor.

**Portfolio-level**: Total PnL (after costs), total commission, trade count, equity curve.

**Risk-adjusted** (from `src/metrics.py`):
- Sharpe ratio (annualized, 252 trading days)
- Sortino ratio (downside deviation only)
- Max drawdown (%) and drawdown duration (bars)
- CAGR and annualized return
- Buy-and-hold comparison

### Not Yet Computed

- Trade duration (time in position)
- Exit reason attribution (SL vs TP vs signal reversal)
- Consecutive win/loss streaks
- Largest single win/loss
- Calmar ratio (CAGR / max drawdown)
- Recovery factor
- Expectancy (avg win × win rate - avg loss × loss rate)
- Ulcer index
- Tail ratio

---

## Known Issues & Bugs

### Critical (P0)

#### 1. Position State Duplication
**Where**: `src/backtest.py` maintains `position_type_tracker` AND `src/bot.py` maintains `bot.position_type`.
**Risk**: State desynchronization — bot thinks one thing, backtest thinks another.
**Fix**: Single source of truth. Backtest should read from bot, not track separately.

#### ~~2. Open Positions Not Force-Closed at End of Backtest~~ (FIXED)
Force-close logic added for both long and short positions. Tested in `test_open_position_is_force_closed`.

#### 3. Strategy-Internal SL/TP Not Validated Against Bot State
**Where**: ATR Breakout and Pivot strategies manage their own SL/TP.
**Risk**: Strategy emits `'sell'` for its internal SL hit, but bot doesn't know why. If strategy logic is wrong, bot blindly follows.
**Fix**: Have strategies report exit reasons; bot validates state consistency.

### Important (P1)

#### ~~4. Profit Factor = Infinity Handled Inconsistently~~ (FIXED)
Capped at 999.99 when `gross_losses == 0`. Tested in `test_profit_factor_not_infinite`.

#### ~~5. Pending Signal Lost on Conflict~~ (FIXED)
Warning now logged when pending signal conflicts with existing position ("Dropped pending" message).

#### 6. Warmup Logic Inconsistent
**Where**: `src/backtest.py:126-127` — OHLCV strategies use their own `warmup_period`, price-only strategies use `max(warmup, long_window)`.
**Why**: Different treatment is confusing and could lead to under-primed indicators.
**Fix**: Unify: always use `max(strategy.warmup_period, long_window)`.

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
| **Trade attribution** | Know *why* each trade exited (SL, TP, signal, time-based). Essential for strategy tuning. |
| **Position sizing** | Always fixed `trade_amount`. Need Kelly criterion, volatility scaling, or risk-parity. |
| ~~**OHLC validation**~~ | **Done.** Warns on invalid bars. Note: `conftest.make_ohlcv` still produces invalid OHLC data — OHLCV strategies (ATR Breakout, Pivot) cannot reliably generate trades with it. |
| **Intraday support** | `periods_per_year=252` hard-coded. Need configurable timeframes (1h, 15m, 5m). |
| **Parameter sensitivity** | No analysis of how small param changes affect results. Fragile params = overfitting. |
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
| ATR-based trailing stops | Currently only fixed %. ATR-based would adapt to volatility. |
| Breakeven stops | Move SL to entry after N% profit. Common risk management technique. |
| Early stopping in optimization | Grid search runs all combos. Could skip unpromising regions. |
| Parallel optimization | Single-threaded. `multiprocessing` would speed up significantly. |
| Vectorized indicators | ATR/Pivot create new Series every bar. Rolling buffers would be faster. |

---

## Improvement Roadmap

### Phase 1: Bug Fixes & Hardening (Priority)

- [ ] **Unify position tracking** — remove `position_type_tracker` from backtest.py, use `bot.position_type` as sole source of truth
- [x] **Force-close positions at backtest end** — long and short positions force-closed; tested
- [x] **Cap profit factor** — capped at 999.99; tested
- [x] **Validate OHLC structure** — warns on invalid bars at backtest start
- [ ] **Reset strategy state between backtests** — ensure `multi_asset.py` always calls reset
- [x] **Log dropped signals** — warns when pending signal conflicts with existing position
- [ ] **Fix `conftest.make_ohlcv`** — helper produces invalid OHLC data (High < Close, Low > Close). OHLCV strategies can't generate trades with it.

### Phase 2: Trade Analytics & Attribution

- [ ] **Exit reason tracking** — add `exit_reason` field to trades: `'sl_hit'`, `'tp_hit'`, `'signal_reversal'`, `'end_of_data'`
- [ ] **Trade duration** — track bars held for each trade
- [ ] **Enhanced trade stats** — consecutive wins/losses, largest win/loss, average hold time
- [ ] **Expectancy calculation** — `(avg_win × win_rate) - (avg_loss × loss_rate)`
- [ ] **Calmar ratio** — CAGR / max drawdown

### Phase 3: Position Sizing & Risk Management

- [ ] **Volatility-scaled sizing** — size positions based on ATR or rolling std
- [ ] **Kelly criterion** — optimal position size based on win rate and win/loss ratio
- [ ] **Max portfolio risk** — limit total exposure as % of equity
- [ ] **ATR trailing stops** — trail stop-loss by N × ATR instead of fixed %
- [ ] **Breakeven stops** — move SL to entry after configurable profit threshold

### Phase 4: Robustness & Validation

- [ ] **Parameter sensitivity analysis** — vary each param ±10-20% and measure metric stability
- [ ] **Regime-aware backtesting** — classify market regimes (trend/range/volatile) and report per-regime performance
- [ ] **Stress testing** — test against 2008 crash, COVID crash, flash crash scenarios
- [ ] **Configurable timeframes** — support intraday bars with correct annualization factors
- [ ] **Walk-forward as standard workflow** — integrate into default backtest runs, not just standalone

### Phase 5: Portfolio & Production

- [ ] **Multi-asset portfolio backtesting** — correlation-aware, capital allocation across symbols
- [ ] **Live/backtest parity** — add slippage and commission to live exchange.py
- [ ] **Bid-ask spread modeling** — variable spreads based on volatility/time-of-day
- [ ] **Data quality pipeline** — outlier detection, gap filling, split adjustment
- [ ] **Performance optimization** — vectorized indicators, parallel grid search, caching

---

## Test Coverage

### Well Tested (167 tests, 12 files)
- Fill models (close, next_open, vwap_slippage) — including model comparison tests
- Slippage & commission math
- Equity curve tracking and initial capital
- Bar permutation (MCPT) — shape, columns, reproducibility, positivity
- All metric calculations (Sharpe, Sortino, CAGR, drawdown, buy-and-hold)
- Optimization and walk-forward parameter validation
- Force-close open positions at backtest end (long)
- OHLC validation warnings on bad data
- Cross-engine validation (src/ vs bt/)
- Multi-asset backtest structure
- All 3 strategies: unit tests + smoke tests in both engines

### Known Test Issues
- `conftest.make_ohlcv` produces invalid OHLC (High < Close, Low > Close). OHLCV strategies (ATR Breakout, Pivot Points) cannot generate trades with it — integration tests for those strategies are smoke-only (no crash, but no trade assertions).
- Cross-engine tests use loose tolerances (50% trade count, 25% win rate).

### Needs More Tests
- [ ] Position tracking consistency between bot and strategies
- [ ] `next_open` model edge cases (last bar, gap bars)
- [ ] Strategy state leakage between backtest runs
- [ ] Real yfinance data (all current tests use synthetic data)
- [ ] ATR Breakout integration test with OHLCV-valid synthetic data

---

## Quick Reference: Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/backtest.py` | ~427 | Main simulation engine |
| `src/bot.py` | ~194 | Position & risk management |
| `src/optimize.py` | ~329 | Parameter optimization |
| `src/metrics.py` | ~237 | Risk metrics |
| `src/strategies/base.py` | ~60 | Strategy ABC |
| `src/strategies/ma_crossover.py` | ~80 | MA crossover strategy |
| `src/strategies/atr_breakout.py` | ~120 | ATR breakout strategy |
| `src/strategies/pivot_points.py` | ~110 | Pivot point strategy |
| `src/multi_asset.py` | ~90 | Multi-symbol backtesting |
| `src/bar_permute.py` | ~70 | MCPT bar permutation |
| `tests/test_backtest.py` | ~380 | Backtest engine tests |
| `tests/test_metrics.py` | ~175 | Metrics tests |
| `tests/conftest.py` | ~109 | Shared fixtures & data generators |
