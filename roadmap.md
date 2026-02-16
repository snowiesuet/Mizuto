# Algorithmic Trading Bot Development Roadmap

This roadmap outlines the key phases and steps to build, test, and deploy an algorithmic trading bot that uses a predictive model.

See [backtesting.md](backtesting.md) for the detailed backtesting system analysis that informed this roadmap.

---

### Phase 1: Foundation & Simple Strategy ✅

The goal of this phase is to build the basic scaffolding for the bot and implement a simple, rule-based strategy.

- [x] **Step 1.1: Setup Development Environment**
  - [x] Install Python and necessary libraries (`pandas`, `numpy`).
  - [x] Set up a version control system like Git.
  - [x] Choose an IDE (like VS Code).

- [x] **Step 1.2: Create a Basic Trading Bot Structure**
  - [x] Create separate modules for the bot logic, exchange interactions, and utilities.
  - [x] Implement placeholder functions for fetching prices and placing orders.

- [x] **Step 1.3: Implement a Simple Trading Strategy**
  - [x] Code a basic strategy, such as a moving average crossover or a simple price threshold system (like the one in `trading_bot_template.py`).

---

### Phase 2: Historical Backtesting ✅

This phase focuses on testing the simple strategy against historical data to gauge its viability.

- [x] **Step 2.1: Acquire Historical Market Data**
  - [x] Use `yfinance` to download historical OHLCV data.
  - [x] Support multiple symbols (BTC-USD, NQ futures via CSV).

- [x] **Step 2.2: Develop a Backtesting Engine**
  - [x] Custom bar-by-bar simulator (`src/backtest.py`) with equity tracking and trade pairing.
  - [x] Secondary engine using `backtesting.py` framework (`bt/`) with interactive plotting.
  - [x] Three fill models: close, next-open, VWAP with volume-scaled slippage.
  - [x] Commission and slippage modeling.

- [x] **Step 2.3: Implement Multiple Strategies**
  - [x] Pluggable strategy pattern via `BaseStrategy` ABC.
  - [x] MA Crossover (SMA 5/20, long-only default).
  - [x] ATR Breakout (ADX filter + ATR-scaled SL/TP, long & short).
  - [x] Pivot Points (S1/R1 bounce/rejection, long & short).

- [x] **Step 2.4: Analyze Backtest Results**
  - [x] Core metrics: PnL, win rate, profit factor.
  - [x] Risk-adjusted: Sharpe ratio, Sortino ratio, CAGR, max drawdown.
  - [x] Equity curve tracking with buy-and-hold comparison.

- [x] **Step 2.5: Parameter Optimization & Validation**
  - [x] Grid search over MA windows and stop-loss parameters.
  - [x] Walk-forward analysis with overfit ratio detection.
  - [x] Rolling walk-forward (multiple train/test windows).
  - [x] Monte Carlo permutation testing (MCPT) with bar shuffling in log-space.
  - [x] Multi-asset sequential backtesting.

---

### Phase 3: Backtesting Hardening & Trade Analytics ← **YOU ARE HERE**

Before building ML models, fix known issues and add the analytics needed to trust backtest results and understand strategy behavior. See [backtesting.md](backtesting.md) for full details.

- [x] **Step 3.1: Fix Known Bugs (P0)**
  - [x] Unify position tracking — remove `position_type_tracker` from `backtest.py`, use `bot.position_type` as sole source of truth.
  - [x] Force-close open positions at end of backtest — include unrealized PnL in final metrics.
  - [x] Cap profit factor — replace `float('inf')` with a capped value (999.99) when no losses.
  - [x] Log dropped signals — warn when pending signal conflicts with existing position in `next_open` model.

- [x] **Step 3.2: Data & State Integrity (P1)**
  - [x] Validate OHLC structure on load (`High >= max(Open, Close)`, `Low <= min(Open, Close)`).
  - [x] Reset strategy state properly between backtests (already working in `multi_asset.py`).
  - [x] Unify warmup logic — always use `max(strategy.warmup_period, long_window)`.
  - [x] Validate that selected strategy supports chosen fill model.

- [x] **Step 3.3: Trade Attribution & Analytics**
  - [x] Add exit reason tracking: strategies/bot return `('sell', reason)` tuples. Reasons: `'sl_hit'`, `'tp_hit'`, `'signal_reversal'`, `'trailing_sl_hit'`, `'fixed_sl_hit'`, `'end_of_data'`.
  - [x] Track trade duration (`bars_held` per trade via `entry_bar_idx`).
  - [x] Compute enhanced stats: `consecutive_wins`, `consecutive_losses`, `largest_win`, `largest_loss`, `avg_bars_held`, `exit_reason_counts`.
  - [x] Add expectancy: `(avg_win × win_rate) - (avg_loss × loss_rate)`.
  - [x] Add Calmar ratio (`compute_calmar_ratio()` in `src/metrics.py`).

- [ ] **Step 3.4: Position Sizing & Risk Management**
  - [ ] Volatility-scaled position sizing (size based on ATR or rolling std).
  - [ ] ATR-based trailing stops (adapt to volatility instead of fixed %).
  - [ ] Breakeven stop option (move SL to entry after configurable profit threshold).
  - [ ] Max portfolio risk limit (cap total exposure as % of equity).

- [ ] **Step 3.5: Robustness Validation**
  - [ ] Parameter sensitivity analysis — vary each param ±10-20%, measure metric stability.
  - [ ] Support configurable timeframes (intraday: 1h, 15m, 5m) with correct annualization.
  - [ ] Integrate walk-forward as a standard part of backtest runs (not standalone only).
  - [ ] Expand test coverage: end-of-backtest edge cases, overlapping signals, state leakage, real yfinance data.

---

### Phase 4: Building the Predictive Model

Build a machine learning model to forecast price movements. The backtesting engine must be hardened (Phase 3) before this phase so results can be trusted.

- [ ] **Step 4.1: Feature Engineering**
  - [ ] Technical indicators: RSI, MACD, Bollinger Bands (extend `src/indicators.py`).
  - [ ] Lagged price features and returns.
  - [ ] Volatility measures (realized vol, ATR regimes).
  - [ ] Market regime labels (trending vs. mean-reverting vs. choppy).
  - [ ] Volume profile features.

- [ ] **Step 4.2: Choose and Train a Model**
  - [ ] Start with XGBoost/LightGBM for tabular features (fast iteration, interpretable).
  - [ ] Experiment with LSTM or Transformer for sequence modeling if tabular underperforms.
  - [ ] Use walk-forward splits (not random train/test) to respect time-series ordering.
  - [ ] Target: predict direction (up/down/flat) or risk-adjusted return for next N bars.

- [ ] **Step 4.3: Evaluate Model Performance**
  - [ ] Classification metrics: accuracy, precision, recall, F1 on directional prediction.
  - [ ] Regression metrics: MSE, MAE if predicting returns.
  - [ ] Out-of-sample testing on multiple time periods (not just one held-out set).
  - [ ] Compare model signal quality vs. existing rule-based strategies.

---

### Phase 5: Integrated Backtesting with Predictive Model

Combine the predictive model with the backtesting engine to see if ML forecasts improve strategy performance.

- [ ] **Step 5.1: Create an ML Strategy Class**
  - [ ] Implement `MLStrategy(BaseStrategy)` that wraps the trained model.
  - [ ] Feed historical features to model, generate prediction for next bar.
  - [ ] Map model output to trading signals (`'buy'`/`'sell'`/`'short'`/`'hold'`).

- [ ] **Step 5.2: Hybrid Strategies**
  - [ ] Combine ML predictions with rule-based filters (e.g., "ML says buy AND ADX > 25").
  - [ ] Use ML confidence scores for position sizing (higher confidence → larger position).
  - [ ] Test ensemble: ML + existing strategies vote on signals.

- [ ] **Step 5.3: Run and Analyze the Predictive Backtest**
  - [ ] Compare ML strategy vs. rule-based strategies using same metrics.
  - [ ] Run walk-forward validation (train model on expanding window, test on next period).
  - [ ] MCPT on ML strategy to verify edge isn't random.
  - [ ] Check for look-ahead bias (model must not see future data during backtest).

---

### Phase 6: Paper Trading (Forward Testing)

Before risking real money, test the bot in a live, simulated environment.

- [ ] **Step 6.1: Connect to a Broker/Exchange API**
  - [ ] Choose an exchange with a paper trading sandbox (e.g., Alpaca, Binance testnet).
  - [ ] Replace `src/exchange.py` placeholder with real API calls (consider `ccxt` for crypto).
  - [ ] Add slippage and commission to live exchange to match backtest assumptions.

- [ ] **Step 6.2: Adapt Bot for Live Data**
  - [ ] Handle live data stream (websockets or polling).
  - [ ] Ensure model generates predictions in real-time within latency budget.
  - [ ] Add reconnection logic, error handling, and state persistence.

- [ ] **Step 6.3: Run and Monitor**
  - [ ] Run in paper trading for an extended period (weeks to months).
  - [ ] Compare live performance to backtest results — track slippage, fill rate, latency.
  - [ ] Build a monitoring dashboard (trade log, equity curve, drawdown alerts).

---

### Phase 7: Live Deployment (Use Extreme Caution)

This is the final step where the bot trades with real capital.

- [ ] **Step 7.1: Set Up a Secure Production Environment**
  - [ ] Deploy on a reliable server or cloud service (AWS, GCP, DigitalOcean) for 24/7 uptime.
  - [ ] Secure API keys (environment variables, secrets manager — never in code).
  - [ ] Set up health checks and auto-restart on failure.

- [ ] **Step 7.2: Implement Robust Risk Management**
  - [ ] **Position Sizing**: Never risk more than 1-2% of capital per trade.
  - [ ] **Stop-Loss**: Hard stop-loss on every position (already implemented, verify live).
  - [ ] **Kill Switch**: Manual and automatic shutdown (max daily loss, max drawdown trigger).
  - [ ] **Rate Limits**: Respect exchange API limits, queue orders if needed.

- [ ] **Step 7.3: Go Live with Minimal Capital**
  - [ ] Start with the smallest amount of capital you are willing to lose.
  - [ ] Monitor closely for the first weeks.
  - [ ] Scale up gradually only after consistent positive results.

---

### Phase 8: Continuous Improvement

A trading bot is never "finished." The market is always changing.

- [ ] **Step 8.1: Ongoing Monitoring and Logging**
  - [ ] Detailed trade logs (entry, exit, reason, duration, slippage).
  - [ ] Alerts for unexpected behavior (drawdown spikes, unusual trade frequency).
  - [ ] Periodic performance reports comparing live vs. backtest.

- [ ] **Step 8.2: Regular Model Retraining**
  - [ ] Retrain ML model periodically with new market data to prevent model drift.
  - [ ] A/B test retrained model vs. current model in paper trading before deploying.

- [ ] **Step 8.3: Strategy Expansion**
  - [ ] Add new strategies as market conditions change.
  - [ ] Portfolio-level backtesting with correlation-aware multi-asset allocation.
  - [ ] Regime detection — automatically switch strategies based on market conditions.
  - [ ] Explore alternative data sources (sentiment, on-chain metrics, order flow).