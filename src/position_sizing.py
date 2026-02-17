"""Position sizing models for the backtest engine.

All functions take current market state and return a trade_amount (float).
"""

import numpy as np
import pandas as pd


def fixed_size(trade_amount, **kwargs):
    """Return the fixed trade amount (default behavior, no-op wrapper)."""
    return trade_amount


def volatility_scaled_size(equity, risk_per_trade, atr, atr_multiplier=2.0):
    """Size position so that a stop-loss at atr_multiplier * ATR risks
    exactly risk_per_trade fraction of equity.

    Args:
        equity: Current portfolio equity.
        risk_per_trade: Fraction of equity to risk (e.g., 0.02 = 2%).
        atr: Current ATR value.
        atr_multiplier: How many ATRs define the stop distance.

    Returns:
        Trade amount (number of units).
    """
    if atr <= 0 or equity <= 0 or risk_per_trade <= 0:
        return 0.0
    dollar_risk = equity * risk_per_trade
    stop_distance = atr * atr_multiplier
    return dollar_risk / stop_distance


def rolling_std_size(equity, risk_per_trade, closes, window=20, multiplier=2.0):
    """Size based on rolling standard deviation of returns.

    Args:
        equity: Current portfolio equity.
        risk_per_trade: Fraction of equity to risk.
        closes: Recent close prices (at least window+1 values).
        window: Lookback period for std calculation.
        multiplier: How many stds define the stop distance.

    Returns:
        Trade amount.
    """
    if len(closes) < window + 1 or equity <= 0:
        return 0.0
    returns = np.diff(closes[-window - 1:]) / np.array(closes[-window - 1:-1])
    vol = float(np.std(returns))
    if vol <= 0:
        return 0.0
    price = closes[-1]
    stop_distance = price * vol * multiplier
    dollar_risk = equity * risk_per_trade
    return dollar_risk / stop_distance


def cap_by_max_risk(trade_amount, entry_price, equity, max_portfolio_risk,
                    existing_exposure=0.0):
    """Cap trade_amount so total exposure doesn't exceed max_portfolio_risk.

    Args:
        trade_amount: Proposed trade amount from sizing model.
        entry_price: Expected fill price.
        equity: Current portfolio equity.
        max_portfolio_risk: Max total exposure as fraction of equity.
        existing_exposure: Current dollar exposure from open positions.

    Returns:
        Capped trade amount.
    """
    if equity <= 0 or entry_price <= 0:
        return 0.0
    max_exposure = equity * max_portfolio_risk
    available = max_exposure - existing_exposure
    if available <= 0:
        return 0.0
    max_units = available / entry_price
    return min(trade_amount, max_units)
