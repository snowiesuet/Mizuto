"""
Risk-adjusted performance metrics for backtesting.

Pure functions operating on equity curves (lists of portfolio values).
"""

import math
import numpy as np
import pandas as pd


def compute_max_drawdown(equity_curve):
    """Compute maximum drawdown percentage and duration.

    Args:
        equity_curve: List of portfolio values over time.

    Returns:
        Dict with 'max_drawdown_pct' (float, negative or zero),
                   'max_drawdown_duration' (int, bars),
                   'drawdown_series' (list of float, per-bar drawdown %).
    """
    if len(equity_curve) < 2:
        return {
            'max_drawdown_pct': 0.0,
            'max_drawdown_duration': 0,
            'drawdown_series': [0.0] * len(equity_curve),
        }

    equity = np.array(equity_curve, dtype=float)
    running_peak = np.maximum.accumulate(equity)
    drawdown = (equity - running_peak) / running_peak

    max_dd_pct = float(np.min(drawdown))

    # Compute max drawdown duration (bars from peak to recovery)
    max_duration = 0
    current_duration = 0
    for dd in drawdown:
        if dd < 0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return {
        'max_drawdown_pct': max_dd_pct,
        'max_drawdown_duration': max_duration,
        'drawdown_series': drawdown.tolist(),
    }


def compute_sharpe_ratio(equity_curve, periods_per_year=252, risk_free_rate=0.0):
    """Annualized Sharpe ratio from equity curve.

    Args:
        equity_curve: List of portfolio values.
        periods_per_year: Trading periods per year (252 for daily).
        risk_free_rate: Annual risk-free rate (default 0).

    Returns:
        Float Sharpe ratio, or 0.0 if insufficient data or zero volatility.
    """
    if len(equity_curve) < 2:
        return 0.0

    equity = np.array(equity_curve, dtype=float)
    returns = np.diff(equity) / equity[:-1]

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period

    return float(np.mean(excess_returns) / np.std(returns) * math.sqrt(periods_per_year))


def compute_sortino_ratio(equity_curve, periods_per_year=252, risk_free_rate=0.0):
    """Annualized Sortino ratio (penalizes only downside volatility).

    Args:
        equity_curve: List of portfolio values.
        periods_per_year: Trading periods per year (252 for daily).
        risk_free_rate: Annual risk-free rate (default 0).

    Returns:
        Float Sortino ratio. Returns float('inf') if no downside deviation,
        0.0 if insufficient data.
    """
    if len(equity_curve) < 2:
        return 0.0

    equity = np.array(equity_curve, dtype=float)
    returns = np.diff(equity) / equity[:-1]

    if len(returns) == 0:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_per_period

    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        mean_excess = np.mean(excess_returns)
        return float('inf') if mean_excess > 0 else 0.0

    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0

    return float(np.mean(excess_returns) / downside_std * math.sqrt(periods_per_year))


def compute_cagr(equity_curve, dates, periods_per_year=252):
    """Compound Annual Growth Rate.

    Args:
        equity_curve: List of portfolio values.
        dates: List of datetime-like objects corresponding to equity_curve.
        periods_per_year: Trading periods per year (for fallback calculation).

    Returns:
        Float CAGR as a decimal (e.g., 0.15 = 15%).
    """
    if len(equity_curve) < 2 or equity_curve[0] <= 0:
        return 0.0

    start_val = equity_curve[0]
    end_val = equity_curve[-1]

    # Try to use actual dates for year calculation
    try:
        start_date = pd.Timestamp(dates[0])
        end_date = pd.Timestamp(dates[-1])
        days = (end_date - start_date).days
        if days <= 0:
            return 0.0
        years = days / 365.25
    except (TypeError, ValueError):
        n_periods = len(equity_curve) - 1
        years = n_periods / periods_per_year

    if years <= 0:
        return 0.0

    return float((end_val / start_val) ** (1 / years) - 1)


def compute_annualized_return(equity_curve, periods_per_year=252):
    """Simple annualized return from equity curve.

    Args:
        equity_curve: List of portfolio values.
        periods_per_year: Trading periods per year.

    Returns:
        Float annualized return as decimal.
    """
    if len(equity_curve) < 2 or equity_curve[0] <= 0:
        return 0.0

    total_return = equity_curve[-1] / equity_curve[0] - 1
    n_periods = len(equity_curve) - 1
    years = n_periods / periods_per_year

    if years <= 0:
        return 0.0

    return float((1 + total_return) ** (1 / years) - 1)


def compute_calmar_ratio(equity_curve, dates, periods_per_year=252):
    """Calmar ratio = CAGR / |max drawdown|.

    Args:
        equity_curve: List of portfolio values.
        dates: List of datetime-like objects corresponding to equity_curve.
        periods_per_year: Trading periods per year.

    Returns:
        Float Calmar ratio. Returns ``float('inf')`` if no drawdown with
        positive CAGR, 0.0 if insufficient data.
    """
    cagr = compute_cagr(equity_curve, dates, periods_per_year)
    dd = compute_max_drawdown(equity_curve)
    max_dd = abs(dd['max_drawdown_pct'])
    if max_dd == 0:
        return float('inf') if cagr > 0 else 0.0
    return cagr / max_dd


def compute_buy_and_hold_return(data, initial_capital):
    """Buy-and-hold return for comparison.

    Args:
        data: DataFrame with 'Close' column (the simulation period data).
        initial_capital: Starting capital.

    Returns:
        Dict with 'bh_return_pct' (float), 'bh_equity_final' (float),
                   'bh_equity_curve' (list of float).
    """
    close = data['Close'].values.astype(float)

    if len(close) < 2 or close[0] <= 0:
        return {
            'bh_return_pct': 0.0,
            'bh_equity_final': initial_capital,
            'bh_equity_curve': [initial_capital] * len(close),
        }

    bh_equity = (initial_capital * (close / close[0])).tolist()
    bh_return_pct = (close[-1] / close[0] - 1) * 100

    return {
        'bh_return_pct': float(bh_return_pct),
        'bh_equity_final': float(bh_equity[-1]),
        'bh_equity_curve': bh_equity,
    }


def compute_all_metrics(equity_curve, equity_dates, data, initial_capital,
                         periods_per_year=252):
    """Compute all risk-adjusted metrics in one call.

    Args:
        equity_curve: List of portfolio values at each bar.
        equity_dates: List of dates corresponding to equity_curve.
        data: DataFrame with 'Close' column (simulation period).
        initial_capital: Starting capital.
        periods_per_year: Trading periods per year.

    Returns:
        Dict with all computed metrics merged together.
    """
    result = {}

    dd = compute_max_drawdown(equity_curve)
    result['max_drawdown_pct'] = dd['max_drawdown_pct']
    result['max_drawdown_duration'] = dd['max_drawdown_duration']
    result['drawdown_series'] = dd['drawdown_series']

    result['sharpe_ratio'] = compute_sharpe_ratio(
        equity_curve, periods_per_year)
    result['sortino_ratio'] = compute_sortino_ratio(
        equity_curve, periods_per_year)
    result['cagr'] = compute_cagr(
        equity_curve, equity_dates, periods_per_year)
    result['annualized_return'] = compute_annualized_return(
        equity_curve, periods_per_year)
    result['calmar_ratio'] = compute_calmar_ratio(
        equity_curve, equity_dates, periods_per_year)

    bh = compute_buy_and_hold_return(data, initial_capital)
    result.update(bh)

    return result
