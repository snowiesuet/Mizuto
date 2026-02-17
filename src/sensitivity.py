"""Parameter sensitivity analysis for strategy robustness testing.

Varies each parameter +-10-20% and measures how stable the chosen metric
remains.  High sensitivity scores indicate fragile parameters that may be
overfit.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from src.backtest import run_backtest_on_data


def analyze_sensitivity(
    data,
    base_params: Dict,
    param_ranges: Optional[Dict[str, list]] = None,
    variation_pct: float = 0.10,
    metric: str = 'sharpe_ratio',
    strategy_factory=None,
    **backtest_kwargs,
) -> Dict:
    """Vary each parameter and measure metric stability.

    Args:
        data: DataFrame for backtesting.
        base_params: Dict of base parameter values, e.g.:
            {'short_window': 5, 'long_window': 20,
             'trailing_stop_pct': 0.05, 'stop_loss_pct': 0.10}
        param_ranges: Optional explicit {param_name: [val1, val2, ...]}
            overrides.  If None, each numeric param is varied by
            +-variation_pct and +-2*variation_pct.
        variation_pct: Fraction to vary each parameter (default 10%).
        metric: Metric to track stability for.
        strategy_factory: Optional callable(params_dict) -> BaseStrategy.
        **backtest_kwargs: Additional kwargs passed to run_backtest_on_data.

    Returns:
        Dict with:
          'base_metric': metric at base params
          'per_param': {param_name: {
              'values_tested': [...],
              'metrics': [...],
              'std': float,
              'range': float,
              'sensitivity_score': float  # normalized std/|base|
          }}
          'overall_stability': float  # mean of all sensitivity_scores
    """
    base_result = _run_with_params(data, base_params, strategy_factory,
                                   **backtest_kwargs)
    base_metric_val = _extract_metric(base_result, metric)

    if param_ranges is None:
        param_ranges = _generate_param_ranges(base_params, variation_pct)

    per_param = {}
    for param_name, values in param_ranges.items():
        metrics_list = []
        for val in values:
            test_params = dict(base_params)
            test_params[param_name] = val
            # Skip invalid combos (short >= long)
            sw = test_params.get('short_window', 0)
            lw = test_params.get('long_window', 999)
            if sw >= lw:
                metrics_list.append(None)
                continue
            result = _run_with_params(data, test_params, strategy_factory,
                                      **backtest_kwargs)
            metrics_list.append(_extract_metric(result, metric))

        valid = [m for m in metrics_list if m is not None]
        if valid:
            std = float(np.std(valid))
            metric_range = max(valid) - min(valid)
            sensitivity = (std / abs(base_metric_val)
                           if base_metric_val != 0 else float('inf'))
        else:
            std = 0.0
            metric_range = 0.0
            sensitivity = 0.0

        per_param[param_name] = {
            'values_tested': values,
            'metrics': metrics_list,
            'std': std,
            'range': metric_range,
            'sensitivity_score': sensitivity,
        }

    scores = [v['sensitivity_score'] for v in per_param.values()
              if v['sensitivity_score'] != float('inf')]
    overall = float(np.mean(scores)) if scores else float('inf')

    return {
        'base_metric': base_metric_val,
        'per_param': per_param,
        'overall_stability': overall,
    }


def _generate_param_ranges(base_params, variation_pct):
    """Generate +-10% and +-20% variations for each numeric parameter."""
    ranges = {}
    for name, val in base_params.items():
        if val is None:
            continue
        if not isinstance(val, (int, float)):
            continue
        variations = []
        for mult in [-2, -1, 1, 2]:
            new_val = val * (1 + mult * variation_pct)
            if isinstance(val, int):
                new_val = max(1, round(new_val))
            elif new_val < 0:
                continue
            variations.append(new_val)
        ranges[name] = sorted(set(variations))
    return ranges


def _run_with_params(data, params, strategy_factory, **kwargs):
    """Run a backtest with the given parameter dict."""
    backtest_params = dict(kwargs)
    backtest_params['quiet'] = True

    for key in ('short_window', 'long_window', 'trailing_stop_pct',
                'stop_loss_pct', 'slippage_pct', 'commission_pct'):
        if key in params:
            backtest_params[key] = params[key]

    if strategy_factory is not None:
        backtest_params['strategy'] = strategy_factory(params)

    return run_backtest_on_data(data, **backtest_params)


def _extract_metric(result, metric):
    """Safely extract a metric from a backtest result."""
    if result is None:
        return 0.0
    val = result.get(metric, 0.0)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    return float(val)
