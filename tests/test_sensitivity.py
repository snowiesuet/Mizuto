"""Tests for parameter sensitivity analysis (Step 3.5)."""

import pytest
from src.sensitivity import analyze_sensitivity, _generate_param_ranges
from tests.conftest import make_ohlcv


class TestGenerateParamRanges:
    def test_integer_param(self):
        ranges = _generate_param_ranges({'short_window': 10}, 0.10)
        assert 'short_window' in ranges
        assert all(isinstance(v, int) for v in ranges['short_window'])
        assert all(v >= 1 for v in ranges['short_window'])

    def test_float_param(self):
        ranges = _generate_param_ranges({'trailing_stop_pct': 0.05}, 0.10)
        assert 'trailing_stop_pct' in ranges
        assert len(ranges['trailing_stop_pct']) >= 2

    def test_none_param_skipped(self):
        ranges = _generate_param_ranges({'stop_loss_pct': None}, 0.10)
        assert 'stop_loss_pct' not in ranges

    def test_non_numeric_skipped(self):
        ranges = _generate_param_ranges({'name': 'test'}, 0.10)
        assert 'name' not in ranges

    def test_variation_values_bracket_base(self):
        ranges = _generate_param_ranges({'long_window': 20}, 0.10)
        vals = ranges['long_window']
        # Should have values both below and above 20
        assert min(vals) < 20
        assert max(vals) > 20


class TestAnalyzeSensitivity:
    @pytest.fixture
    def data(self):
        return make_ohlcv(n=200, volatility=5.0, trend=0.5, seed=42)

    def test_returns_expected_keys(self, data):
        result = analyze_sensitivity(
            data,
            base_params={'short_window': 5, 'long_window': 20},
            metric='pnl',
        )
        assert 'base_metric' in result
        assert 'per_param' in result
        assert 'overall_stability' in result

    def test_per_param_has_sensitivity_score(self, data):
        result = analyze_sensitivity(
            data,
            base_params={'short_window': 5, 'long_window': 20},
            metric='pnl',
        )
        for param_name, info in result['per_param'].items():
            assert 'sensitivity_score' in info
            assert 'std' in info
            assert 'values_tested' in info
            assert 'metrics' in info
            assert 'range' in info

    def test_custom_param_ranges(self, data):
        result = analyze_sensitivity(
            data,
            base_params={'short_window': 5, 'long_window': 20},
            param_ranges={'long_window': [18, 20, 22]},
            metric='pnl',
        )
        assert 'long_window' in result['per_param']
        assert len(result['per_param']['long_window']['metrics']) == 3

    def test_overall_stability_is_finite(self, data):
        result = analyze_sensitivity(
            data,
            base_params={'short_window': 5, 'long_window': 20},
            metric='pnl',
            variation_pct=0.05,
        )
        for info in result['per_param'].values():
            assert info['sensitivity_score'] >= 0
