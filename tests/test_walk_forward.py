"""Tests for walk-forward optimization in src.optimize."""

import pytest

from src.optimize import walk_forward_optimize, rolling_walk_forward
from tests.conftest import make_ohlcv


class TestWalkForwardOptimize:
    @pytest.fixture
    def data(self):
        return make_ohlcv(n=300, volatility=5.0, trend=0.5, seed=42)

    def test_returns_expected_keys(self, data):
        result = walk_forward_optimize(
            data, train_ratio=0.7,
            short_window_range=range(3, 10),
            long_window_range=range(15, 30, 5),
        )
        expected_keys = {
            'train_params', 'train_metric', 'test_result',
            'test_metric', 'train_dates', 'test_dates', 'overfit_ratio',
        }
        assert expected_keys == set(result.keys())

    def test_train_test_split_dates(self, data):
        result = walk_forward_optimize(
            data, train_ratio=0.7,
            short_window_range=range(3, 10),
            long_window_range=range(15, 30, 5),
        )
        train_start, train_end = result['train_dates']
        test_start, test_end = result['test_dates']
        # Train should come before test
        assert train_end < test_start or train_end == test_start

    def test_short_less_than_long(self, data):
        result = walk_forward_optimize(
            data, train_ratio=0.7,
            short_window_range=range(3, 10),
            long_window_range=range(15, 30, 5),
        )
        params = result['train_params']
        assert params['short_window'] < params['long_window']

    def test_overfit_ratio_computed(self, data):
        result = walk_forward_optimize(
            data, train_ratio=0.7,
            short_window_range=range(3, 10),
            long_window_range=range(15, 30, 5),
        )
        assert isinstance(result['overfit_ratio'], float)

    def test_train_params_is_dict(self, data):
        result = walk_forward_optimize(
            data, train_ratio=0.7,
            short_window_range=range(3, 10),
            long_window_range=range(15, 30, 5),
        )
        params = result['train_params']
        assert 'short_window' in params
        assert 'long_window' in params

    def test_empty_split_raises(self):
        # With only 1 row, any ratio produces an empty train or test set
        data = make_ohlcv(n=1)
        with pytest.raises(ValueError):
            walk_forward_optimize(
                data, train_ratio=0.5,
                short_window_range=range(3, 5),
                long_window_range=range(10, 20, 5),
            )


class TestRollingWalkForward:
    @pytest.fixture
    def data(self):
        return make_ohlcv(n=500, volatility=5.0, trend=0.3, seed=123)

    def test_returns_n_windows(self, data):
        result = rolling_walk_forward(
            data, n_windows=3, train_ratio=0.6,
            short_window_range=range(3, 8),
            long_window_range=range(15, 25, 5),
        )
        assert len(result['windows']) == 3

    def test_aggregate_metrics_are_means(self, data):
        result = rolling_walk_forward(
            data, n_windows=3, train_ratio=0.6,
            short_window_range=range(3, 8),
            long_window_range=range(15, 25, 5),
        )
        test_metrics = [w['test_metric'] for w in result['windows']]
        expected_mean = sum(test_metrics) / len(test_metrics)
        assert result['aggregate_test_metric'] == pytest.approx(expected_mean)

    def test_returns_expected_keys(self, data):
        result = rolling_walk_forward(
            data, n_windows=2, train_ratio=0.6,
            short_window_range=range(3, 8),
            long_window_range=range(15, 25, 5),
        )
        expected = {'windows', 'aggregate_test_metric',
                    'aggregate_train_metric', 'aggregate_overfit_ratio'}
        assert expected == set(result.keys())

    def test_too_many_windows_raises(self):
        data = make_ohlcv(n=30)
        with pytest.raises(ValueError):
            rolling_walk_forward(
                data, n_windows=100, train_ratio=0.7,
                short_window_range=range(3, 5),
                long_window_range=range(10, 20, 5),
            )
