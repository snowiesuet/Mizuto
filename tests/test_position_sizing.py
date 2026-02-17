"""Tests for position sizing models (Step 3.4)."""

import pytest
import numpy as np

from src.position_sizing import (
    fixed_size, volatility_scaled_size, rolling_std_size, cap_by_max_risk,
)


class TestFixedSize:
    def test_returns_trade_amount(self):
        assert fixed_size(1.0) == 1.0
        assert fixed_size(5.5) == 5.5

    def test_kwargs_ignored(self):
        assert fixed_size(2.0, extra=True) == 2.0


class TestVolatilityScaledSize:
    def test_basic_sizing(self):
        # equity=10000, risk=2%, ATR=50, multiplier=2 -> stop=100
        # dollar_risk = 200, units = 200/100 = 2.0
        result = volatility_scaled_size(10000, 0.02, 50.0, 2.0)
        assert result == pytest.approx(2.0)

    def test_zero_atr_returns_zero(self):
        assert volatility_scaled_size(10000, 0.02, 0.0) == 0.0

    def test_negative_atr_returns_zero(self):
        assert volatility_scaled_size(10000, 0.02, -5.0) == 0.0

    def test_zero_equity_returns_zero(self):
        assert volatility_scaled_size(0, 0.02, 50.0) == 0.0

    def test_zero_risk_returns_zero(self):
        assert volatility_scaled_size(10000, 0.0, 50.0) == 0.0

    def test_higher_risk_means_larger_size(self):
        small = volatility_scaled_size(10000, 0.01, 50.0)
        large = volatility_scaled_size(10000, 0.04, 50.0)
        assert large > small

    def test_higher_atr_means_smaller_size(self):
        large_atr = volatility_scaled_size(10000, 0.02, 100.0)
        small_atr = volatility_scaled_size(10000, 0.02, 25.0)
        assert small_atr > large_atr


class TestRollingStdSize:
    def test_basic_sizing(self):
        # Create prices with known non-zero volatility
        closes = [100 + i * 0.5 + (i % 3) * 0.2 for i in range(25)]
        result = rolling_std_size(10000, 0.02, closes, window=20)
        assert result > 0

    def test_insufficient_data_returns_zero(self):
        closes = [100.0] * 5
        assert rolling_std_size(10000, 0.02, closes, window=20) == 0.0

    def test_flat_prices_returns_zero(self):
        closes = [100.0] * 25
        assert rolling_std_size(10000, 0.02, closes, window=20) == 0.0

    def test_zero_equity_returns_zero(self):
        closes = [100 + i for i in range(25)]
        assert rolling_std_size(0, 0.02, closes, window=20) == 0.0


class TestCapByMaxRisk:
    def test_no_cap_when_within_limit(self):
        result = cap_by_max_risk(1.0, 100.0, 10000, 0.50, 0.0)
        assert result == 1.0  # 100 < 5000 limit

    def test_caps_when_exceeds_limit(self):
        # equity=10000, max_risk=10%, max_exposure=1000
        # proposed: 20 units @ 100 = 2000 exposure
        result = cap_by_max_risk(20.0, 100.0, 10000, 0.10, 0.0)
        assert result == pytest.approx(10.0)  # 1000/100

    def test_zero_available_returns_zero(self):
        result = cap_by_max_risk(5.0, 100.0, 10000, 0.10, 1000.0)
        assert result == 0.0  # already at max

    def test_negative_available_returns_zero(self):
        result = cap_by_max_risk(5.0, 100.0, 10000, 0.10, 1500.0)
        assert result == 0.0

    def test_zero_equity_returns_zero(self):
        assert cap_by_max_risk(5.0, 100.0, 0, 0.50, 0.0) == 0.0

    def test_zero_price_returns_zero(self):
        assert cap_by_max_risk(5.0, 0, 10000, 0.50, 0.0) == 0.0
