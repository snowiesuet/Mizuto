"""Unit tests for bt/helpers.py indicator functions."""

import numpy as np
import pytest

from bt.helpers import sma, atr, adx, rolling_max, rolling_min, pivot_points
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


class TestSMA:
    def test_known_values(self):
        result = sma([10.0, 20.0, 30.0, 40.0, 50.0], window=3)
        assert len(result) == 5
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(20.0)
        assert result[3] == pytest.approx(30.0)
        assert result[4] == pytest.approx(40.0)

    def test_output_length_matches_input(self):
        data = list(range(1, 51))
        result = sma(data, window=10)
        assert len(result) == len(data)

    def test_window_larger_than_data(self):
        """Window > data length → all NaN, no crash."""
        result = sma([1.0, 2.0, 3.0], window=10)
        assert len(result) == 3
        assert all(np.isnan(result))


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


class TestATR:
    def test_positive_values_after_warmup(self):
        df = make_ohlcv(50)
        result = atr(df["High"], df["Low"], df["Close"], length=14)
        assert len(result) == 50
        # After warmup period (ta library uses 0.0 for warmup, not NaN),
        # later values should be positive
        post_warmup = result[14:]
        assert (post_warmup > 0).all()

    def test_zero_during_warmup(self):
        """ta library fills warmup period with 0.0 (not NaN)."""
        df = make_ohlcv(30)
        result = atr(df["High"], df["Low"], df["Close"], length=14)
        assert result[0] == 0.0


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


class TestADX:
    def test_returns_three_arrays(self):
        df = make_ohlcv(60)
        adx_val, plus_di, minus_di = adx(df["High"], df["Low"], df["Close"], length=14)
        assert len(adx_val) == 60
        assert len(plus_di) == 60
        assert len(minus_di) == 60

    def test_values_in_range(self):
        """ADX and DI values should be between 0 and 100 where not NaN."""
        df = make_ohlcv(100)
        adx_val, plus_di, minus_di = adx(df["High"], df["Low"], df["Close"], length=14)
        for arr in (adx_val, plus_di, minus_di):
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                assert (valid >= 0).all()
                assert (valid <= 100).all()


# ---------------------------------------------------------------------------
# Rolling max / min
# ---------------------------------------------------------------------------


class TestRollingMaxMin:
    def test_known_values(self):
        data = [1.0, 3.0, 2.0, 5.0, 4.0]
        rmax = rolling_max(data, window=3)
        rmin = rolling_min(data, window=3)
        assert len(rmax) == 5
        assert len(rmin) == 5
        # First 2 are NaN (window=3)
        assert np.isnan(rmax[0])
        assert np.isnan(rmax[1])
        # rolling_max([1,3,2])=3, ([3,2,5])=5, ([2,5,4])=5
        assert rmax[2] == pytest.approx(3.0)
        assert rmax[3] == pytest.approx(5.0)
        assert rmax[4] == pytest.approx(5.0)
        # rolling_min([1,3,2])=1, ([3,2,5])=2, ([2,5,4])=2
        assert rmin[2] == pytest.approx(1.0)
        assert rmin[3] == pytest.approx(2.0)
        assert rmin[4] == pytest.approx(2.0)

    def test_window_one_returns_original(self):
        data = [5.0, 3.0, 8.0, 1.0]
        rmax = rolling_max(data, window=1)
        rmin = rolling_min(data, window=1)
        np.testing.assert_array_almost_equal(rmax, data)
        np.testing.assert_array_almost_equal(rmin, data)


# ---------------------------------------------------------------------------
# Pivot points
# ---------------------------------------------------------------------------


class TestPivotPoints:
    def test_known_values(self):
        """Verify floor pivot formula: PP = (prevH + prevL + prevC) / 3."""
        high = [110.0, 120.0]
        low = [90.0, 95.0]
        close = [100.0, 105.0]
        result = pivot_points(high, low, close)

        # First element should be NaN (no previous bar)
        assert np.isnan(result["PP"][0])
        assert np.isnan(result["S1"][0])

        # Second element uses bar 0's HLC: PP = (110+90+100)/3 = 100
        pp = 100.0
        assert result["PP"][1] == pytest.approx(pp)
        assert result["S1"][1] == pytest.approx(2 * pp - 110)  # 90
        assert result["R1"][1] == pytest.approx(2 * pp - 90)   # 110
        assert result["S2"][1] == pytest.approx(pp - (110 - 90))  # 80
        assert result["R2"][1] == pytest.approx(pp + (110 - 90))  # 120

    def test_all_keys_present(self):
        result = pivot_points([100, 110], [90, 85], [95, 100])
        expected_keys = {"PP", "S1", "S2", "S3", "R1", "R2", "R3"}
        assert set(result.keys()) == expected_keys

    def test_first_bar_is_nan(self):
        result = pivot_points([100], [90], [95])
        assert np.isnan(result["PP"][0])
