"""Cross-engine validation: compare src/ and bt/ on the same data."""

import pytest

from src.backtest import run_backtest_on_data
from bt.runner import run_bt
from tests.conftest import make_ohlcv


class TestCrossEngineValidation:
    """Compare src/ and bt/ engines on the same data with MACrossoverStrategy."""

    @pytest.fixture
    def shared_data(self):
        """Synthetic data both engines will use.
        Use strong trend + low volatility for clearer crossover signals.
        """
        return make_ohlcv(n=300, base=100.0, volatility=3.0, trend=0.5, seed=42)

    def _run_both_engines(self, data, short_window=5, long_window=20):
        src_result = run_backtest_on_data(
            data, short_window=short_window, long_window=long_window,
            slippage_pct=0.0, commission_pct=0.0, quiet=True,
        )

        bt_stats = run_bt(
            "ma_crossover", data=data,
            cash=10_000, commission=0.0,
            short_window=short_window, long_window=long_window,
        )

        return src_result, bt_stats

    def test_both_engines_run_successfully(self, shared_data):
        src_result, bt_stats = self._run_both_engines(shared_data)
        # Both should complete without error
        assert bt_stats is not None
        # src_result may be None if no trades, but bt should have stats

    def test_trade_count_within_tolerance(self, shared_data):
        """Both engines should produce similar trade counts."""
        src_result, bt_stats = self._run_both_engines(shared_data)

        if src_result is None:
            pytest.skip("src engine produced no trades")

        src_trades = src_result['trade_count']
        bt_trades = bt_stats['# Trades']

        # Allow tolerance: within 50% or absolute diff <= 3
        if max(src_trades, bt_trades) > 0:
            ratio = min(src_trades, bt_trades) / max(src_trades, bt_trades)
            assert ratio > 0.5 or abs(src_trades - bt_trades) <= 3, \
                f"Trade count mismatch: src={src_trades}, bt={bt_trades}"

    def test_return_direction_agrees(self, shared_data):
        """Both engines should agree on whether strategy is profitable."""
        src_result, bt_stats = self._run_both_engines(shared_data)

        if src_result is None or src_result['trade_count'] == 0:
            pytest.skip("src engine produced no trades")

        src_profitable = src_result['pnl'] > 0
        bt_profitable = bt_stats['Return [%]'] > 0
        # Soft check: log disagreement but don't fail on edge cases
        if src_result['trade_count'] > 3:
            assert src_profitable == bt_profitable, \
                f"Direction mismatch: src pnl={src_result['pnl']:.2f}, bt return={bt_stats['Return [%]']:.2f}%"

    def test_win_rate_within_tolerance(self, shared_data):
        """Win rates should be approximately similar."""
        src_result, bt_stats = self._run_both_engines(shared_data)

        if src_result is None or src_result['trade_count'] < 3:
            pytest.skip("src engine produced too few trades")

        src_win_rate = src_result['wins'] / src_result['trade_count']
        bt_win_rate = bt_stats['Win Rate [%]'] / 100.0

        # 25 percentage point tolerance
        assert abs(src_win_rate - bt_win_rate) < 0.25, \
            f"Win rate mismatch: src={src_win_rate:.2%}, bt={bt_win_rate:.2%}"
