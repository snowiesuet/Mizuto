"""Classic floor pivot-point bounce strategy for the src/ backtest engine.

Ported from ``bt/strategies/pivot_points.py`` with short-selling support.
"""

import logging
import math

import pandas as pd

from src.strategies.base import BaseStrategy
from src.indicators import compute_pivot_points


class PivotPointStrategy(BaseStrategy):
    """Trade bounces off classic floor pivot levels.

    Long on S1 bounce (price crosses back above S1), stop at S2, TP at PP.
    Short on R1 rejection (price crosses back below R1), stop at R2, TP at PP.

    Set *use_s2_r2* to ``True`` for second-level entries (S2/R2 bounces with
    stops at S3/R3 and TP at S1/R1).
    """

    name = "Pivot Points"

    def __init__(self, use_s2_r2: bool = False):
        self.use_s2_r2 = use_s2_r2

        # Previous bar's HLC for pivot calculation
        self._prev_high: float | None = None
        self._prev_low: float | None = None
        self._prev_close: float | None = None

        # Previous bar close for crossover detection
        self._prev_bar_close: float | None = None

        # Internal position tracking for SL/TP
        self._sl_price: float | None = None
        self._tp_price: float | None = None
        self._position_type: str | None = None

    @property
    def requires_ohlcv(self) -> bool:
        return True

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before the strategy can produce signals."""
        return 3

    # --- BaseStrategy interface ---------------------------------------------------

    def on_price(self, price: float, has_position: bool) -> str:
        """Fallback for close-only data — not recommended for this strategy."""
        bar = {'Open': price, 'High': price, 'Low': price, 'Close': price, 'Volume': 0}
        return self.on_bar(bar, has_position)

    def on_bar(self, bar: dict, has_position: bool, position_type: str = None) -> str:
        price = float(bar['Close'])
        high = float(bar['High'])
        low = float(bar['Low'])

        # Need at least one previous bar to compute pivots
        if self._prev_high is None or self._prev_bar_close is None:
            self._prev_high = high
            self._prev_low = low
            self._prev_close = price
            self._prev_bar_close = price
            return 'hold'

        # Compute pivot levels from previous bar
        pivots = compute_pivot_points(self._prev_high, self._prev_low, self._prev_close)
        pp = pivots['PP']
        s1, s2, s3 = pivots['S1'], pivots['S2'], pivots['S3']
        r1, r2, r3 = pivots['R1'], pivots['R2'], pivots['R3']

        prev_close = self._prev_bar_close

        # Update previous bar data for next iteration
        self._prev_high = high
        self._prev_low = low
        self._prev_close = price
        self._prev_bar_close = price

        # --- Exit logic (strategy-managed SL/TP) ---
        if has_position and self._position_type is not None:
            if self._position_type == 'long':
                if self._sl_price is not None and price <= self._sl_price:
                    logging.info(f"Pivot Points: long SL hit at {price:.2f}")
                    self._clear_internal_position()
                    return 'sell'
                if self._tp_price is not None and price >= self._tp_price:
                    logging.info(f"Pivot Points: long TP hit at {price:.2f}")
                    self._clear_internal_position()
                    return 'sell'
            elif self._position_type == 'short':
                if self._sl_price is not None and price >= self._sl_price:
                    logging.info(f"Pivot Points: short SL hit at {price:.2f}")
                    self._clear_internal_position()
                    return 'sell'
                if self._tp_price is not None and price <= self._tp_price:
                    logging.info(f"Pivot Points: short TP hit at {price:.2f}")
                    self._clear_internal_position()
                    return 'sell'
            return 'hold'

        if has_position:
            return 'hold'

        # --- Entry logic ---
        # S1 bounce: previous close below S1, current crosses back above
        if prev_close < s1 and price >= s1:
            self._sl_price = s2
            self._tp_price = pp
            self._position_type = 'long'
            logging.info(f"Pivot Points: LONG entry (S1 bounce) at {price:.2f}, SL={s2:.2f}, TP={pp:.2f}")
            return 'buy'

        # R1 rejection: previous close above R1, current crosses back below
        if prev_close > r1 and price <= r1:
            self._sl_price = r2
            self._tp_price = pp
            self._position_type = 'short'
            logging.info(f"Pivot Points: SHORT entry (R1 rejection) at {price:.2f}, SL={r2:.2f}, TP={pp:.2f}")
            return 'short'

        # Optional second-level entries
        if self.use_s2_r2:
            if prev_close < s2 and price >= s2:
                self._sl_price = s3
                self._tp_price = s1
                self._position_type = 'long'
                logging.info(f"Pivot Points: LONG entry (S2 bounce) at {price:.2f}, SL={s3:.2f}, TP={s1:.2f}")
                return 'buy'

            if prev_close > r2 and price <= r2:
                self._sl_price = r3
                self._tp_price = r1
                self._position_type = 'short'
                logging.info(f"Pivot Points: SHORT entry (R2 rejection) at {price:.2f}, SL={r3:.2f}, TP={r1:.2f}")
                return 'short'

        return 'hold'

    def reset(self) -> None:
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._prev_bar_close = None
        self._sl_price = None
        self._tp_price = None
        self._position_type = None

    def load_ohlcv_history(self, df: pd.DataFrame) -> None:
        if len(df) >= 1:
            last = df.iloc[-1]
            self._prev_high = float(last['High'])
            self._prev_low = float(last['Low'])
            self._prev_close = float(last['Close'])
            self._prev_bar_close = float(last['Close'])
        logging.info(f"Pivot Points loaded history ({len(df)} bars, using last bar for pivots).")

    # --- Internal helpers --------------------------------------------------------

    def _clear_internal_position(self):
        """Reset strategy-internal position tracking (called on exit signals)."""
        self._sl_price = None
        self._tp_price = None
        self._position_type = None
