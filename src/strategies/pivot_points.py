"""Classic floor pivot-point bounce strategy for the src/ backtest engine.

Ported from ``bt/strategies/pivot_points.py`` with short-selling support.
Supports timeframe-aware pivot calculation: intraday data uses daily pivots,
daily data uses weekly pivots, weekly data uses monthly pivots.
"""

import logging

import pandas as pd

from src.strategies.base import BaseStrategy
from src.indicators import compute_pivot_points


def _infer_pivot_timeframe(df: pd.DataFrame) -> str | None:
    """Infer the appropriate pivot timeframe from data frequency.

    Returns ``'D'``, ``'W'``, ``'M'``, or ``None`` if insufficient data.
    """
    if len(df) < 2:
        return None
    try:
        deltas = pd.Series(df.index).diff().dropna()
        median_delta = deltas.median()
    except Exception:
        return None
    if median_delta <= pd.Timedelta(hours=2):
        return 'D'   # intraday / hourly → daily pivots
    elif median_delta <= pd.Timedelta(days=2):
        return 'W'   # daily → weekly pivots
    else:
        return 'M'   # weekly → monthly pivots


def _period_key(ts, timeframe: str):
    """Extract the period grouping key from a timestamp."""
    if timeframe == 'D':
        return ts.date() if hasattr(ts, 'date') else ts
    elif timeframe == 'W':
        iso = ts.isocalendar()
        return (iso[0], iso[1])
    elif timeframe == 'M':
        return (ts.year, ts.month)
    return None


class PivotPointStrategy(BaseStrategy):
    """Trade bounces off classic floor pivot levels.

    Long on S1 bounce (price crosses back above S1), stop at S2, TP at PP.
    Short on R1 rejection (price crosses back below R1), stop at R2, TP at PP.

    Set *use_s2_r2* to ``True`` for second-level entries (S2/R2 bounces with
    stops at S3/R3 and TP at S1/R1).

    Parameters
    ----------
    pivot_timeframe : str
        ``'auto'`` (default) infers from data: intraday→daily, daily→weekly,
        weekly→monthly.  ``'D'``/``'W'``/``'M'`` for explicit override.
        ``None`` for legacy single-bar behavior.
    """

    name = "Pivot Points"

    def __init__(self, use_s2_r2: bool = False, pivot_timeframe: str = 'auto'):
        self.use_s2_r2 = use_s2_r2
        self.pivot_timeframe = pivot_timeframe

        # Resolved timeframe: explicit values resolve immediately,
        # 'auto' defers to load_ohlcv_history
        if pivot_timeframe in ('D', 'W', 'M'):
            self._resolved_timeframe: str | None = pivot_timeframe
        else:
            self._resolved_timeframe: str | None = None

        # Previous completed period's HLC (pivot source)
        self._prev_high: float | None = None
        self._prev_low: float | None = None
        self._prev_close: float | None = None

        # Current period accumulator (timeframe-aware mode)
        self._period_high: float | None = None
        self._period_low: float | None = None
        self._period_close: float | None = None
        self._current_period_key = None

        # Cached pivot levels (recomputed only on period boundary)
        self._cached_pivots: dict | None = None

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
        ts = bar.get('Timestamp')

        # --- Determine pivot levels ---
        pivots = self._update_pivots(high, low, price, ts)
        if pivots is None:
            # Not enough data yet
            self._prev_bar_close = price
            return 'hold'

        pp = pivots['PP']
        s1, s2, s3 = pivots['S1'], pivots['S2'], pivots['S3']
        r1, r2, r3 = pivots['R1'], pivots['R2'], pivots['R3']

        prev_close = self._prev_bar_close
        self._prev_bar_close = price

        if prev_close is None:
            return 'hold'

        # --- Exit logic (strategy-managed SL/TP) ---
        if has_position and self._position_type is not None:
            if self._position_type == 'long':
                if self._sl_price is not None and price <= self._sl_price:
                    logging.info(f"Pivot Points: long SL hit at {price:.2f}")
                    self._clear_internal_position()
                    return ('sell', 'sl_hit')
                if self._tp_price is not None and price >= self._tp_price:
                    logging.info(f"Pivot Points: long TP hit at {price:.2f}")
                    self._clear_internal_position()
                    return ('sell', 'tp_hit')
            elif self._position_type == 'short':
                if self._sl_price is not None and price >= self._sl_price:
                    logging.info(f"Pivot Points: short SL hit at {price:.2f}")
                    self._clear_internal_position()
                    return ('sell', 'sl_hit')
                if self._tp_price is not None and price <= self._tp_price:
                    logging.info(f"Pivot Points: short TP hit at {price:.2f}")
                    self._clear_internal_position()
                    return ('sell', 'tp_hit')
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
        self._period_high = None
        self._period_low = None
        self._period_close = None
        self._current_period_key = None
        self._cached_pivots = None
        self._sl_price = None
        self._tp_price = None
        self._position_type = None
        # Note: _resolved_timeframe is NOT reset — it's config derived from data

    def load_ohlcv_history(self, df: pd.DataFrame) -> None:
        # Resolve timeframe
        if self.pivot_timeframe == 'auto':
            self._resolved_timeframe = _infer_pivot_timeframe(df)
        elif self.pivot_timeframe is not None:
            self._resolved_timeframe = self.pivot_timeframe
        else:
            self._resolved_timeframe = None

        if self._resolved_timeframe is not None and len(df) >= 2:
            self._load_htf_warmup(df)
        elif len(df) >= 1:
            # Legacy single-bar mode
            last = df.iloc[-1]
            self._prev_high = float(last['High'])
            self._prev_low = float(last['Low'])
            self._prev_close = float(last['Close'])
            self._prev_bar_close = float(last['Close'])

        logging.info(
            f"Pivot Points loaded history ({len(df)} bars, "
            f"timeframe={self._resolved_timeframe})."
        )

    # --- Internal helpers --------------------------------------------------------

    def _update_pivots(self, high: float, low: float, price: float, ts) -> dict | None:
        """Update pivot levels and return current pivots, or None if not ready."""
        tf = self._resolved_timeframe

        if tf is not None and ts is not None:
            # --- Timeframe-aware mode ---
            key = _period_key(ts, tf)

            if self._current_period_key is None:
                # Very first bar: initialize accumulator
                self._current_period_key = key
                self._period_high = high
                self._period_low = low
                self._period_close = price
                return self._cached_pivots  # may be None or pre-loaded

            if key != self._current_period_key:
                # Period boundary: finalize completed period as pivot source
                self._prev_high = self._period_high
                self._prev_low = self._period_low
                self._prev_close = self._period_close
                self._cached_pivots = compute_pivot_points(
                    self._prev_high, self._prev_low, self._prev_close
                )
                # Start new period
                self._current_period_key = key
                self._period_high = high
                self._period_low = low
                self._period_close = price
            else:
                # Same period: update running HLC
                self._period_high = max(self._period_high, high)
                self._period_low = min(self._period_low, low)
                self._period_close = price

            return self._cached_pivots
        else:
            # --- Legacy single-bar mode ---
            if self._prev_high is None:
                self._prev_high = high
                self._prev_low = low
                self._prev_close = price
                return None

            pivots = compute_pivot_points(self._prev_high, self._prev_low, self._prev_close)
            self._prev_high = high
            self._prev_low = low
            self._prev_close = price
            return pivots

    def _load_htf_warmup(self, df: pd.DataFrame) -> None:
        """Pre-aggregate warmup data into higher-timeframe periods."""
        tf = self._resolved_timeframe
        keys = [_period_key(ts, tf) for ts in df.index]
        df_tmp = df.copy()
        df_tmp['_pk'] = keys
        unique_keys = list(dict.fromkeys(keys))  # preserves order

        if len(unique_keys) >= 2:
            # Use second-to-last complete period as pivot source
            prev_key = unique_keys[-2]
            prev_data = df_tmp[df_tmp['_pk'] == prev_key]
            self._prev_high = float(prev_data['High'].max())
            self._prev_low = float(prev_data['Low'].min())
            self._prev_close = float(prev_data['Close'].iloc[-1])
            self._cached_pivots = compute_pivot_points(
                self._prev_high, self._prev_low, self._prev_close
            )
            # Initialize current period accumulator from last (partial) period
            cur_key = unique_keys[-1]
            cur_data = df_tmp[df_tmp['_pk'] == cur_key]
            self._current_period_key = cur_key
            self._period_high = float(cur_data['High'].max())
            self._period_low = float(cur_data['Low'].min())
            self._period_close = float(cur_data['Close'].iloc[-1])
        else:
            # Only one period — accumulate, no pivots yet
            self._current_period_key = unique_keys[0]
            period_data = df_tmp[df_tmp['_pk'] == unique_keys[0]]
            self._period_high = float(period_data['High'].max())
            self._period_low = float(period_data['Low'].min())
            self._period_close = float(period_data['Close'].iloc[-1])

        self._prev_bar_close = float(df.iloc[-1]['Close'])

    def _clear_internal_position(self):
        """Reset strategy-internal position tracking (called on exit signals)."""
        self._sl_price = None
        self._tp_price = None
        self._position_type = None
