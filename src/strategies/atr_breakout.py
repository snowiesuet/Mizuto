"""ADX + ATR Trend Breakout strategy for the src/ backtest engine.

Ported from ``bt/strategies/atr_breakout.py`` with short-selling support.
"""

import logging
import math

import pandas as pd

from src.strategies.base import BaseStrategy
from src.indicators import compute_atr, compute_adx, compute_rolling_max, compute_rolling_min


class ATRBreakoutStrategy(BaseStrategy):
    """ADX + ATR breakout with strategy-managed SL/TP.

    Entry logic:
        - ADX above *adx_threshold* signals a strong trend.
        - +DI > -DI → long;  -DI > +DI → short.
        - Price breaks above/below ATR-offset recent high/low.
        - 5-bar cooldown between entries.

    Exit logic:
        - Strategy-internal ATR-based stop-loss and take-profit.
        - Emergency close when ADX drops below ``adx_threshold * 0.7``.
    """

    name = "ATR Breakout"

    def __init__(
        self,
        adx_length: int = 14,
        adx_threshold: float = 25,
        atr_length: int = 14,
        atr_multiplier: float = 1.5,
        lookback_period: int = 20,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 3.0,
    ):
        self.adx_length = adx_length
        self.adx_threshold = adx_threshold
        self.atr_length = atr_length
        self.atr_multiplier = atr_multiplier
        self.lookback_period = lookback_period
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr

        # Rolling OHLCV history
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []

        # Internal position tracking for SL/TP
        self._sl_price: float | None = None
        self._tp_price: float | None = None
        self._position_type: str | None = None  # 'long' or 'short'

        self._bars_since_entry: int = 5  # start at 5 so first entry is allowed

    @property
    def requires_ohlcv(self) -> bool:
        return True

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before the strategy can produce signals."""
        return max(self.adx_length * 2, self.atr_length, self.lookback_period) + 1

    # --- BaseStrategy interface ---------------------------------------------------

    def on_price(self, price: float, has_position: bool) -> str:
        """Fallback for close-only data — not recommended for this strategy."""
        bar = {'Open': price, 'High': price, 'Low': price, 'Close': price, 'Volume': 0}
        return self.on_bar(bar, has_position)

    def on_bar(self, bar: dict, has_position: bool, position_type: str = None) -> str:
        price = float(bar['Close'])
        high = float(bar['High'])
        low = float(bar['Low'])

        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(price)

        # Cap history to avoid unbounded growth
        max_len = self.warmup_period + 50
        if len(self._closes) > max_len:
            self._highs = self._highs[-max_len:]
            self._lows = self._lows[-max_len:]
            self._closes = self._closes[-max_len:]

        # Need enough data for indicators
        min_needed = self.warmup_period
        if len(self._closes) < min_needed:
            return 'hold'

        # Compute indicators
        atr_val = compute_atr(self._highs, self._lows, self._closes, self.atr_length)
        adx_val, plus_di, minus_di = compute_adx(
            self._highs, self._lows, self._closes, self.adx_length
        )

        if math.isnan(adx_val) or math.isnan(atr_val):
            return 'hold'

        recent_high = compute_rolling_max(self._highs, self.lookback_period)
        recent_low = compute_rolling_min(self._lows, self.lookback_period)

        if math.isnan(recent_high) or math.isnan(recent_low):
            return 'hold'

        breakout_high = recent_high + atr_val * self.atr_multiplier
        breakout_low = recent_low - atr_val * self.atr_multiplier

        strong_trend = adx_val > self.adx_threshold
        weak_trend = adx_val < self.adx_threshold * 0.7

        self._bars_since_entry += 1

        # --- Exit logic (strategy-managed SL/TP) ---
        if has_position and self._position_type is not None:
            # Emergency exit on weak trend
            if weak_trend:
                logging.info(f"ATR Breakout: emergency exit — ADX {adx_val:.1f} < {self.adx_threshold * 0.7:.1f}")
                self._clear_internal_position()
                return 'sell'

            # Check SL/TP
            if self._position_type == 'long':
                if self._sl_price is not None and price <= self._sl_price:
                    logging.info(f"ATR Breakout: long SL hit at {price:.2f}")
                    self._clear_internal_position()
                    return 'sell'
                if self._tp_price is not None and price >= self._tp_price:
                    logging.info(f"ATR Breakout: long TP hit at {price:.2f}")
                    self._clear_internal_position()
                    return 'sell'
            elif self._position_type == 'short':
                if self._sl_price is not None and price >= self._sl_price:
                    logging.info(f"ATR Breakout: short SL hit at {price:.2f}")
                    self._clear_internal_position()
                    return 'sell'
                if self._tp_price is not None and price <= self._tp_price:
                    logging.info(f"ATR Breakout: short TP hit at {price:.2f}")
                    self._clear_internal_position()
                    return 'sell'

            return 'hold'

        # --- Entry logic ---
        if has_position:
            return 'hold'

        # Long entry
        if (
            strong_trend
            and plus_di > minus_di
            and price > breakout_high
            and self._bars_since_entry >= 5
        ):
            self._sl_price = price - atr_val * self.stop_loss_atr
            self._tp_price = price + atr_val * self.take_profit_atr
            self._position_type = 'long'
            self._bars_since_entry = 0
            logging.info(f"ATR Breakout: LONG entry at {price:.2f}, SL={self._sl_price:.2f}, TP={self._tp_price:.2f}")
            return 'buy'

        # Short entry
        if (
            strong_trend
            and minus_di > plus_di
            and price < breakout_low
            and self._bars_since_entry >= 5
        ):
            self._sl_price = price + atr_val * self.stop_loss_atr
            self._tp_price = price - atr_val * self.take_profit_atr
            self._position_type = 'short'
            self._bars_since_entry = 0
            logging.info(f"ATR Breakout: SHORT entry at {price:.2f}, SL={self._sl_price:.2f}, TP={self._tp_price:.2f}")
            return 'short'

        return 'hold'

    def reset(self) -> None:
        self._highs = []
        self._lows = []
        self._closes = []
        self._sl_price = None
        self._tp_price = None
        self._position_type = None
        self._bars_since_entry = 5

    def load_ohlcv_history(self, df: pd.DataFrame) -> None:
        self._highs = df['High'].dropna().values.flatten().tolist()
        self._lows = df['Low'].dropna().values.flatten().tolist()
        self._closes = df['Close'].dropna().values.flatten().tolist()
        logging.info(f"ATR Breakout loaded {len(self._closes)} historical bars.")

    # --- Internal helpers --------------------------------------------------------

    def _clear_internal_position(self):
        """Reset strategy-internal position tracking (called on exit signals)."""
        self._sl_price = None
        self._tp_price = None
        self._position_type = None
