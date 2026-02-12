"""ADX + ATR Trend Breakout strategy for backtesting.py.

Ported from ``pinescript_ideas/ADX_ATR_Breakout.ps``.
"""

import numpy as np
from backtesting import Strategy

from bt.helpers import adx as calc_adx, atr as calc_atr, rolling_max, rolling_min


class ATRBreakoutStrategy(Strategy):
    """ADX + ATR breakout with ATR-scaled stop-loss and take-profit.

    Entry logic:
        - ADX above *adx_threshold* signals a strong trend.
        - +DI > -DI → bullish; -DI > +DI → bearish.
        - Price breaks above/below ATR-offset recent high/low.

    Exit logic:
        - ATR-based stop-loss and take-profit on every entry.
        - Emergency close when ADX drops below threshold * 0.7.
    """

    adx_length = 14
    adx_threshold = 25
    atr_length = 14
    atr_multiplier = 1.5
    lookback_period = 20
    stop_loss_atr = 2.0
    take_profit_atr = 3.0

    def init(self):
        self._adx, self._plus_di, self._minus_di = self.I(
            calc_adx,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.adx_length,
            overlay=False,
        )
        self._atr = self.I(
            calc_atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_length,
            overlay=False,
        )
        self._recent_high = self.I(rolling_max, self.data.High, self.lookback_period)
        self._recent_low = self.I(rolling_min, self.data.Low, self.lookback_period)

        self._bars_since_entry = 0

    def next(self):
        adx_val = self._adx[-1]
        plus_di = self._plus_di[-1]
        minus_di = self._minus_di[-1]
        atr_val = self._atr[-1]
        price = self.data.Close[-1]
        recent_high = self._recent_high[-1]
        recent_low = self._recent_low[-1]

        if np.isnan(adx_val) or np.isnan(atr_val) or np.isnan(recent_high):
            return

        breakout_high = recent_high + atr_val * self.atr_multiplier
        breakout_low = recent_low - atr_val * self.atr_multiplier

        strong_trend = adx_val > self.adx_threshold
        weak_trend = adx_val < self.adx_threshold * 0.7

        # Emergency exit on weak trend
        if weak_trend and self.position:
            self.position.close()
            return

        self._bars_since_entry += 1

        # Long entry
        if (
            not self.position
            and strong_trend
            and plus_di > minus_di
            and price > breakout_high
            and self._bars_since_entry >= 5
        ):
            sl = price - atr_val * self.stop_loss_atr
            tp = price + atr_val * self.take_profit_atr
            self.buy(sl=sl, tp=tp)
            self._bars_since_entry = 0

        # Short entry
        elif (
            not self.position
            and strong_trend
            and minus_di > plus_di
            and price < breakout_low
            and self._bars_since_entry >= 5
        ):
            sl = price + atr_val * self.stop_loss_atr
            tp = price - atr_val * self.take_profit_atr
            self.sell(sl=sl, tp=tp)
            self._bars_since_entry = 0
