"""Classic floor pivot-point bounce strategy for backtesting.py."""

import numpy as np
from backtesting import Strategy

from bt.helpers import pivot_points as calc_pivots


class PivotPointStrategy(Strategy):
    """Trade bounces off classic floor pivot levels.

    Long on S1 bounce (price crosses back above S1), stop at S2, TP at PP.
    Short on R1 rejection (price crosses back below R1), stop at R2, TP at PP.

    Set *use_s2_r2* to ``True`` for second-level entries (S2/R2 bounces with
    stops at S3/R3 and TP at S1/R1).
    """

    use_s2_r2 = False

    def init(self):
        pivots = calc_pivots(self.data.High, self.data.Low, self.data.Close)
        self._pp = self.I(lambda: pivots["PP"], overlay=True)
        self._s1 = self.I(lambda: pivots["S1"], overlay=True)
        self._s2 = self.I(lambda: pivots["S2"], overlay=True)
        self._s3 = self.I(lambda: pivots["S3"], overlay=True)
        self._r1 = self.I(lambda: pivots["R1"], overlay=True)
        self._r2 = self.I(lambda: pivots["R2"], overlay=True)
        self._r3 = self.I(lambda: pivots["R3"], overlay=True)

    def next(self):
        if len(self.data.Close) < 3:
            return

        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]
        pp = self._pp[-1]
        s1 = self._s1[-1]
        s2 = self._s2[-1]
        s3 = self._s3[-1]
        r1 = self._r1[-1]
        r2 = self._r2[-1]
        r3 = self._r3[-1]

        if np.isnan(pp):
            return

        if self.position:
            return

        # S1 bounce: price was below S1, now crosses back above
        if prev_price < s1 and price >= s1:
            self.buy(sl=s2, tp=pp)
            return

        # R1 rejection: price was above R1, now crosses back below
        if prev_price > r1 and price <= r1:
            self.sell(sl=r2, tp=pp)
            return

        # Optional second-level entries
        if self.use_s2_r2:
            if prev_price < s2 and price >= s2:
                self.buy(sl=s3, tp=s1)
                return
            if prev_price > r2 and price <= r2:
                self.sell(sl=r3, tp=r1)
                return
