"""MA Crossover strategy for backtesting.py.

Same default windows (short=5, long=20) as the custom engine version in
``src/strategies/ma_crossover.py``.
"""

from backtesting import Strategy
from backtesting.lib import crossover

from bt.helpers import sma


class MACrossoverBT(Strategy):
    """Simple moving-average crossover — long only.

    Buy when the short MA crosses above the long MA; sell when it crosses
    below.  Parameters are class-level so ``Backtest.optimize()`` can sweep
    them.
    """

    short_window = 5
    long_window = 20

    def init(self):
        self.short_ma = self.I(sma, self.data.Close, self.short_window)
        self.long_ma = self.I(sma, self.data.Close, self.long_window)

    def next(self):
        if crossover(self.short_ma, self.long_ma):
            self.buy()
        elif crossover(self.long_ma, self.short_ma):
            self.position.close()
