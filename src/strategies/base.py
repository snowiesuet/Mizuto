from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    Subclasses must implement `on_price` and `reset`, and set a `name` attribute.

    Strategies that need OHLCV bars (not just close prices) should override
    `on_bar`, `load_ohlcv_history`, and set `requires_ohlcv = True`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this strategy."""
        ...

    @abstractmethod
    def on_price(self, price: float, has_position: bool) -> str:
        """Process a new price tick and return a signal.

        Args:
            price: The current market price.
            has_position: Whether the bot currently holds a position.

        Returns:
            One of ``'buy'``, ``'sell'``, ``'short'``, or ``'hold'``.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state so the strategy can be re-used for another run."""
        ...

    @property
    def requires_ohlcv(self) -> bool:
        """Whether this strategy needs full OHLCV bars instead of just close prices."""
        return False

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before the strategy can produce meaningful signals.

        Subclasses should override this to reflect their indicator requirements.
        """
        return 0

    @property
    def supported_fill_models(self) -> tuple[str, ...]:
        """Fill models this strategy supports.

        The backtest engine warns (but does not block) if the chosen fill model
        is not in this tuple.  Override to restrict.
        """
        return ('close', 'next_open', 'vwap_slippage')

    def on_bar(self, bar: dict, has_position: bool, position_type: str = None) -> str:
        """Process a full OHLCV bar and return a signal.

        The default implementation delegates to ``on_price(bar['Close'], has_position)``.
        Strategies that need OHLCV data should override this method.

        Args:
            bar: Dict with keys 'Open', 'High', 'Low', 'Close', 'Volume'.
            has_position: Whether the bot currently holds a position.
            position_type: ``'long'``, ``'short'``, or ``None``.

        Returns:
            One of ``'buy'``, ``'sell'``, ``'short'``, or ``'hold'``.
        """
        return self.on_price(bar['Close'], has_position)

    def load_price_history(self, prices: list[float]) -> None:
        """Optionally seed the strategy with historical prices.

        The default implementation is a no-op.  Strategies that need historical
        context (e.g. moving-average based) should override this.
        """
        pass

    def load_ohlcv_history(self, df: pd.DataFrame) -> None:
        """Optionally seed the strategy with historical OHLCV data.

        The default implementation extracts the Close column and calls
        ``load_price_history``.  OHLCV strategies should override this.
        """
        if 'Close' in df.columns:
            self.load_price_history(df['Close'].dropna().values.flatten().tolist())
