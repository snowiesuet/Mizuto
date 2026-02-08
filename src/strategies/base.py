from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    Subclasses must implement `on_price` and `reset`, and set a `name` attribute.
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
            One of ``'buy'``, ``'sell'``, or ``'hold'``.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state so the strategy can be re-used for another run."""
        ...

    def load_price_history(self, prices: list[float]) -> None:
        """Optionally seed the strategy with historical prices.

        The default implementation is a no-op.  Strategies that need historical
        context (e.g. moving-average based) should override this.
        """
        pass
