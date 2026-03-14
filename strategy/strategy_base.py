from abc import ABC, abstractmethod


class StrategyBase(ABC):
    """
    Base class for all trading strategies
    """

    @abstractmethod
    def generate_signal(self, data):
        """
        Must return one of:
        - 'BUY'
        - 'SELL'
        - 'HOLD'
        """
        pass
