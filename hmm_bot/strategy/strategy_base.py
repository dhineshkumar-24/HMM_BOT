"""
strategy/strategy_base.py — Abstract base class for all HMM Bot strategies.

generate_signal() now returns a signal dict or None, replacing the
old str ("BUY"/"SELL"/"HOLD") return type. This matches the structured
signal format required by the executor and router.

Signal format:
    {
        "direction": "BUY" | "SELL",   # trade direction
        "entry":     float,             # entry price (current close)
        "sl":        float,             # stop-loss price
        "tp":        float,             # take-profit price
        "atr":       float,             # current ATR value (for trailing)
        "reason":    str,               # human-readable trigger description
    }
    OR None — meaning no trade this bar.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class StrategyBase(ABC):
    """
    Abstract base class for all HMM Bot trading strategies.

    Subclasses must implement:
        calculate_indicators(df) → enriched DataFrame
        generate_signal(df, regime, session) → signal dict or None
    """

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich a raw OHLCV DataFrame with strategy-specific indicator columns.

        Args:
            df: Raw candle DataFrame from MT5.

        Returns:
            DataFrame with additional indicator columns appended in-place copy.
        """
        ...

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        regime: Optional[int] = None,
        session: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Analyse the latest closed candle and produce a trade signal.

        Args:
            df:      Enriched DataFrame (output of calculate_indicators).
            regime:  HMM regime label (0=mean_reverting, 1=trending, 2=high_vol).
                     None during HMM warm-up — strategy decides whether to trade.
            session: Active session label from helpers.detect_session().
                     None if called without session context.

        Returns:
            Signal dict:
                {
                    "direction": "BUY" | "SELL",
                    "entry":     float,
                    "sl":        float,
                    "tp":        float,
                    "atr":       float,
                    "reason":    str,
                }
            OR None if no trade opportunity this bar.
        """
        ...
