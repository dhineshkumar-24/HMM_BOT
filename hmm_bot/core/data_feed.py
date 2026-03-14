"""
core/data_feed.py — MT5 OHLCV data fetcher.

Thin wrapper around mt5.copy_rates_from_pos().
No strategy or indicator logic lives here.
"""

import pandas as pd
import MetaTrader5 as mt5


class DataFeed:
    """Fetches candle data from the MT5 terminal."""

    def __init__(self, symbol: str, timeframe: int = mt5.TIMEFRAME_M1, bars: int = 500):
        """
        Args:
            symbol:    Instrument ticker (e.g. "EURUSD").
            timeframe: MT5 timeframe constant (e.g. mt5.TIMEFRAME_M1).
            bars:      Default number of bars to fetch.
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars

    def get_candles(self, n: int | None = None) -> pd.DataFrame:
        """
        Fetch the most recent `n` OHLCV candles from MT5.

        Args:
            n: Number of bars (uses self.bars if None).

        Returns:
            DataFrame with columns: time, open, high, low, close,
            tick_volume, spread, real_volume.

        Raises:
            RuntimeError: If MT5 returns no data.
        """
        num_bars = n if n is not None else self.bars
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, num_bars)

        if rates is None:
            raise RuntimeError(
                f"[DataFeed] MT5 returned no data for {self.symbol}. "
                f"Error: {mt5.last_error()}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df
