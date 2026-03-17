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

    def get_candles_tf(self, timeframe_str: str, n: int = 200):
        """Fetch candles for any timeframe string: H4, H1, M15, M1."""
        tf_map = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
        }
        tf = tf_map.get(timeframe_str, mt5.TIMEFRAME_M1)
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, n)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def get_4h_bias(self, ema_fast: int = 50, ema_slow: int = 200) -> str:
        """Return 'UP', 'DOWN', or 'NEUTRAL' based on 4H EMA alignment."""
        df = self.get_candles_tf("H4", n=250)
        if df is None or len(df) < ema_slow:
            return "NEUTRAL"
        close = df["close"]
        ema_f = close.ewm(span=ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=ema_slow, adjust=False).mean()
        if float(ema_f.iloc[-1]) > float(ema_s.iloc[-1]):
            return "UP"
        elif float(ema_f.iloc[-1]) < float(ema_s.iloc[-1]):
            return "DOWN"
        return "NEUTRAL"
