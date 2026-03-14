import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime


class DataFeed:
    def __init__(self, symbol: str, timeframe=mt5.TIMEFRAME_M5, bars: int = 100):
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars

    def get_candles(self, n=None) -> pd.DataFrame:
        """
        Fetch OHLC candle data from MT5
        """
        num_bars = n if n else self.bars
        rates = mt5.copy_rates_from_pos(
            self.symbol,
            self.timeframe,
            0,
            num_bars
        )

        if rates is None:
            raise RuntimeError(f"Failed to fetch data for {self.symbol}")

        df = pd.DataFrame(rates)

        # Convert timestamp to datetime
        df["time"] = pd.to_datetime(df["time"], unit="s")

        return df
