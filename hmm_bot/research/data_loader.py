"""
research/data_loader.py — Historical data loader for backtesting.

Two sources:
    1. MT5 live terminal  — load_mt5_history(symbol, timeframe, bars)
    2. CSV file           — load_csv_history(path)

Both return a cleaned pandas DataFrame with OHLCV columns normalised to:
    time, open, high, low, close, tick_volume
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# MT5 Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_mt5_history(
    symbol:    str,
    timeframe: int,
    bars:      int = 50_000,
    start:     Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from the MT5 terminal.

    Args:
        symbol:    Instrument symbol (e.g. "EURUSD").
        timeframe: MT5 timeframe constant (e.g. mt5.TIMEFRAME_M1).
        bars:      Number of bars to fetch from present.
        start:     If provided, fetch from this datetime onwards (overrides bars).

    Returns:
        DataFrame with columns: time, open, high, low, close, tick_volume.

    Raises:
        RuntimeError: If MT5 is not initialised or data not available.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError("MetaTrader5 package is required for live data loading.")

    if not mt5.terminal_info():
        # Auto-initialise if not already connected
        if not mt5.initialize():
            raise RuntimeError("MT5 failed to initialize. Open MT5 terminal first.")

    if start is not None:
        rates = mt5.copy_rates_from(symbol, timeframe, start, bars)
    else:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        raise RuntimeError(
            f"MT5 returned no data for {symbol}. "
            f"Error: {err}"
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = _clean(df)
    print(f"[DataLoader] Loaded {len(df):,} bars of {symbol} from MT5.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CSV Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_history(path: str) -> pd.DataFrame:
    """
    Load historical data from a CSV file.

    Expected columns (case-insensitive):
        time/date/datetime, open, high, low, close, volume/tick_volume

    Args:
        path: Absolute or relative path to the CSV file.

    Returns:
        Cleaned DataFrame with normalised OHLCV columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Normalise time column
    for time_col in ("time", "date", "datetime", "timestamp"):
        if time_col in df.columns:
            df = df.rename(columns={time_col: "time"})
            break

    df["time"] = pd.to_datetime(df["time"])

    # Normalise volume column
    for vol_col in ("volume", "tick_volume", "vol"):
        if vol_col in df.columns and vol_col != "tick_volume":
            df = df.rename(columns={vol_col: "tick_volume"})
            break

    df = _clean(df)
    print(f"[DataLoader] Loaded {len(df):,} bars from CSV: {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise, sort, drop rows with missing OHLC, reset index."""
    required = ["time", "open", "high", "low", "close"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0.0

    df = (
        df[["time", "open", "high", "low", "close", "tick_volume"]]
        .dropna(subset=["open", "high", "low", "close"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    return df


def split_date_range(
    df:         pd.DataFrame,
    train_frac: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple chronological train/test split by fraction.

    Args:
        df:         Full DataFrame, sorted by time.
        train_frac: Fraction of rows to use for training (default 70%).

    Returns:
        (train_df, test_df) with no overlap.
    """
    n     = len(df)
    split = int(n * train_frac)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
