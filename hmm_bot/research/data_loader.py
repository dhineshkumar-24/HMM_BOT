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
# Date-Range Loader (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def load_mt5_history_range(
    symbol:      str,
    timeframe:   int,
    start_date:  str,           # "YYYY-MM-DD"
    end_date:    str,           # "YYYY-MM-DD"
    cache_dir:   str = "data/cache",
    force_refetch: bool = False,
) -> pd.DataFrame:
    """
    Fetch a full date-range of OHLCV data from MT5 using chunked pagination.

    MT5 brokers typically cap per-request bar counts. This function breaks the
    date range into 6-month windows and stitches them together. Results are
    cached to Parquet so subsequent runs are instant.

    Args:
        symbol:       Instrument (e.g. "EURUSD").
        timeframe:    MT5 timeframe constant.
        start_date:   "YYYY-MM-DD" string.
        end_date:     "YYYY-MM-DD" string.
        cache_dir:    Folder for Parquet cache files.
        force_refetch: If True, ignore cache and re-fetch from MT5.

    Returns:
        Cleaned DataFrame with columns: time, open, high, low, close,
        tick_volume. Sorted chronologically, no duplicates.
    """
    from datetime import timedelta
    import os

    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end   = datetime.strptime(end_date,   "%Y-%m-%d")

    # ── Check local cache first ───────────────────────────────────────────────
    tf_name     = _timeframe_name(timeframe)
    cache_base  = os.path.join(cache_dir, f"{symbol}_{tf_name}_{start_date}_{end_date}")
    cache_file  = f"{cache_base}.parquet"
    cache_fallback_file = f"{cache_base}.pkl"

    if not force_refetch and os.path.exists(cache_file):
        print(f"[DataLoader] Loading cached data: {cache_file}")
        try:
            df = pd.read_parquet(cache_file)
            print(f"[DataLoader] Cache hit — {len(df):,} bars of {symbol} ({start_date} → {end_date})")
            return df
        except (ImportError, ValueError) as exc:
            print(f"[DataLoader] Parquet cache unavailable ({exc}). Falling back to pickle cache.")

    if not force_refetch and os.path.exists(cache_fallback_file):
        print(f"[DataLoader] Loading fallback cache: {cache_fallback_file}")
        df = pd.read_pickle(cache_fallback_file)
        print(f"[DataLoader] Cache hit — {len(df):,} bars of {symbol} ({start_date} → {end_date})")
        return df

    # ── MT5 chunked fetch ─────────────────────────────────────────────────────
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError("MetaTrader5 package required.")

    if not mt5.terminal_info():
        if not mt5.initialize():
            raise RuntimeError("MT5 failed to initialize. Open MT5 terminal first.")

    chunks       = []
    chunk_start  = dt_start
    chunk_months = 6          # fetch 6 months at a time — safe for all brokers
    total_fetched = 0

    print(f"[DataLoader] Fetching {symbol} {tf_name} from {start_date} to {end_date}...")

    while chunk_start < dt_end:
        chunk_end = min(
            chunk_start + timedelta(days=chunk_months * 30),
            dt_end
        )

        rates = mt5.copy_rates_range(symbol, timeframe, chunk_start, chunk_end)

        if rates is not None and len(rates) > 0:
            chunk_df = pd.DataFrame(rates)
            chunk_df["time"] = pd.to_datetime(chunk_df["time"], unit="s")
            chunks.append(chunk_df)
            total_fetched += len(chunk_df)
            print(
                f"  Chunk {chunk_start.date()} → {chunk_end.date()} "
                f"| {len(chunk_df):,} bars | Total so far: {total_fetched:,}"
            )
        else:
            # Broker doesn't have this period — skip silently
            print(f"  Chunk {chunk_start.date()} → {chunk_end.date()} | No data (broker gap)")

        chunk_start = chunk_end

    if not chunks:
        raise RuntimeError(
            f"MT5 returned no data for {symbol} between {start_date} and {end_date}.\n"
            f"Your broker may not have data this far back.\n"
            f"Try a shorter range first, e.g. start_date='2022-01-01'."
        )

    # ── Stitch & deduplicate ──────────────────────────────────────────────────
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    df = _clean(df)

    # ── Cache to disk ─────────────────────────────────────────────────────────
    os.makedirs(cache_dir, exist_ok=True)
    try:
        df.to_parquet(cache_file, index=False)
        print(f"[DataLoader] Saved {len(df):,} bars to cache: {cache_file}")
    except (ImportError, ValueError) as exc:
        df.to_pickle(cache_fallback_file)
        print(
            f"[DataLoader] Parquet engine missing ({exc}). "
            f"Saved fallback cache: {cache_fallback_file}"
        )

    return df


def _timeframe_name(timeframe: int) -> str:
    """Convert MT5 timeframe int to a readable string for cache filenames."""
    try:
        import MetaTrader5 as mt5
        _MAP = {
            mt5.TIMEFRAME_M1:  "M1",
            mt5.TIMEFRAME_M5:  "M5",
            mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_H1:  "H1",
            mt5.TIMEFRAME_H4:  "H4",
            mt5.TIMEFRAME_D1:  "D1",
        }
        return _MAP.get(timeframe, str(timeframe))
    except ImportError:
        return str(timeframe)
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
