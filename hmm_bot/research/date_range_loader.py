"""
research/date_range_loader.py — Utility for large date-range data fetching.

Use this when you need more control than load_mt5_history_range() provides:
  - Multi-symbol batch download
  - Explicit cache management
  - Date range validation helpers

Usage:
    from research.date_range_loader import DateRangeLoader
    loader = DateRangeLoader(symbol="EURUSD", timeframe="M5",
                             cache_dir="data/cache")
    df = loader.load("2020-01-24", "2026-01-01")
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


# ── Timeframe string → MT5 constant map ──────────────────────────────────────
_TF_STR_MAP = {
    "M1":  1,
    "M5":  5,
    "M15": 15,
    "H1":  16385,
    "H4":  16388,
    "D1":  16408,
}


class DateRangeLoader:
    """
    Fetches and caches a date-range of OHLCV data from MT5.

    Features:
        - Chunked fetching (6-month windows) to bypass broker limits
        - Parquet caching — re-runs are instant
        - Validates date range and prints progress
        - Handles broker data gaps gracefully
    """

    def __init__(
        self,
        symbol:    str,
        timeframe: str  = "M5",
        cache_dir: str  = "data/cache",
    ):
        self.symbol    = symbol
        self.tf_str    = timeframe
        self.cache_dir = cache_dir

        # Convert string to MT5 constant lazily (MT5 may not be imported yet)
        self._tf_int: Optional[int] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(
        self,
        start_date:    str,
        end_date:      str,
        force_refetch: bool = False,
    ) -> pd.DataFrame:
        """
        Load data for the given date range.

        Args:
            start_date:    "YYYY-MM-DD"
            end_date:      "YYYY-MM-DD"
            force_refetch: Ignore cache.

        Returns:
            Clean OHLCV DataFrame, sorted chronologically.
        """
        self._validate_dates(start_date, end_date)
        cache_path = self._cache_path(start_date, end_date)

        if not force_refetch and os.path.exists(cache_path):
            print(f"[DateRangeLoader] Cache hit → {cache_path}")
            df = pd.read_parquet(cache_path)
            self._print_summary(df, start_date, end_date)
            return df

        df = self._fetch_from_mt5(start_date, end_date)
        self._save_cache(df, cache_path)
        self._print_summary(df, start_date, end_date)
        return df

    def split(
        self,
        df:         pd.DataFrame,
        train_frac: float = 0.70,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chronological train/test split with no data leakage.

        Args:
            df:         Full dataset (sorted by time).
            train_frac: Fraction for training (default 0.70 = 70%).

        Returns:
            (df_train, df_test) — both reset-indexed.
        """
        n     = len(df)
        split = int(n * train_frac)
        df_train = df.iloc[:split].reset_index(drop=True)
        df_test  = df.iloc[split:].reset_index(drop=True)

        print(f"[DateRangeLoader] Split → Train: {len(df_train):,} bars "
              f"({df_train['time'].iloc[0].date()} → {df_train['time'].iloc[-1].date()})")
        print(f"[DateRangeLoader]         Test:  {len(df_test):,} bars "
              f"({df_test['time'].iloc[0].date()} → {df_test['time'].iloc[-1].date()})")
        return df_train, df_test

    def cache_info(self, start_date: str, end_date: str) -> dict:
        """Return metadata about the cached file if it exists."""
        path = self._cache_path(start_date, end_date)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1_048_576
            return {"exists": True, "path": path, "size_mb": round(size_mb, 2)}
        return {"exists": False, "path": path}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _mt5_tf(self) -> int:
        if self._tf_int is None:
            try:
                import MetaTrader5 as mt5
                _MAP = {
                    "M1":  mt5.TIMEFRAME_M1,
                    "M5":  mt5.TIMEFRAME_M5,
                    "M15": mt5.TIMEFRAME_M15,
                    "H1":  mt5.TIMEFRAME_H1,
                    "H4":  mt5.TIMEFRAME_H4,
                    "D1":  mt5.TIMEFRAME_D1,
                }
                self._tf_int = _MAP.get(self.tf_str, mt5.TIMEFRAME_M5)
            except ImportError:
                self._tf_int = _TF_STR_MAP.get(self.tf_str, 5)
        return self._tf_int

    def _cache_path(self, start_date: str, end_date: str) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        fname = f"{self.symbol}_{self.tf_str}_{start_date}_{end_date}.parquet"
        return os.path.join(self.cache_dir, fname)

    def _fetch_from_mt5(self, start_date: str, end_date: str) -> pd.DataFrame:
        import MetaTrader5 as mt5

        if not mt5.terminal_info():
            if not mt5.initialize():
                raise RuntimeError("MT5 failed to initialize.")

        dt_start = datetime.strptime(start_date, "%Y-%m-%d")
        dt_end   = datetime.strptime(end_date,   "%Y-%m-%d")
        chunks   = []
        current  = dt_start

        print(f"[DateRangeLoader] Fetching {self.symbol} {self.tf_str} "
              f"from {start_date} to {end_date}")

        while current < dt_end:
            chunk_end = min(current + timedelta(days=180), dt_end)
            rates = mt5.copy_rates_range(self.symbol, self._mt5_tf(), current, chunk_end)

            if rates is not None and len(rates) > 0:
                chunk_df = pd.DataFrame(rates)
                chunk_df["time"] = pd.to_datetime(chunk_df["time"], unit="s")
                chunks.append(chunk_df)
                print(f"  {current.date()} → {chunk_end.date()} | {len(chunk_df):,} bars")
            else:
                print(f"  {current.date()} → {chunk_end.date()} | No data (broker gap)")

            current = chunk_end

        if not chunks:
            raise RuntimeError(
                f"No data returned for {self.symbol} {start_date}→{end_date}.\n"
                f"Broker may not have data this far back."
            )

        df = pd.concat(chunks, ignore_index=True)
        df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

        # Normalize columns
        if "tick_volume" not in df.columns and "real_volume" in df.columns:
            df["tick_volume"] = df["real_volume"]
        keep = ["time", "open", "high", "low", "close", "tick_volume"]
        df   = df[[c for c in keep if c in df.columns]]
        return df

    @staticmethod
    def _validate_dates(start: str, end: str) -> None:
        fmt = "%Y-%m-%d"
        try:
            s = datetime.strptime(start, fmt)
            e = datetime.strptime(end,   fmt)
        except ValueError as exc:
            raise ValueError(f"Date format must be YYYY-MM-DD. Got: {exc}") from exc
        if s >= e:
            raise ValueError(f"start_date ({start}) must be before end_date ({end})")
        if s < datetime(2000, 1, 1):
            raise ValueError("start_date before 2000 is unreliable with most brokers.")

    @staticmethod
    def _print_summary(df: pd.DataFrame, start: str, end: str) -> None:
        actual_start = df["time"].iloc[0].date()
        actual_end   = df["time"].iloc[-1].date()
        coverage     = (actual_end - actual_start).days
        print(f"[DateRangeLoader] {len(df):,} bars | "
              f"{actual_start} → {actual_end} ({coverage} days)")