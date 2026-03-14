"""
utils/indicators.py — Pure numeric indicator functions.

All functions accept Pandas Series / DataFrames and return Series.
No MT5, no strategy, no side effects.
"""

import numpy as np
import pandas as pd


# ── Price / Volume ────────────────────────────────────────────────────────────

def compute_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling Volume-Weighted Average Price."""
    v = df["tick_volume"]
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * v
    return pv.rolling(window=window).sum() / v.rolling(window=window).sum()


def compute_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Z-Score of a series relative to its rolling mean / std."""
    r_mean = series.rolling(window=window).mean()
    r_std = series.rolling(window=window).std()
    return (series - r_mean) / r_std


# ── Volatility ────────────────────────────────────────────────────────────────

def compute_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of log-returns."""
    return returns.rolling(window=window).std()


def compute_volatility_slope(volatility_series: pd.Series, window: int = 5) -> pd.Series:
    """Slope of the volatility curve (first difference over window)."""
    return volatility_series.diff(window).fillna(0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures absolute volatility in price units."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Momentum / Mean-Reversion indicators ─────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_autocorrelation(returns: pd.Series, lag: int = 1) -> pd.Series:
    """Rolling lag-1 autocorrelation of returns."""
    return returns.rolling(window=20).corr(returns.shift(lag))


def compute_hurst(series: pd.Series) -> pd.Series:
    """
    Rolling Hurst Exponent via R/S variance-ratio approximation.

    H < 0.5  → Mean-reverting
    H ≈ 0.5  → Random walk
    H > 0.5  → Trending
    """
    def _hurst_scalar(ts: np.ndarray) -> float:
        if len(ts) < 20:
            return 0.5
        lags = range(2, 20)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        if min(tau) == 0:
            return 0.5
        try:
            m = np.polyfit(np.log(list(lags)), np.log(tau), 1)
            return float(m[0])
        except Exception:
            return 0.5

    return series.rolling(100).apply(_hurst_scalar, raw=True)


# ── Candle Structure ──────────────────────────────────────────────────────────

def compute_wick_body_ratio(df: pd.DataFrame) -> pd.Series:
    """Ratio of wick length to body length — high values flag indecision."""
    body = (df["close"] - df["open"]).abs()
    total_range = df["high"] - df["low"]
    return (total_range - body) / body.replace(0, 0.00001)


def compute_rolling_skewness(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling skewness — useful HMM feature for distribution shape."""
    return series.rolling(window=window).skew()


# ── Trend indicators ─────────────────────────────────────────────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_ema_slope(series: pd.Series, ema_period: int, slope_window: int = 5) -> pd.Series:
    """
    Slope of an EMA as first-difference over `slope_window` bars.

    Positive → price trending up.
    Near-zero → flat / ranging.
    Negative  → price trending down.
    """
    ema = compute_ema(series, period=ema_period)
    return ema.diff(slope_window)


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index (ADX) with +DI and -DI components.

    ADX > 25 → strong trend.
    ADX < 20 → no trend / range-bound.

    Returns:
        DataFrame with columns: adx, plus_di, minus_di.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder smoothing
    atr_s      = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_dm_s  = plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di  = 100 * (plus_dm_s  / atr_s.replace(0, np.nan))
    minus_di = 100 * (minus_dm_s / atr_s.replace(0, np.nan))

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return pd.DataFrame(
        {"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
        index=df.index,
    )

