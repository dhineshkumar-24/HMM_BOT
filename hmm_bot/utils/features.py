"""
utils/features.py — HMM feature engineering pipeline.

Computes and normalizes the 7-feature vector used by the HMM regime detector:

    1. log_return           — per-bar log return
    2. realized_vol         — rolling 20-bar realized volatility
    3. vol_of_vol           — rolling std of realized_vol (vol regime changepoint)
    4. autocorr             — lag-1 autocorrelation of returns
    5. atr_norm             — ATR normalized by close price
    6. volume_zscore        — z-score of tick volume
    7. momentum             — n-bar rate-of-change of close price

NOTE: Normalization (StandardScaler) is handled inside HMMRegimeDetector.fit()
and HMMRegimeDetector.predict(), not here. This module returns raw feature values
so the scaler can be fit on training data and reused for live prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Feature column names — must stay stable between train & predict
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "log_return",
    "realized_vol",
    "vol_of_vol",
    "autocorr",
    "atr_norm",
    "volume_zscore",
    "momentum",
    "skewness",
    "kurtosis",
    "drawdown_pct",
    "volume_trend",
    "ema_slope",
    "hurst_rolling",
]


# ─────────────────────────────────────────────────────────────────────────────
# Individual feature calculators (vectorised Pandas, no loops)
# ─────────────────────────────────────────────────────────────────────────────

def _log_returns(close: pd.Series) -> pd.Series:
    """Bar-by-bar log return: ln(close_t / close_{t-1})."""
    return np.log(close / close.shift(1))


def _realized_volatility(
    log_ret: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling standard deviation of log returns (annualized-ready)."""
    return log_ret.rolling(window=window).std()


def _vol_of_vol(
    realized_vol: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Rolling standard deviation of realized volatility.

    High vol-of-vol → regime likely transitioning.
    Low vol-of-vol  → regime is stable.
    """
    return realized_vol.rolling(window=window).std()


def _autocorrelation(
    log_ret: pd.Series,
    lag: int = 1,
    window: int = 20,
) -> pd.Series:
    """
    Rolling lag-1 autocorrelation of log returns.

    Strongly negative → mean-reverting tendency.
    Near zero          → random walk.
    Strongly positive  → trending / momentum.
    """
    return log_ret.rolling(window=window).corr(log_ret.shift(lag))


def _atr_normalized(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range divided by close price.

    Gives a dimensionless volatility measure comparable across price levels.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr / df["close"]


def _volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Z-score of tick volume relative to its rolling mean and std.

    Spikes indicate unusual market participation (news, breakout, etc.).
    """
    roll_mean = volume.rolling(window=window).mean()
    roll_std  = volume.rolling(window=window).std().replace(0, np.nan)
    return (volume - roll_mean) / roll_std


def _momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Rate-of-change momentum: (close_t / close_{t-period}) - 1.

    Positive → upward momentum.
    Negative → downward momentum.
    Near-zero → ranging/consolidating.
    """
    return (close / close.shift(period)) - 1


def _skewness(log_ret: pd.Series, window: int = 20) -> pd.Series:
    """Rolling skewness of log returns (negative = downside tail risk)."""
    return log_ret.rolling(window=window).skew()


def _kurtosis(log_ret: pd.Series, window: int = 20) -> pd.Series:
    """Rolling kurtosis of log returns (high = fat tails)."""
    return log_ret.rolling(window=window).kurt()


def _drawdown_pct(close: pd.Series, window: int = 20) -> pd.Series:
    """Percentage drawdown from recent rolling high."""
    rolling_max = close.rolling(window=window).max()
    return (close - rolling_max) / rolling_max


def _volume_trend(volume: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling correlation between volume and close price."""
    return volume.rolling(window=window).corr(close)


def _ema_slope(close: pd.Series, span: int = 50) -> pd.Series:
    """Normalized slope of the EMA50."""
    ema = close.ewm(span=span, adjust=False).mean()
    return ema.diff() / close.shift(1)


def _hurst_rolling(close: pd.Series, window: int = 100) -> pd.Series:
    """Rolling Hurst Exponent."""
    from utils.indicators import compute_hurst
    return compute_hurst(close)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    vol_window:   int = 20,
    atr_period:   int = 14,
    vol_lag:      int = 1,
    autocorr_lag: int = 1,
    mom_period:   int = 10,
) -> pd.DataFrame:
    """
    Compute the 7-feature matrix from a raw OHLCV candle DataFrame.

    The returned DataFrame has exactly FEATURE_COLS as columns.
    All rows containing NaN (from warm-up periods) are dropped.

    Args:
        df:            Raw candle DataFrame with columns:
                       open, high, low, close, tick_volume, time.
        vol_window:    Rolling window for realized vol and vol-of-vol (bars).
        atr_period:    ATR smoothing period (bars).
        vol_lag:       Lag for vol-of-vol rolling window (same as vol_window).
        autocorr_lag:  Autocorrelation lag (default 1 = lag-1).
        mom_period:    Lookback for momentum rate-of-change (bars).

    Returns:
        pd.DataFrame of shape (n_valid_bars, 7) with FEATURE_COLS columns.
        Index matches the input df index (for bar alignment).

    Example:
        >>> features = build_feature_matrix(df)
        >>> X = features.values          # numpy array for HMM
    """
    df = df.copy()

    # ── Raw computations ─────────────────────────────────────────────────────
    log_ret  = _log_returns(df["close"])
    rvol     = _realized_volatility(log_ret, window=vol_window)
    vov      = _vol_of_vol(rvol, window=vol_window)
    ac       = _autocorrelation(log_ret, lag=autocorr_lag, window=vol_window)
    atr_n    = _atr_normalized(df, period=atr_period)
    vol_z    = _volume_zscore(df["tick_volume"], window=vol_window)
    mom      = _momentum(df["close"], period=mom_period)

    skew     = _skewness(log_ret, window=vol_window)
    kurt     = _kurtosis(log_ret, window=vol_window)
    dd_pct   = _drawdown_pct(df["close"], window=vol_window)
    vol_t    = _volume_trend(df["tick_volume"], df["close"], window=vol_window)
    ema_s    = _ema_slope(df["close"], span=50)
    hurst_r  = _hurst_rolling(df["close"], window=100)

    # ── Assemble ──────────────────────────────────────────────────────────────
    features = pd.DataFrame(
        {
            "log_return":    log_ret,
            "realized_vol":  rvol,
            "vol_of_vol":    vov,
            "autocorr":      ac,
            "atr_norm":      atr_n,
            "volume_zscore": vol_z,
            "momentum":      mom,
            "skewness":      skew,
            "kurtosis":      kurt,
            "drawdown_pct":  dd_pct,
            "volume_trend":  vol_t,
            "ema_slope":     ema_s,
            "hurst_rolling": hurst_r,
        },
        index=df.index,
    )

    # Drop NaN warm-up rows; drop any inf that can appear in extreme data
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)

    return features


def get_feature_names() -> list[str]:
    """Return the canonical ordered list of feature column names."""
    return list(FEATURE_COLS)
