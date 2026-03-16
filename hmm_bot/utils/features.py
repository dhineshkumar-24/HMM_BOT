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
    "momentum",
    "skewness",
    "kurtosis",
    "drawdown_pct",
    "ema_slope",
    #"hurst_rolling",
    #"volume_zscore",
    #"volume_trend",
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

def build_feature_matrix(df, vol_window=20, atr_period=14,
                          autocorr_lag=1, mom_period=10):
    df = df.copy()

    log_ret = _log_returns(df["close"])
    rvol    = _realized_volatility(log_ret, window=vol_window)
    vov     = _vol_of_vol(rvol, window=vol_window)
    ac      = _autocorrelation(log_ret, lag=autocorr_lag, window=vol_window)
    atr_n   = _atr_normalized(df, period=atr_period)
    mom     = _momentum(df["close"], period=mom_period)
    skew    = _skewness(log_ret, window=vol_window)
    kurt    = _kurtosis(log_ret, window=vol_window)
    dd_pct  = _drawdown_pct(df["close"], window=vol_window)
    ema_s   = _ema_slope(df["close"], span=50)

    features = pd.DataFrame({
        "log_return":   log_ret,
        "realized_vol": rvol,
        "vol_of_vol":   vov,
        "autocorr":     ac,
        "atr_norm":     atr_n,
        "momentum":     mom,
        "skewness":     skew,
        "kurtosis":     kurt,
        "drawdown_pct": dd_pct,
        "ema_slope":    ema_s,
    }, index=df.index)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)
    return features


def get_feature_names() -> list[str]:
    """Return the canonical ordered list of feature column names."""
    return list(FEATURE_COLS)

# ─────────────────────────────────────────────────────────────────────────────
# Statistical Alpha Feature Matrix (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def build_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Statistical alpha features tuned for M1 EURUSD mean-reverting regime.

    Key insight: On M1, 20-bar momentum picks up EXHAUSTED moves.
    Use r60 (1-hour) for momentum — genuine directional flow.
    Weight mean reversion highest — dominant M1 effect.
    """
    close   = df["close"]
    log_ret = np.log(close / close.shift(1))

    # ── Returns ───────────────────────────────────────────────────────────────
    r5  = close.pct_change(5)    # 5-min pullback detector
    r60 = close.pct_change(60)   # 1-hour trend (was r20 — too short)

    # ── Volatility horizons ───────────────────────────────────────────────────
    vol10 = log_ret.rolling(10).std()
    vol50 = log_ret.rolling(50).std()

    # ── Structural features ───────────────────────────────────────────────────
    trend_strength = log_ret.rolling(20).corr(log_ret.shift(1))
    vol_regime     = vol10 / vol50.replace(0, np.nan)

    # ── Three alpha signals ───────────────────────────────────────────────────
    # Alpha 1: SHORT-TERM MEAN REVERSION (dominant M1 effect)
    # Fades overextended 5-bar moves back to mean
    # Positive alpha_mr = price pulled back = expect bounce
    alpha_mr  = -(r5 / vol10.replace(0, np.nan))

    # Alpha 2: MEDIUM-TERM MOMENTUM (1-hour directional flow)
    # r60 captures genuine order flow, not noise
    # Positive alpha_mom = 1-hour uptrend = buy with trend
    alpha_mom = r60 / vol50.replace(0, np.nan)

    # Alpha 3: VOLATILITY EXPANSION (breakout / clustering)
    # vol10 > vol50 means volatility expanding = directional move
    alpha_vol = vol_regime - 1.0

    # ── Combined alpha — raw values, no z-score normalization ─────────────────
    # Raw alpha units: alpha_mr ≈ ±1 to ±5, alpha_mom ≈ ±0.5 to ±3
    # Z-score normalization was destroying signal in calm markets
    mom_agree = np.sign(alpha_mr) * np.sign(alpha_mom)  # +1 if agree, -1 if conflict
    combined   = alpha_mr * (1.0 + 0.20 * mom_agree.clip(-1, 1))


    result = pd.DataFrame({
        "r5":             r5,
        "r60":            r60,
        "vol10":          vol10,
        "vol50":          vol50,
        "trend_strength": trend_strength,
        "vol_regime":     vol_regime,
        "alpha_mr":       alpha_mr,
        "alpha_mom":      alpha_mom,
        "alpha_vol":      alpha_vol,
        "combined_alpha": combined,
    }, index=df.index)

    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result
