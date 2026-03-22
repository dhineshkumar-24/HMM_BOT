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

def build_alpha_features(df: pd.DataFrame, vwap_window: int = 20, mom_period: int = 60) -> pd.DataFrame:
    """
    Regime-adaptive alpha features.
    Mean reversion alpha:  fade short-term overextension vs VWAP
    Momentum alpha:        ride directional 1-hour order flow
    Vol regime:            classify current volatility environment
    """
    close   = df["close"]
    high    = df["high"]
    low     = df["low"]
    log_ret = np.log(close / close.shift(1))

    # ── Vol horizons ──────────────────────────────────────────────────────────
    vol10 = log_ret.rolling(10).std()
    vol50 = log_ret.rolling(50).std()

    # ── MEAN REVERSION ALPHA ──────────────────────────────────────────────────
    # VWAP-based Z-score: price vs its N-bar mean, scaled by recent vol
    # Positive = price ABOVE mean = overextended UP = SELL candidate
    # Negative = price BELOW mean = overextended DOWN = BUY candidate
    roll_mean = close.rolling(vwap_window).mean()
    roll_std  = close.rolling(vwap_window).std().replace(0, np.nan)
    z_vwap    = (close - roll_mean) / roll_std          # standard z-score
    alpha_mr  = -z_vwap                                 # invert: positive = BUY
    # Scale: alpha_mr > 1.5 means price is 1.5 sigma below mean = strong BUY setup

    # ── MOMENTUM ALPHA ────────────────────────────────────────────────────────
    # Time-series momentum: cumulative N-bar sign
    # Use sign of returns summed — avoids magnitude domination
    r5  = close.pct_change(5)
    r20 = close.pct_change(20)
    r_mom = close.pct_change(mom_period)
    # Normalize by volatility so it's comparable across calm and volatile periods
    alpha_mom = r_mom / vol50.replace(0, np.nan)
    # Clip to ±5 to prevent extreme momentum values from dominating
    alpha_mom = alpha_mom.clip(-5, 5)

    # ── VOL REGIME ────────────────────────────────────────────────────────────
    vol_regime = (vol10 / vol50.replace(0, np.nan)) - 1.0  # >0 = expanding vol

    # ── TREND STRENGTH ────────────────────────────────────────────────────────
    # Rolling autocorrelation: positive = trending, negative = mean-reverting
    trend_strength = log_ret.rolling(20).corr(log_ret.shift(1))

    # ── COMBINED — not used directly, regime routing selects which alpha ──────
    # This is a fallback composite only used if regime is unknown
    combined = alpha_mr  # default to mean reversion

    # ── EXHAUSTION DETECTOR ───────────────────────────────────────────────────
    # Problem: a large 600-pip move looks like "momentum" but may be exhausted
    # Solution: compare RECENT momentum to LONGER-TERM momentum
    # If recent (20-bar) has slowed vs long-term (60-bar) → exhaustion signal

    r20_norm  = r20 / vol50.replace(0, np.nan)
    r20_norm  = r20_norm.clip(-5, 5)

    # Exhaustion = recent momentum has decelerated vs 60-bar trend
    # Positive exhaustion_long = was going up, slowing down = sell setup
    exhaustion_long  = alpha_mom - r20_norm   # > 0 means trend slowing (was up, now flat)
    exhaustion_short = r20_norm - (close.pct_change(5) / vol10.replace(0, np.nan)).clip(-5,5)

    # ── MEAN REVERSION QUALITY ────────────────────────────────────────────────
    # How far is price from equilibrium RELATIVE to current vol?
    # This is the purest mean reversion entry quality score
    # |z_vwap| > 2 AND vol_regime < 0 (shrinking vol) = ideal MR setup
    mr_quality = np.abs(z_vwap) * (1.0 - vol_regime.clip(-1, 1))
    # mr_quality > 2.0 = high quality mean reversion setup

    result = pd.DataFrame({
    "r5":              r5,
    "r20":             r20,
    "r60":             r_mom,
    "vol10":           vol10,
    "vol50":           vol50,
    "z_vwap":          z_vwap,
    "trend_strength":  trend_strength,
    "vol_regime":      vol_regime,
    "alpha_mr":        alpha_mr,
    "alpha_mom":       alpha_mom,
    "exhaustion_long": exhaustion_long,   # NEW
    "mr_quality":      mr_quality,        # NEW
    "combined_alpha":  combined,
    }, index=df.index)

    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result
