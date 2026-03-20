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
    "multi_lag_autocorr",   
    "atr_norm",
    "efficiency_ratio",     
    "variance_ratio",       
    "skewness",
    "kurtosis",
    "drawdown_pct",
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

def _efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Kaufman Efficiency Ratio.
    ER = net displacement / total path length over window bars.
    ER → 1.0: trending. ER → 0.0: mean-reverting/choppy.
    More powerful than autocorrelation for EURUSD M5.
    """
    net_move   = close.diff(window).abs()
    total_path = close.diff().abs().rolling(window).sum()
    return (net_move / total_path.replace(0, np.nan)).clip(0, 1)


def _variance_ratio(close: pd.Series, k: int = 5, window: int = 100) -> pd.Series:
    """
    Lo-MacKinlay Variance Ratio at horizon k.
    VR > 1: trending. VR = 1: random walk. VR < 1: mean-reverting.
    k=5 on M5 tests 25-minute price behavior.
    Centered around 1.0 then de-meaned for HMM feature use.
    """
    r1 = np.log(close / close.shift(1))
    rk = np.log(close / close.shift(k))
    var_1 = r1.rolling(window).var()
    var_k = rk.rolling(window).var()
    vr    = var_k / (k * var_1.replace(0, np.nan))
    return (vr - 1.0).clip(-1, 2)   # de-mean: 0 = random walk


def _multi_lag_autocorr(
    log_ret: pd.Series,
    lags: list = None,
    window: int = 60,
) -> pd.Series:
    """
    Composite autocorrelation: average over multiple lags.
    Reduces noise by ~50% vs single-lag while preserving signal.
    Lags [1, 3, 5, 10] cover tick noise, 15min, 25min, 50min patterns.
    """
    if lags is None:
        lags = [1, 3, 5, 10]
    lag_corrs = [
        log_ret.rolling(window).corr(log_ret.shift(lag))
        for lag in lags
    ]
    return pd.concat(lag_corrs, axis=1).mean(axis=1)
# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(df, vol_window=20, atr_period=14,
                          autocorr_lag=1, mom_period=10):
    df = df.copy()

    log_ret = _log_returns(df["close"])
    rvol    = _realized_volatility(log_ret, window=vol_window)
    vov     = _vol_of_vol(rvol, window=vol_window)
    mlac    = _multi_lag_autocorr(log_ret, window=vol_window * 3)
    atr_n   = _atr_normalized(df, period=atr_period)
    er      = _efficiency_ratio(df["close"], window=vol_window)
    vr      = _variance_ratio(df["close"], k=5, window=100)
    skew    = _skewness(log_ret, window=vol_window)
    kurt    = _kurtosis(log_ret, window=vol_window)
    dd_pct  = _drawdown_pct(df["close"], window=vol_window)
    ema_s   = _ema_slope(df["close"], span=50)
    hurst   = _hurst_rolling(df["close"], window=200)

    features = pd.DataFrame({
        "log_return":        log_ret,
        "realized_vol":      rvol,
        "vol_of_vol":        vov,
        "multi_lag_autocorr": mlac,
        "atr_norm":          atr_n,
        "efficiency_ratio":  er,
        "variance_ratio":    vr,
        "skewness":          skew,
        "kurtosis":          kurt,
        "drawdown_pct":      dd_pct,
        "ema_slope":         ema_s,
        "hurst_rolling":     hurst,
    }, index=df.index)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)
    return features

def get_intrabar_regime(df: pd.DataFrame) -> str:
    """
    Fast intrabar trend/MR classifier using ADX + Efficiency Ratio.
    Replaces HMM trend/MR routing — HMM only provides vol regime now.

    Returns:
        "trending"     — ADX > 22 AND ER > 0.35
        "mean_rev"     — ADX < 18 AND ER < 0.25
        "neutral"      — between thresholds, use vol regime default

    Called every bar from strategy_router.py before signal dispatch.
    """
    if len(df) < 30:
        return "neutral"

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ADX — fast period 10 for M5 intraday responsiveness
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up   = high - high.shift(1)
    down = low.shift(1) - low
    pdm  = up.where((up > down) & (up > 0), 0.0)
    mdm  = down.where((down > up) & (down > 0), 0.0)

    period = 10
    atr_s  = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    pdm_s  = pdm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    mdm_s  = mdm.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    pdi = 100 * (pdm_s / atr_s.replace(0, np.nan))
    mdi = 100 * (mdm_s / atr_s.replace(0, np.nan))
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # ER — 20-bar window
    net_move   = close.diff(20).abs()
    total_path = close.diff().abs().rolling(20).sum()
    er = (net_move / total_path.replace(0, np.nan)).clip(0, 1)

    last_adx = float(adx.iloc[-2]) if len(adx) >= 2 else 0.0
    last_er  = float(er.iloc[-2])  if len(er) >= 2 else 0.0

    if last_adx > 22 and last_er > 0.35:
        return "trending"
    elif last_adx < 18 and last_er < 0.25:
        return "mean_rev"
    else:
        return "neutral"

def get_feature_names() -> list[str]:
    """Return the canonical ordered list of feature column names."""
    return list(FEATURE_COLS)

# ─────────────────────────────────────────────────────────────────────────────
# Statistical Alpha Feature Matrix (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def build_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
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
    # VWAP-based Z-score: price vs its 20-bar mean, scaled by recent vol
    # Positive = price ABOVE mean = overextended UP = SELL candidate
    # Negative = price BELOW mean = overextended DOWN = BUY candidate
    roll_mean = close.rolling(20).mean()
    roll_std  = close.rolling(20).std().replace(0, np.nan)
    z_vwap    = (close - roll_mean) / roll_std          # standard z-score
    alpha_mr  = -z_vwap                                 # invert: positive = BUY
    # Scale: alpha_mr > 1.5 means price is 1.5 sigma below mean = strong BUY setup

    # ── MOMENTUM ALPHA ────────────────────────────────────────────────────────
    # Time-series momentum: cumulative 60-bar sign
    # Use sign of returns summed — avoids magnitude domination
    r5  = close.pct_change(5)
    r20 = close.pct_change(20)
    r60 = close.pct_change(60)
    # Normalize by volatility so it's comparable across calm and volatile periods
    alpha_mom = r60 / vol50.replace(0, np.nan)
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
    "r60":             r60,
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
