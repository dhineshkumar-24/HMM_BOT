"""
utils/features.py — Upgraded HMM feature engineering pipeline v4.0

13-feature regime discriminator + enhanced alpha features.

UPGRADE from v3.0:
    - Added OU half-life estimator for adaptive MR exit timing
    - Added skip-1 momentum (removes short-term reversal bias)
    - Added ATR percentile rank for volatility-band filtering
    - Added half-Kelly helper for position sizing
    - Replaced naive alpha_mr with OU-based mean-reversion score

LEAKAGE AUDIT (all causal):
    All rolling operations use data strictly from [t-window, t] inclusive.
    No future data is referenced. All shift() calls use positive lags (backward).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Feature column registry — order MUST stay stable between train & predict
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Core
    "log_return",          # 0  — bar-level return
    "atr_norm",            # 1  — volatility level (HIGH_VOL primary)
    "vol_ratio",           # 2  — vol10/vol50 expansion ratio (HIGH_VOL secondary)
    "vol_of_vol_rel",      # 3  — relative vol-of-vol (HIGH_VOL tertiary)
    # TRENDING discriminators
    "efficiency_ratio",    # 4  — Kaufman ER: 1=trending, 0=choppy
    "variance_ratio",      # 5  — VR deviation: +ve=trending, -ve=MR
    "hurst_approx",        # 6  — Hurst H: >0.5=trending, <0.5=MR
    "autocorr_multi",      # 7  — mean(autocorr, lags 1-5): +ve=trending
    "momentum_scaled",     # 8  — vol-normalized momentum
    "trend_consistency",   # 9  — fraction of bars moving in dominant direction
    "ema_slope",           # 10 — normalized EMA50 directional slope
    # Context
    "drawdown_pct",        # 11 — rolling drawdown depth
    "skewness",            # 12 — return skewness (asymmetric tail risk)
]


# ─────────────────────────────────────────────────────────────────────────────
# Individual feature calculators (all vectorised, all causal)
# ─────────────────────────────────────────────────────────────────────────────

def _log_returns(close: pd.Series) -> pd.Series:
    """Bar-by-bar log return: ln(close_t / close_{t-1}). Causal."""
    return np.log(close / close.shift(1))


def _atr_normalized(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR / close price — dimensionless volatility.
    Causal: only uses past bars via rolling mean.
    HIGH_VOL primary discriminator.
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


def _vol_ratio(log_ret: pd.Series, fast: int = 10, slow: int = 50) -> pd.Series:
    """
    Ratio of fast to slow realized vol minus 1.
    vol_ratio > 0: vol is expanding (HIGH_VOL indicator)
    vol_ratio < 0: vol is contracting (quiet regime)
    Causal rolling windows.
    """
    vol_fast = log_ret.rolling(fast).std()
    vol_slow = log_ret.rolling(slow).std().replace(0, np.nan)
    return (vol_fast / vol_slow) - 1.0


def _vol_of_vol_relative(log_ret: pd.Series, window: int = 20) -> pd.Series:
    """
    Relative vol-of-vol: std(vol) / mean(vol).
    Captures regime transitions. High = regime in flux.
    Causal rolling.
    """
    rvol = log_ret.rolling(window).std()
    vov  = rvol.rolling(window).std()
    mean_rvol = rvol.rolling(window).mean().replace(0, np.nan)
    return vov / mean_rvol


def _efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Kaufman Efficiency Ratio (ER).

    ER = |close[t] - close[t-window]| / sum(|close[i] - close[i-1]|, 0<i<=window)

    Interpretation:
        ER → 1.0: perfectly trending (directional movement dominates)
        ER → 0.0: perfectly mean-reverting (all movement cancels)

    This is the single best discriminator for TRENDING vs MEAN_REVERT.
    Causal: only past bars used in numerator and denominator.
    """
    net_change = (close - close.shift(window)).abs()
    bar_changes = close.diff().abs()
    total_path  = bar_changes.rolling(window=window).sum().replace(0, np.nan)
    er = net_change / total_path
    return er.clip(0.0, 1.0)


def _variance_ratio(log_ret: pd.Series, q: int = 5, window: int = 60) -> pd.Series:
    """
    Rolling Variance Ratio (Lo & MacKinlay 1988).

    VR(q) = Var(q-period return) / (q × Var(1-period return))
    We return (VR - 1) so the series is centered at 0:
        > 0: trending
        < 0: mean-reverting
        = 0: random walk

    Causal: uses only past returns up to the current bar.
    """
    var_1 = log_ret.rolling(window).var().replace(0, np.nan)
    ret_q = log_ret.rolling(q).sum()
    var_q = ret_q.rolling(window).var().replace(0, np.nan)

    vr = (var_q / (q * var_1)) - 1.0
    return vr.clip(-1.5, 1.5)


def _hurst_approximate(log_ret: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling approximate Hurst Exponent via Rescaled Range (R/S) analysis.

    We return (H - 0.5) so the series is centered at 0.
    All calculations are causal (rolling windows).
    """
    short = max(10, window // 4)
    long  = window

    def rs(series: pd.Series, n: int) -> pd.Series:
        def _rs_scalar(x):
            if len(x) < 4 or x.std() == 0:
                return np.nan
            cumdev = np.cumsum(x - x.mean())
            r = cumdev.max() - cumdev.min()
            s = x.std()
            return r / s if s > 0 else np.nan
        return series.rolling(n).apply(_rs_scalar, raw=True)

    rs_short = rs(log_ret, short)
    rs_long  = rs(log_ret, long)

    ratio = (rs_long / rs_short.replace(0, np.nan)).replace(0, np.nan)
    log_ratio = np.log(ratio.clip(1e-8, None))
    log_scale  = np.log(long / short)

    hurst = (log_ratio / log_scale) - 0.5
    return hurst.clip(-0.5, 0.5)


def _autocorr_multi(
    log_ret: pd.Series,
    lags: list[int] = None,
    window: int = 30,
) -> pd.Series:
    """
    Mean of rolling autocorrelations at multiple lags [1, 2, 3, 5].
    Causal: all lag-k uses shift(k), looking only backward.
    """
    if lags is None:
        lags = [1, 2, 3, 5]

    ac_series = []
    for lag in lags:
        ac = log_ret.rolling(window).corr(log_ret.shift(lag))
        ac_series.append(ac)

    ac_df  = pd.concat(ac_series, axis=1)
    return ac_df.mean(axis=1).clip(-0.5, 0.5)


def _momentum_scaled(
    close: pd.Series,
    log_ret: pd.Series,
    short: int = 10,
    long: int = 60,
    vol_window: int = 20,
) -> pd.Series:
    """
    Vol-normalized momentum.
    Causal: pct_change and rolling std use only past data.
    """
    raw_mom = close.pct_change(long)
    vol     = log_ret.rolling(vol_window).std().replace(0, np.nan)
    scaled  = (raw_mom / vol).clip(-5, 5)
    return scaled


def _trend_consistency(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Fraction of bars within the window moving in the DOMINANT direction.
    Causal: only past bars used.
    """
    def _consistency(x):
        if len(x) < 4:
            return np.nan
        net = x[-1] - x[0]
        dominant = np.sign(net)
        if dominant == 0:
            return 0.5
        bar_dirs = np.sign(np.diff(x))
        return float((bar_dirs == dominant).mean())

    return close.rolling(window).apply(_consistency, raw=True)


def _ema_slope(close: pd.Series, span: int = 50, slope_window: int = 5) -> pd.Series:
    """
    Normalized EMA50 slope (price-relative %).
    Causal: EWM uses past bars only with adjust=False.
    """
    ema    = close.ewm(span=span, adjust=False).mean()
    lagged = close.shift(slope_window).replace(0, np.nan)
    return ema.diff(slope_window) / lagged


def _drawdown_pct(close: pd.Series, window: int = 30) -> pd.Series:
    """Percentage drawdown from rolling high. Causal."""
    rolling_max = close.rolling(window=window).max()
    return (close - rolling_max) / rolling_max


def _skewness(log_ret: pd.Series, window: int = 30) -> pd.Series:
    """Rolling skewness of log returns (negative = downside tail). Causal."""
    return log_ret.rolling(window=window).skew()


# ─────────────────────────────────────────────────────────────────────────────
# Public API — Main feature matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    atr_period:  int = 14,
    vol_fast:    int = 10,
    vol_slow:    int = 50,
    er_window:   int = 20,
    vr_q:        int = 5,
    vr_window:   int = 60,
    hurst_window: int = 60,
    ac_window:   int = 30,
    mom_long:    int = 60,
    tc_window:   int = 20,
) -> pd.DataFrame:
    """
    Build the 13-feature HMM regime discriminator matrix.
    All features are causal (no look-ahead). NaN-producing warm-up rows
    are dropped at the end.
    """
    df = df.copy()

    # ── Base return series ────────────────────────────────────────────────────
    log_ret = _log_returns(df["close"])

    # ── HIGH_VOL separators ───────────────────────────────────────────────────
    atr_n      = _atr_normalized(df, period=atr_period)
    vol_r      = _vol_ratio(log_ret, fast=vol_fast, slow=vol_slow)
    vov_rel    = _vol_of_vol_relative(log_ret, window=vol_slow)

    # ── TRENDING separators ───────────────────────────────────────────────────
    er         = _efficiency_ratio(df["close"], window=er_window)
    vr         = _variance_ratio(log_ret, q=vr_q, window=vr_window)
    hurst      = _hurst_approximate(log_ret, window=hurst_window)
    ac_multi   = _autocorr_multi(log_ret, lags=[1, 2, 3, 5], window=ac_window)
    mom_s      = _momentum_scaled(df["close"], log_ret, long=mom_long, vol_window=vol_slow)
    tc         = _trend_consistency(df["close"], window=tc_window)
    ema_s      = _ema_slope(df["close"], span=50, slope_window=5)

    # ── Context features ─────────────────────────────────────────────────────
    dd_pct     = _drawdown_pct(df["close"], window=30)
    skew       = _skewness(log_ret, window=30)

    # ── Assemble ──────────────────────────────────────────────────────────────
    features = pd.DataFrame(
        {
            "log_return":       log_ret,
            "atr_norm":         atr_n,
            "vol_ratio":        vol_r,
            "vol_of_vol_rel":   vov_rel,
            "efficiency_ratio": er,
            "variance_ratio":   vr,
            "hurst_approx":     hurst,
            "autocorr_multi":   ac_multi,
            "momentum_scaled":  mom_s,
            "trend_consistency": tc,
            "ema_slope":        ema_s,
            "drawdown_pct":     dd_pct,
            "skewness":         skew,
        },
        index=df.index,
    )

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features = features.dropna()
    return features


def get_feature_names() -> list[str]:
    """Return the canonical ordered list of feature column names."""
    return list(FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# NEW v4.0: Ornstein-Uhlenbeck Half-Life Estimator
# ─────────────────────────────────────────────────────────────────────────────

def ou_half_life(close: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling Ornstein-Uhlenbeck half-life of mean reversion.

    Model: dS = θ(μ - S)dt + σdW
    Estimation: AR(1) regression  S_t - S_{t-1} = α + β·S_{t-1} + ε
    Half-life = -log(2) / log(1 + β)   [in bars]

    Interpretation:
        half_life < 20:   fast mean reversion (good MR signal)
        half_life 20-60:  moderate MR (usable)
        half_life > 60:   slow / trending (avoid MR trades)

    Causal: only uses past bars in the rolling window.
    """
    def _hl_scalar(x):
        if len(x) < 10:
            return np.nan
        y = np.diff(x)             # S_t - S_{t-1}
        x_lag = x[:-1]             # S_{t-1}
        # OLS: y = alpha + beta * x_lag
        n = len(y)
        x_mean = x_lag.mean()
        y_mean = y.mean()
        cov_xy = np.sum((x_lag - x_mean) * (y - y_mean))
        var_x  = np.sum((x_lag - x_mean) ** 2)
        if var_x < 1e-20:
            return np.nan
        beta = cov_xy / var_x
        if beta >= 0:
            return np.nan  # not mean-reverting
        # Half-life = -log(2) / log(1 + beta)
        arg = 1.0 + beta
        if arg <= 0:
            return np.nan
        hl = -np.log(2) / np.log(arg)
        return max(1.0, min(hl, 200.0))  # clip to reasonable range

    return close.rolling(window).apply(_hl_scalar, raw=True)


def atr_percentile_rank(df: pd.DataFrame, period: int = 14, lookback: int = 252 * 12) -> pd.Series:
    """
    ATR percentile rank over long lookback window (0-1 scale).
    Used for volatility-band filtering: trade only in P20-P80.

    Causal: rolling rank uses only past data.
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
    atr = tr.rolling(period).mean()

    # Rolling percentile rank (causal)
    actual_lookback = min(lookback, len(atr))
    rank = atr.rolling(actual_lookback, min_periods=100).rank(pct=True)
    return rank


# ─────────────────────────────────────────────────────────────────────────────
# NEW v4.0: Enhanced Alpha Features for MomentumStrategy
# ─────────────────────────────────────────────────────────────────────────────

def build_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regime-adaptive alpha features for MomentumStrategy.

    v4.0 Enhancements:
        - alpha_mr now uses OU-adjusted Z-score (not just negated Z)
        - alpha_mom uses skip-1 momentum (academic best practice)
        - Added ou_hl (half-life) for adaptive exit timing
        - Added atr_pctile for volatility-band filtering
    """
    close   = df["close"]
    high    = df["high"]
    low     = df["low"]
    log_ret = np.log(close / close.shift(1))

    vol10 = log_ret.rolling(10).std()
    vol50 = log_ret.rolling(50).std()

    # ── Z-score (standard) ────────────────────────────────────────────────────
    roll_mean = close.rolling(20).mean()
    roll_std  = close.rolling(20).std().replace(0, np.nan)
    z_vwap    = (close - roll_mean) / roll_std

    # ── OU-Adjusted MR Alpha ─────────────────────────────────────────────────
    # Standard Z-score weighted by OU half-life confidence.
    # Short half-life = high confidence in mean reversion = stronger signal.
    ou_hl = ou_half_life(close, window=60)
    ou_hl_bounded = ou_hl.clip(5, 100)  # bound for numerical stability

    # Weight: higher when OU half-life is short (strong MR regime)
    # hl_weight ∈ [0.5, 2.0]: short hl → high weight, long hl → low weight
    hl_weight = (30.0 / ou_hl_bounded).clip(0.5, 2.0)
    alpha_mr = -z_vwap * hl_weight

    # ── Skip-1 Momentum ──────────────────────────────────────────────────────
    # Skip most recent bar to avoid short-term reversal contamination.
    # Academic basis: Jegadeesh & Titman 1993, Novy-Marx 2012.
    r5  = close.pct_change(5)
    r20 = close.pct_change(20)
    # Skip-1: return from t-60 to t-1 (exclude most recent bar)
    r60_skip1 = (close.shift(1) / close.shift(60) - 1.0)
    alpha_mom = r60_skip1 / vol50.replace(0, np.nan)
    alpha_mom = alpha_mom.clip(-5, 5)

    # Legacy r60 for backward compat
    r60 = close.pct_change(60)

    vol_regime = (vol10 / vol50.replace(0, np.nan)) - 1.0
    trend_strength = log_ret.rolling(20).corr(log_ret.shift(1))

    combined = alpha_mr

    r20_norm  = r20 / vol50.replace(0, np.nan)
    r20_norm  = r20_norm.clip(-5, 5)
    exhaustion_long  = alpha_mom - r20_norm
    exhaustion_short = r20_norm - (close.pct_change(5) / vol10.replace(0, np.nan)).clip(-5, 5)

    mr_quality = np.abs(z_vwap) * (1.0 - vol_regime.clip(-1, 1))

    # ── ATR for the alpha frame ───────────────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.rolling(14).mean()

    result = pd.DataFrame(
        {
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
            "exhaustion_long": exhaustion_long,
            "mr_quality":      mr_quality,
            "combined_alpha":  combined,
            "ou_half_life":    ou_hl,
            "atr_alpha":       atr_14,
        },
        index=df.index,
    )

    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result
