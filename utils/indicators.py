import numpy as np
import pandas as pd

def compute_vwap(df, window=20):
    """Calculates Rolling Volume Weighted Average Price."""
    v = df['tick_volume']
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # Rolling VWAP: Sum(TP*V, window) / Sum(V, window)
    pv = tp * v
    return pv.rolling(window=window).sum() / v.rolling(window=window).sum()

def compute_zscore(series, window=20):
    """Calculates Z-Score relative to rolling mean and std."""
    r_mean = series.rolling(window=window).mean()
    r_std = series.rolling(window=window).std()
    return (series - r_mean) / r_std

def compute_volatility(returns, window=20):
    """Calculates rolling volatility (standard deviation of returns)."""
    return returns.rolling(window=window).std()

def compute_volatility_slope(volatility_series, window=5):
    """Calculates the slope of the volatility line."""
    return volatility_series.diff(window).fillna(0)

def compute_autocorrelation(returns, lag=1):
    """Calculates rolling autocorrelation."""
    return returns.rolling(window=20).corr(returns.shift(lag))

def compute_hurst(series, lags=[2, 20]):
    """
    Calculates the Hurst Exponent to determine market regime.
    H < 0.5: Mean Reverting
    H ~ 0.5: Random Walk
    H > 0.5: Trending
    
    Using a simplified RS Analysis or variance ratio approximation for speed.
    """
    # Simplified Rolling Hurst (Variance Ratio method for efficiency)
    # H = log(RS) / log(n) approx
    
    # We will use a trusted scalar implementation that can be rolled
    # For a rolling pandas series, this is slow.
    # Optimization: Function to be applied on rolling window.
    
    # Placeholder for efficient implementation:
    # Use standard deviation of differences vs standard deviation of series
    
    # Efficient approximation:
    # H ~ 0.5 is random.
    # This function expects a Series and returns a Series (Rolling)
    
    def get_hurst_scalar(ts):
        if len(ts) < 20: return 0.5
        lags_range = range(2, 20)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags_range]
        if min(tau) == 0: return 0.5 # Prevent log(0)
        
        # polyfit(log(lags), log(tau), 1)[0] returns H
        try:
            m = np.polyfit(np.log(lags_range), np.log(tau), 1)
            return m[0]
        except:
            return 0.5

    # Apply rolling (Slow but accurate enough for M5)
    return series.rolling(100).apply(get_hurst_scalar, raw=True)

def compute_rsi(series, period=14):
    """
    Calculates RSI (Relative Strength Index).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Wilder's Smoothing
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_wick_body_ratio(df):
    body = (df['close'] - df['open']).abs()
    range_total = df['high'] - df['low']
    ratio = (range_total - body) / body.replace(0, 0.00001) 
    return ratio
