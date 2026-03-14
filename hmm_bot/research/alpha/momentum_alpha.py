import numpy as np
import pandas as pd

def time_series_momentum(log_ret: pd.Series, window: int = 20) -> pd.Series:
    """20-bar return sign persistence (is price consistently going one direction?)."""
    # Sum of signs: +20 means perfectly bullish, -20 means perfectly bearish
    return np.sign(log_ret).rolling(window=window).sum()

def ema_crossover_adx(close: pd.Series, adx: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    """Fast EMA crosses slow EMA only when ADX confirms trend."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    # Signal: 1 for bullish cross, -1 for bearish cross, 0 otherwise
    cross_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    cross_dn = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
    
    # Combine with ADX (say ADX > 25 indicates trend)
    adx_filter = adx > 25
    
    signal = pd.Series(0, index=close.index)
    signal[cross_up & adx_filter] = 1
    signal[cross_dn & adx_filter] = -1
    return signal

def breakout_signal(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Price closes above N-bar high with volume surge."""
    highest_high = close.rolling(window=window).max().shift(1)
    lowest_low = close.rolling(window=window).min().shift(1)
    vol_mean = volume.rolling(window=window).mean()
    
    signal = pd.Series(0, index=close.index)
    
    bull_break = (close > highest_high) & (volume > vol_mean)
    bear_break = (close < lowest_low) & (volume > vol_mean)
    
    signal[bull_break] = 1
    signal[bear_break] = -1
    return signal
