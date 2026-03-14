import pandas as pd
import numpy as np

def volatility_adjusted_zscore(close: pd.Series, atr: pd.Series, window: int = 20) -> pd.Series:
    """Z-score divided by rolling ATR (auto-scales with volatility)."""
    roll_mean = close.rolling(window=window).mean()
    roll_std = close.rolling(window=window).std()
    z_score = (close - roll_mean) / roll_std
    return z_score / atr

def liquidity_shock_reversal(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Spike in volume + price deviation = mean reversion setup."""
    # Volume spike > 2 std
    vol_mean = volume.rolling(window).mean()
    vol_std = volume.rolling(window).std()
    vol_z = (volume - vol_mean) / vol_std
    
    # Price deviation
    price_ret = close.pct_change()
    price_z = (price_ret - price_ret.rolling(window).mean()) / price_ret.rolling(window).std()
    
    # Combined score 
    # (buy if price drops hard on high volume, sell if price spikes hard on high volume)
    return -price_z * vol_z

def vwap_spread_dislocation(close: pd.Series, vwap: pd.Series, atr: pd.Series) -> pd.Series:
    """Price distance from VWAP measured in ATR units."""
    return (close - vwap) / atr
