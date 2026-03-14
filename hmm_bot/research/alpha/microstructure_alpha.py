import pandas as pd
import numpy as np

def volume_shock(volume: pd.Series, window: int = 20, threshold: float = 2.5) -> pd.Series:
    """Tick volume > 2.5 standard deviations above rolling mean."""
    vol_mean = volume.rolling(window=window).mean()
    vol_std = volume.rolling(window=window).std()
    z_score = (volume - vol_mean) / vol_std
    
    # Return binary or continuous score
    return (z_score > threshold).astype(int)

def volatility_clustering(realized_vol: pd.Series) -> pd.Series:
    """Realized vol today vs realized vol yesterday ratio."""
    return realized_vol / realized_vol.shift(1)

def intraday_reversal(df: pd.DataFrame) -> pd.Series:
    """When a large candle body is followed by a doji = exhaustion."""
    # Body size relative to range
    body = (df['close'] - df['open']).abs()
    hl_range = df['high'] - df['low']
    
    body_pct = body / hl_range.replace(0, np.nan)
    
    # Large body defined as > 60% of range
    is_large_body = body_pct > 0.6
    
    # Doji defined as body < 10% of range
    is_doji = body_pct < 0.1
    
    # Reversal signal: Large up -> Doji => Bearish (-1)
    # Large down -> Doji => Bullish (1)
    large_up = is_large_body & (df['close'] > df['open'])
    large_down = is_large_body & (df['close'] < df['open'])
    
    signal = pd.Series(0, index=df.index)
    signal[large_up.shift(1) & is_doji] = -1
    signal[large_down.shift(1) & is_doji] = 1
    
    return signal
