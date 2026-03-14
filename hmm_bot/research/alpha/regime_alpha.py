import pandas as pd
import numpy as np

def volatility_expansion(atr: pd.Series, window: int = 20) -> pd.Series:
    """When ATR crosses above its 20-period mean = regime change incoming."""
    atr_mean = atr.rolling(window=window).mean()
    cross_up = (atr > atr_mean) & (atr.shift(1) <= atr_mean.shift(1))
    return cross_up.astype(int)

def regime_stability_score(regime_series: pd.Series, decay_factor: float = 0.9) -> pd.Series:
    """
    Probability decay function based on how long we've been in the current regime.
    Assumes regime_series contains discrete integer states (0, 1, 2).
    Returns a score 1.0 -> 0.0 representing stability/confidence.
    """
    if len(regime_series) == 0:
        return pd.Series(dtype=float)
        
    current_regime = regime_series.iloc[0]
    duration = 0
    scores = []
    
    for state in regime_series:
        if state == current_regime:
            duration += 1
        else:
            current_regime = state
            duration = 1
            
        # Score decays the longer we stay in the same regime
        score = max(0.1, np.power(decay_factor, duration))
        scores.append(score)
        
    stability = pd.Series(scores, index=regime_series.index)
    return stability
