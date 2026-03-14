import numpy as np
import pandas as pd
import scipy.stats as stats

def information_coefficient(signal_series: pd.Series, forward_return_series: pd.Series) -> float:
    """
    Spearman rank correlation between signal and next-bar return.
    IC > 0.05 is considered useful in practice.
    """
    # Align indices and drop NaNs
    df = pd.concat([signal_series, forward_return_series], axis=1).dropna()
    if len(df) < 2:
        return 0.0
        
    ic, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return float(ic)

def t_statistic(signal_series: pd.Series, forward_return_series: pd.Series) -> float:
    """
    Is the IC statistically significant? t > 2.0 = valid.
    """
    ic = information_coefficient(signal_series, forward_return_series)
    df_aligned = pd.concat([signal_series, forward_return_series], axis=1).dropna()
    n = len(df_aligned)
    
    # Avoid div by zero or math domain errors
    if n <= 2 or abs(ic) == 1.0:
        return 0.0
        
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2)
    return float(t_stat)

def signal_decay(signal_series: pd.Series, forward_returns: pd.Series, max_lag: int = 10) -> list:
    """
    How many bars does the signal stay predictive?
    Tells you the ideal holding period.
    Returns a list of IC values from lag 1 to max_lag.
    """
    decay_ic = []
    
    # Lag 1 means IC between signal[t] and return[t+1]
    # Lag 2 means IC between signal[t] and return[t+2]
    # forward_returns usually represents t+1, so we shift it backwards to get future returns
    for lag in range(0, max_lag):
        # Shift forward_returns backward by lag (negative shift puts future values in current row)
        future_ret = forward_returns.shift(-lag)
        ic = information_coefficient(signal_series, future_ret)
        decay_ic.append(ic)
        
    return decay_ic

def sharpe_of_signal(signal_series: pd.Series, forward_returns: pd.Series) -> float:
    """
    Signal-level Sharpe (before transaction costs).
    Assume 15-minute bars -> roughly 96 bars a day -> 24192 bars a year.
    Returns the annualized Sharpe Ratio.
    """
    # Simple strategy: position size is exactly the signal value at time t
    strategy_returns = (signal_series * forward_returns).dropna()
    
    if len(strategy_returns) < 2 or strategy_returns.std() == 0:
        return 0.0
        
    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()
    
    # Annualizing assuming 15M bars
    annualizer = np.sqrt(24192)
    return float((mean_ret / std_ret) * annualizer)
