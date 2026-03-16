import sys
import os
import yaml
import pandas as pd
from hmm_bot.research.backtester import run_backtest
from hmm_bot.strategy.strategy_router import StrategyRouter
import hmm_bot.utils.logger as log

def main():
    print("Loading config...")
    with open("hmm_bot/config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Loading dummy data...")
    # Create simple dummy EURUSD data
    dates = pd.date_range("2023-01-01", periods=1000, freq="1Min")
    df = pd.DataFrame({
        "time": dates,
        "open": [1.1000] * 1000,
        "high": [1.1020] * 1000,
        "low": [1.0980] * 1000,
        "close": [1.1005] * 1000,
        "tick_volume": [100] * 1000
    })
    
    # Introduce some variation
    import numpy as np
    np.random.seed(42)
    df["close"] = 1.1000 + np.random.randn(1000).cumsum() * 0.0001
    df["high"] = df["close"] + 0.0005
    df["low"] = df["close"] - 0.0005
    df["open"] = df["close"].shift(1).fillna(1.1000)

    print("Running backtest...")
    from hmm_bot.core.hmm_model import HMMRegimeDetector
    # hmm = HMMRegimeDetector(n_train_bars=500, n_pred_bars=100)
    # hmm.fit(df) # Skip hmm training for speed

    res = run_backtest(df=df, config=config, hmm=None, verbose=True)
    print(f"Total trades: {len(res.trades)}")

if __name__ == "__main__":
    main()
