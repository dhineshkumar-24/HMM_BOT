import sys
import yaml
import pandas as pd
from hmm_bot.strategy.strategy_router import StrategyRouter

def main():
    with open("hmm_bot/config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Disable session filter for testing
    config["sessions"] = {
        "asian_start": "00:00",
        "asian_end": "23:59",
        "london_start": "00:00",
        "london_end": "23:59",
        "newyork_start": "00:00",
        "newyork_end": "23:59",
    }

    dates = pd.date_range("2023-01-01", periods=2000, freq="1Min")
    df = pd.DataFrame({
        "time": dates,
        "open": [1.1000] * 2000,
        "high": [1.1020] * 2000,
        "low": [1.0980] * 2000,
        "close": [1.1005] * 2000,
        "tick_volume": [100] * 2000
    })
    
    import numpy as np
    np.random.seed(42)
    df["close"] = 1.1000 + np.random.randn(2000).cumsum() * 0.0001
    df["high"] = df["close"] + 0.0005
    df["low"] = df["close"] - 0.0005
    df["open"] = df["close"].shift(1).fillna(1.1000)

    router = StrategyRouter(config)
    df = router.calculate_indicators(df)
    
    signals = 0
    combiner_directions = 0
    momentum_signals = 0
    mean_rev_signals = 0

    from hmm_bot.core.hmm_model import REGIME_MEAN_REVERT, REGIME_TRENDING
    
    for i in range(200, len(df)):
        window = df.iloc[: i + 1]
        candle_time = window.iloc[-1]["time"]
        
        # Test mean revert regime
        signal = router.route(window, candle_time=candle_time, regime=REGIME_MEAN_REVERT)
        if signal: signals += 1

        # Test combiner manually
        combined = router._combiner.combine(window, regime=REGIME_MEAN_REVERT)
        if combined.get("direction"): combiner_directions += 1

        # Test strategies manually
        mr = router.mean_reversion.generate_signal(window, regime=REGIME_MEAN_REVERT, session="asian")
        if mr: mean_rev_signals += 1

        mo = router.momentum.generate_signal(window, regime=REGIME_TRENDING, session="london")
        if mo: momentum_signals += 1

    print(f"Total Router Signals: {signals}")
    print(f"Combiner Directions: {combiner_directions}")
    print(f"Mean Reversion Signals: {mean_rev_signals}")
    print(f"Momentum Signals: {momentum_signals}")

if __name__ == "__main__":
    main()
