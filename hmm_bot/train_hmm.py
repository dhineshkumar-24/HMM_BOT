import MetaTrader5 as mt5
import pandas as pd
from core.hmm_model import HMMRegimeDetector

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
BARS = 80000

print("Connecting to MT5...")

if not mt5.initialize():
    print("MT5 failed to initialize")
    quit()

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

print(f"Downloaded {len(df)} candles")

detector = HMMRegimeDetector()

print("Training HMM...")
detector.fit(df)

detector.save()

print("Model saved to models/hmm.pkl")

mt5.shutdown()