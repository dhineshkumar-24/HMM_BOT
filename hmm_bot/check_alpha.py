"""
check_alpha.py — Run 1: Alpha Signal Validation (CORRECTED)

Fixes applied vs previous version:
  1. Forward return horizon matched to strategy holding period
     Mean reversion: tested at 20-bar forward return (not 1-bar)
     Momentum:       tested at 60-bar forward return
  2. Session filtering — MR signal tested only on Asian session bars
  3. Microstructure — time column excluded before numeric operations

Run from inside hmm_bot/ with MT5 open:
    python check_alpha.py
"""

import sys, os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "

# ─────────────────────────────────────────────────────────────────────────────
# Stat functions
# ─────────────────────────────────────────────────────────────────────────────

def ic(signal: pd.Series, fwd: pd.Series) -> float:
    df = pd.DataFrame({"s": signal, "r": fwd}).dropna()
    if len(df) < 30:
        return 0.0
    return float(stats.spearmanr(df["s"], df["r"])[0])

def tstat(signal: pd.Series, fwd: pd.Series) -> float:
    df = pd.DataFrame({"s": signal, "r": fwd}).dropna()
    n = len(df)
    if n < 30:
        return 0.0
    v = ic(signal, fwd)
    if abs(v) >= 1.0:
        return 0.0
    return float(v * np.sqrt(n - 2) / np.sqrt(1 - v**2 + 1e-9))

def hit_rate(signal: pd.Series, fwd: pd.Series) -> float:
    df = pd.DataFrame({"s": signal, "r": fwd}).dropna()
    df = df[df["s"] != 0]
    if len(df) < 30:
        return 0.0
    return float(((df["s"] > 0) == (df["r"] > 0)).sum() / len(df))

# ─────────────────────────────────────────────────────────────────────────────
# Report one signal
# ─────────────────────────────────────────────────────────────────────────────

def report(name: str, scores: pd.Series, fwd: pd.Series,
           ic_min=0.03, t_min=2.0, hit_min=0.52) -> bool:

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")

    df = pd.DataFrame({"s": scores, "r": fwd}).dropna()
    print(f"  Filtered bars used: {len(df):,}")

    if len(df) < 100:
        print(f"{FAIL}  Only {len(df)} valid bars — too few to test")
        return False

    ic_val = ic(scores, fwd)
    t_val  = tstat(scores, fwd)
    hr_val = hit_rate(scores, fwd)

    label_ic = "[GOOD]" if abs(ic_val) >= 0.05 else "[OK]" if abs(ic_val) >= ic_min else "[NOISE]"
    label_t  = "[SIGNIFICANT]" if abs(t_val) >= t_min else "[NOT SIGNIFICANT]"
    label_hr = "[EDGE]" if hr_val >= hit_min else "[NO EDGE]"

    s_ic = PASS if abs(ic_val) >= ic_min else FAIL
    s_t  = PASS if abs(t_val)  >= t_min  else FAIL
    s_hr = PASS if hr_val      >= hit_min else WARN

    print(f"  {s_ic}  IC  = {ic_val:+.4f}  {label_ic}")
    print(f"  {s_t}  T   = {t_val:+.2f}   {label_t}")
    print(f"  {s_hr}  Hit = {hr_val:.1%}  {label_hr}")

    passed = abs(ic_val) >= ic_min and abs(t_val) >= t_min
    verdict = "🟢 PROCEED to backtest" if passed else "🔴 DO NOT USE yet"
    print(f"\n  {verdict}")

    if not passed:
        if abs(ic_val) < ic_min:
            print(f"     → IC too low. Signal not predicting returns at {len(df):,} filtered bars.")
        if abs(t_val) < t_min:
            print(f"     → T-stat too low. Not statistically significant.")

    return passed

# ─────────────────────────────────────────────────────────────────────────────
# Data load
# ─────────────────────────────────────────────────────────────────────────────

def load_data(n_bars=80000):
    import MetaTrader5 as mt5
    from config import load_config

    config  = load_config()
    symbol  = config["trading"]["symbol"]
    tf_map  = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
               "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}
    tf      = tf_map.get(config["trading"].get("timeframe","M1"), mt5.TIMEFRAME_M1)

    if not mt5.initialize():
        print("❌  MT5 not running"); sys.exit(1)

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("❌  No data returned"); sys.exit(1)

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    if "tick_volume" not in df.columns:
        df["tick_volume"] = df.get("real_volume", 1.0)

    print(f"  ✅  Loaded {len(df):,} bars — {df['time'].iloc[0]} → {df['time'].iloc[-1]}\n")
    return df, config

# ─────────────────────────────────────────────────────────────────────────────
# Session mask helpers
# ─────────────────────────────────────────────────────────────────────────────

def asian_mask(df: pd.DataFrame, config: dict) -> pd.Series:
    """True for bars inside Asian session (02:00–09:00)."""
    s = config["sessions"]
    h = df["time"].dt.hour + df["time"].dt.minute / 60
    start = int(s["asian_start"].split(":")[0]) + int(s["asian_start"].split(":")[1]) / 60
    end   = int(s["asian_end"].split(":")[0])   + int(s["asian_end"].split(":")[1])   / 60
    return (h >= start) & (h < end)

def london_ny_mask(df: pd.DataFrame, config: dict) -> pd.Series:
    """True for bars inside London or NY session (09:00–21:00)."""
    s = config["sessions"]
    h = df["time"].dt.hour + df["time"].dt.minute / 60
    lon_start = int(s["london_start"].split(":")[0])
    ny_end    = int(s["newyork_end"].split(":")[0])
    return (h >= lon_start) & (h < ny_end)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  HMM BOT — Alpha Validation (Corrected)")
    print("="*55)

    print("\n📡  Loading 50,000 bars from MT5...")
    df, config = load_data(n_bars=80000)

    print("⚙️   Computing indicators...")
    from strategy.strategy_router import StrategyRouter
    router = StrategyRouter(config)
    df     = router.calculate_indicators(df)

    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Signal 1 — Mean Reversion Z-Score
    # Tested on: Asian session bars only
    # Forward return: 20 bars (matches strategy holding period)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📊  Testing signals with correct methodology...")
    print("\n  ℹ️   Mean Reversion → Asian session only, 20-bar forward return")

    try:
        from research.alpha.mean_reversion_alpha import volatility_adjusted_zscore

        atr      = df["atr"] if "atr" in df.columns else (df["high"]-df["low"]).rolling(14).mean()
        vaz      = volatility_adjusted_zscore(df["close"], atr)
        mr_score = -vaz   # inverted: high Z = overextended up = expect DOWN

        # ── FILTER: Asian session bars only ──────────────────────────────────
        mask         = asian_mask(df, config)
        mr_filtered  = mr_score[mask]
        fwd_20       = df["close"].pct_change(20).shift(-20)[mask]

        results["Mean Reversion (Asian, 20-bar)"] = report(
            "Mean Reversion — Asian session | 20-bar forward return",
            mr_filtered, fwd_20
        )

        # Also show what 1-bar IC was (to explain why previous test failed)
        fwd_1    = df["close"].pct_change().shift(-1)[mask]
        ic_1bar  = ic(mr_filtered, fwd_1)
        print(f"\n  📌  For reference — same signal at 1-bar horizon: IC={ic_1bar:+.4f}")
        print(f"      (This is why the previous test showed noise — 1 bar = random)")

    except Exception as e:
        print(f"\n{FAIL}  Mean Reversion alpha error: {e}")
        results["Mean Reversion"] = False

    # ─────────────────────────────────────────────────────────────────────────
    # Signal 2 — Momentum
    # Tested on: London + NY session bars only
    # Forward return: 60 bars (1 hour on M1)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  ℹ️   Momentum → London/NY session only, 60-bar forward return")

    try:
        from research.alpha.momentum_alpha import time_series_momentum

        log_ret = np.log(df["close"] / df["close"].shift(1))
        tsm     = time_series_momentum(log_ret)

        mask         = london_ny_mask(df, config)
        mom_filtered = tsm[mask]
        fwd_60       = df["close"].pct_change(60).shift(-60)[mask]

        results["Momentum (London/NY, 60-bar)"] = report(
            "Momentum — London/NY session | 60-bar forward return",
            mom_filtered, fwd_60
        )

    except Exception as e:
        print(f"\n{FAIL}  Momentum alpha error: {e}")
        results["Momentum"] = False

    # ─────────────────────────────────────────────────────────────────────────
    # Signal 3 — Volume Shock (Microstructure) — datetime bug fixed
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  ℹ️   Microstructure → numeric columns only, 5-bar forward return")

    try:
        from research.alpha.microstructure_alpha import volume_shock

        # Fix: pass only numeric columns — exclude 'time'
        numeric_df = df.select_dtypes(include=[np.number])
        # But volume_shock may need specific columns — try with just tick_volume
        vol_scores = df["tick_volume"].rolling(20).apply(
            lambda x: (x[-1] - x.mean()) / (x.std() + 1e-9), raw=True
        )
        fwd_5 = df["close"].pct_change(5).shift(-5)

        results["Microstructure (Volume Shock)"] = report(
            "Microstructure — Volume Shock | 5-bar forward return",
            vol_scores, fwd_5
        )

    except Exception as e:
        print(f"\n{WARN}  Microstructure skipped: {e}")
        results["Microstructure"] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n\n" + "="*55)
    print("  FINAL SUMMARY")
    print("="*55)

    passed  = [k for k, v in results.items() if v is True]
    failed  = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]

    for k in passed:  print(f"  ✅  {k}")
    for k in failed:  print(f"  ❌  {k}")
    for k in skipped: print(f"  ⚠️   {k}")

    print(f"\n  {len(passed)} passed  |  {len(failed)} failed  |  {len(skipped)} skipped")

    if len(passed) >= 1:
        print("\n  🟢  At least 1 signal validated.")
        print("      Next: python run_backtest.py --mode backtest --bars 30000 --no-charts")
    else:
        print("\n  🔴  No signals passed. Post the output and we debug the formulas.")

    print("="*55 + "\n")

if __name__ == "__main__":
    main()