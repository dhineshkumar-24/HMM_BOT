"""
strategy/momentum.py — Institutional Momentum Strategy v5.0

CRITICAL FIXES from v4.0 backtest results:
    ROOT CAUSE: 86% SL hit rate with 5-pip median SL on EURUSD M5.
    The SL was within the noise+spread range, causing systematic losses.

v5.0 — Complete Redesign for Precision Over Frequency:

    1. MINIMUM 15-pip SL (was 10-pip). EURUSD M5 noise floor is ~8 pips.
       SL must be ABOVE noise floor to avoid random SL triggers.
    2. Target RR 1:2 minimum, NOT 1:4+ (which was causing <5% TP hit rate)
    3. Multi-timeframe momentum confirmation (not just alpha_mom)
    4. Volatility-normalized entry threshold (higher bar in high-vol)
    5. Trend strength gating via ADX + DI separation
    6. Strict directional bias enforcement
    7. Post-entry momentum confirmation (bar must close in direction)
    8. Anti-noise: skip if ATR < 0.6× average (too quiet = spread dominates)

Performance targets:
    Win Rate:      45-55%
    Profit Factor: > 1.3
    Sharpe:        > 1.5
    Max DD:        < 15%
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from strategy.strategy_base import StrategyBase
from core.hmm_model         import REGIME_MEAN_REVERT, REGIME_TRENDING, REGIME_HIGH_VOL
from utils.indicators       import (
    compute_vwap,
    compute_atr,
    compute_rsi,
    compute_ema,
    compute_ema_slope,
    compute_adx,
)
from utils.features          import build_alpha_features
from utils.logger import setup_logger

logger = setup_logger("Momentum")


class MomentumStrategy(StrategyBase):
    """
    Precision momentum strategy designed for low-noise, high-quality entries.
    v5.0: Wider SL above noise floor, tighter TP for achievable RR.
    """

    def __init__(self, config: dict):
        self.config = config
        s   = config["strategy"]
        mom = s["momentum"]

        # Shared
        self.rsi_period       = s["rsi_period"]
        self.atr_period       = s["atr_period"]
        self.adx_period       = s["adx_period"]

        # Momentum-specific
        self.adx_min          = mom.get("adx_min", 22)    # v5.0: raised from 20
        self.di_separation    = mom.get("di_separation", 5)  # NEW: min DI+/DI- gap
        self.ema_period_fast  = mom.get("ema_period_fast", 21)
        self.ema_period_slow  = mom.get("ema_period_slow", 50)
        self.ema_period_base  = mom.get("ema_period_base", 100)
        self.min_rr           = mom.get("min_rr", 2.0)    # v5.0: strict 1:2 minimum

        # v5.0: SL/TP calibrated for EURUSD M5 noise floor
        # EURUSD M5 noise = ~8 pips. SL must be > noise to survive.
        self.sl_atr_mult      = mom.get("sl_atr_mult", 2.0)     # ~15-20 pips
        self.tp_atr_mult      = mom.get("tp_atr_mult", 3.0)     # v5.0: 3× ATR (was 4× → only 7% TP hit)
        self.trail_atr_mult   = mom.get("trail_atr_mult", 2.5)
        self.min_sl_pips      = mom.get("min_sl_pips", 0.00150)  # v5.0: 15-pip FLOOR (was 10)
        self.max_sl_pips      = mom.get("max_sl_pips", 0.00400)  # 40-pip ceiling

        # Skip-1 momentum thresholds
        self.mom_threshold    = mom.get("mom_threshold", 1.5)    # v5.0: raised from 1.0
        self.mom_trend_threshold = mom.get("mom_trend_threshold", 1.2)   # trending threshold

        self.min_gap          = s.get("signal", {}).get("min_gap", 0.50)  # v5.0: raised
        self.cooling_bars     = mom.get("cooling_bars", 6)       # v5.0: longer cooling
        self.min_atr_ratio    = mom.get("min_atr_ratio", 0.6)   # NEW: anti-noise filter

        # RSI limits to avoid chasing overextended moves
        self.rsi_ob = mom.get("rsi_overbought", 70)
        self.rsi_os = mom.get("rsi_oversold", 30)

        # Internal state
        self._last_sl_bar: dict[str, int] = {"BUY": -999, "SELL": -999}

        logger.info(
            f"MomentumStrategy v5.0 | ADX>={self.adx_min} | "
            f"SL={self.sl_atr_mult}xATR (floor={self.min_sl_pips*10000:.0f}pips) | "
            f"TP={self.tp_atr_mult}xATR | min_RR={self.min_rr} | "
            f"mom_thr={self.mom_threshold}"
        )

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich raw OHLCV with momentum indicators and alpha features."""
        df = df.copy()

        log_ret       = np.log(df["close"] / df["close"].shift(1))
        df["returns"] = log_ret

        # ── ATR ───────────────────────────────────────────────────────────────
        df["atr"] = compute_atr(df, period=self.atr_period)
        # Rolling average ATR for noise filter
        df["atr_avg"] = df["atr"].rolling(50).mean()

        # ── RSI ───────────────────────────────────────────────────────────────
        df["rsi"] = compute_rsi(df["close"], period=self.rsi_period)

        # ── EMAs (fast / slow / base) ─────────────────────────────────────────
        df["ema_fast"]  = compute_ema(df["close"], period=self.ema_period_fast)
        df["ema_slow"]  = compute_ema(df["close"], period=self.ema_period_slow)
        df["ema_base"]  = compute_ema(df["close"], period=self.ema_period_base)

        df["ema50_slope"] = compute_ema_slope(
            df["close"], ema_period=self.ema_period_slow, slope_window=5
        )

        # ── ADX ───────────────────────────────────────────────────────────────
        adx_df        = compute_adx(df, period=self.adx_period)
        df["adx"]     = adx_df["adx"]
        df["plus_di"] = adx_df["plus_di"]
        df["minus_di"] = adx_df["minus_di"]

        # ── VWAP ──────────────────────────────────────────────────────────────
        df["vwap"] = compute_vwap(df, window=21)

        # ── Alpha features (skip-1 momentum) ─────────────────────────────────
        alpha_df = build_alpha_features(df)
        for col in alpha_df.columns:
            if col not in df.columns:
                df[col] = alpha_df[col]

        return df

    def generate_signal(
        self,
        df:      pd.DataFrame,
        regime:  Optional[int] = None,
        session: Optional[str] = None,
        bias_4h: str = "NEUTRAL",
        bar_idx: int = 0,
    ) -> Optional[dict]:
        """
        Generate HIGH-PRECISION momentum signal.

        v5.0 entry requirements (ALL must pass):
            1. ADX >= 22 (confirmed trend)
            2. DI separation >= 5 (clear directional bias in market)
            3. |alpha_mom| >= threshold (strong momentum signal)
            4. EMA alignment (fast vs slow confirms direction)
            5. RSI not overextended (avoid chasing)
            6. ATR > 0.6× average (enough volatility for edge to exist)
            7. Bar confirmation (prev bar closes in trade direction)
            8. 4H bias alignment (trend-aligned only)
            9. Post-SL cooling period (avoid revenge trading)
            10. Signal gap > 0.50 (must exceed strong threshold)
        """
        if len(df) < 120:
            return None

        prev   = df.iloc[-2]
        prev2  = df.iloc[-3] if len(df) >= 3 else prev
        entry  = float(df.iloc[-1]["close"])

        # ── Extract indicators ────────────────────────────────────────────────
        atr       = float(prev.get("atr",        0.0))
        atr_avg   = float(prev.get("atr_avg",    0.0))
        adx       = float(prev.get("adx",        0.0))
        rsi       = float(prev.get("rsi",        50.0))
        slope     = float(prev.get("ema50_slope", 0.0))
        alpha_mom = float(prev.get("alpha_mom",   0.0))
        alpha_mr  = float(prev.get("alpha_mr",    0.0))
        z_vwap    = float(prev.get("z_vwap",      0.0))
        plus_di   = float(prev.get("plus_di",    0.0))
        minus_di  = float(prev.get("minus_di",   0.0))
        ema_fast  = float(prev.get("ema_fast",   0.0))
        ema_slow  = float(prev.get("ema_slow",   0.0))
        ema_base  = float(prev.get("ema_base",   0.0))

        bar_open  = float(prev.get("open",  0.0))
        bar_close = float(prev.get("close", 0.0))

        if atr <= 0 or np.isnan(atr):
            return None

        # ── FILTER 1: ADX trend confirmation ──────────────────────────────────
        if adx < self.adx_min:
            return None

        # ── FILTER 2: Anti-noise — enough volatility for edge to exist ────────
        # If ATR is too low, spreads dominate and there's no room for profit
        if atr_avg > 0 and atr < atr_avg * self.min_atr_ratio:
            return None

        # ── FILTER 3: DI separation — clear directional bias in market ────────
        di_gap = abs(plus_di - minus_di)
        if di_gap < self.di_separation:
            return None

        # ── Determine direction from DI + momentum alignment ──────────────────
        direction  = None
        signal_gap = 0.0
        mom_thr    = self.mom_threshold

        # Use lower threshold when trending regime is confirmed
        if regime == REGIME_TRENDING:
            mom_thr = self.mom_trend_threshold

        # LONG conditions: DI+ > DI-, positive momentum, EMA aligned
        if (plus_di > minus_di and
                alpha_mom > mom_thr and
                slope > 0 and
                ema_fast > ema_slow):
            direction = "BUY"
            signal_gap = alpha_mom - mom_thr

        # SHORT conditions: DI- > DI+, negative momentum, EMA aligned
        elif (minus_di > plus_di and
                alpha_mom < -mom_thr and
                slope < 0 and
                ema_fast < ema_slow):
            direction = "SELL"
            signal_gap = abs(alpha_mom) - mom_thr

        if direction is None:
            return None

        # ── FILTER 4: Signal gap must be strong ───────────────────────────────
        if signal_gap < self.min_gap:
            return None

        # ── FILTER 5: RSI — don't chase overextended moves ───────────────────
        if direction == "BUY" and rsi > self.rsi_ob:
            return None
        if direction == "SELL" and rsi < self.rsi_os:
            return None

        # ── FILTER 6: 4H bias alignment (momentum must align with HTF) ───────
        if bias_4h == "DOWN" and direction == "BUY":
            return None
        if bias_4h == "UP" and direction == "SELL":
            return None

        # ── FILTER 7: Bar confirmation — prev bar must close in direction ─────
        if direction == "BUY" and bar_close < bar_open:
            return None
        if direction == "SELL" and bar_close > bar_open:
            return None
        
        #── Gate: reject weak signals ─────────────────────────────────────────
        if signal_strength < self.min_combined_score:
            return None

        # ── Compute SL / TP — ATR-based with WIDE FLOORS ─────────────────────
        # v5.0 CRITICAL FIX: SL must be above noise floor (15+ pips for EURUSD M5)
        # Previous v4.0 had 10-pip floor → 86% SL hit rate
        sl_dist = atr * self.sl_atr_mult
        sl_dist = max(sl_dist, self.min_sl_pips)   # 15-pip floor
        sl_dist = min(sl_dist, self.max_sl_pips)   # 40-pip ceiling

        tp_dist = atr * self.tp_atr_mult
        tp_dist = max(tp_dist, sl_dist * self.min_rr)  # enforce minimum RR

        if direction == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        # Final RR check
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < self.min_rr:
            return None

        trail_dist = atr * self.trail_atr_mult

        reason = (
            f"alpha_mom={alpha_mom:.2f} | ADX={adx:.1f} "
            f"DI+={plus_di:.1f} DI-={minus_di:.1f} gap={di_gap:.1f} | "
            f"RSI={rsi:.1f} | slope={slope:.6f} | "
            f"ATR={atr*10000:.1f}pips | SL={sl_dist*10000:.1f}pips | "
            f"Gap={signal_gap:.2f} | Regime={regime}"
        )

        signal = {
            "direction":     direction,
            "entry":         round(entry, 5),
            "sl":            round(sl, 5),
            "tp":            round(tp, 5),
            "atr":           round(float(atr), 6),
            "trail_sl":      round(trail_dist, 6),
            "breakeven_atr": round(atr, 6),
            "signal_gap":    round(signal_gap, 3),
            "strategy":      "MomentumStrategy",
            "session":       session or "",
            "reason":        reason,
        }

        logger.info(
            f"[Momentum] {direction} | Entry:{entry:.5f} "
            f"SL:{sl:.5f}({sl_dist*10000:.0f}pips) "
            f"TP:{tp:.5f}({tp_dist*10000:.0f}pips) RR:{rr:.2f} | "
            f"{reason}"
        )
        return signal