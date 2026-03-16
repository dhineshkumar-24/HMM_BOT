"""
strategy/mean_reversion.py — Mean Reversion Strategy.

Active conditions:
    Session : Asian only  (02:00 – 09:00 broker time)
    Regime  : MEAN_REVERT (0)

Entry rules (all must pass):
    1. |Z-score| >= z_score_trigger      (price is at an extreme)
    2. EMA50 slope is near-flat           (no strong trend present)
    3. ADX < adx_max (22)                 (market is not trending)
    4. RSI confirms direction             (overbought for SELL, oversold for BUY)

SL / TP / Trail:
    SL    = entry ± ATR * 2.0
    TP    = rolling VWAP (mean-reversion target)
    Trail = ATR * 1.5  (stored in signal for executor to use)

Signal format returned:
    {
        "direction": "BUY" | "SELL",
        "entry":     float,
        "sl":        float,
        "tp":        float,
        "atr":       float,
        "reason":    str,
    }
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from strategy.strategy_base import StrategyBase
from core.hmm_model          import REGIME_MEAN_REVERT
from utils.helpers           import SESSION_ASIAN
from utils.indicators        import (
    compute_vwap,
    compute_atr,
    compute_rsi,
    compute_ema,
    compute_ema_slope,
    compute_adx,
    compute_volatility,
)
from utils.logger import setup_logger

logger = setup_logger("MeanReversion")


class MeanReversionStrategy(StrategyBase):
    """
    Mean-reversion strategy for low-vol, mean-reverting market regimes.

    Only activates in the Asian session when the HMM detects regime 0.
    Uses Z-score extremes confirmed by RSI and a near-flat EMA50 slope
    and low ADX to avoid entering into trending moves.
    """

    def __init__(self, config: dict):
        self.config = config
        s = config["strategy"]
        mr = s["mean_reversion"]

        # Shared
        self.rsi_period   = s["rsi_period"]
        self.atr_period   = s["atr_period"]
        self.adx_period   = s["adx_period"]
        self.vwap_window  = s["vwap_window"]

        # Strategy-specific
        self.z_trigger    = mr["z_score_trigger"]     # |Z| >= this
        self.rsi_ob       = mr["rsi_overbought"]      # sell if RSI > this
        self.rsi_os       = mr["rsi_oversold"]        # buy  if RSI < this
        self.ema_flat_thr = mr["ema_slope_flat"]      # |slope| threshold
        self.adx_max      = mr["adx_max"]             # ADX must be below this
        self.sl_mult      = mr["sl_atr_mult"]         # SL = ATR * this
        self.trail_mult   = mr["trail_atr_mult"]
        self.min_rr       = mr.get("min_rr", 1.3)      # trailing = ATR * this

        logger.info(
            f"MeanReversionStrategy ready | "
            f"Z>={self.z_trigger} | ADX<{self.adx_max} | "
            f"RSI({self.rsi_ob}/{self.rsi_os}) | SL={self.sl_mult}xATR"
        )

    # ── StrategyBase interface ─────────────────────────────────────────────────

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich raw OHLCV DataFrame with mean-reversion indicators."""
        df = df.copy()

        log_ret       = np.log(df["close"] / df["close"].shift(1))
        df["returns"] = log_ret

        # VWAP and Z-score
        df["vwap"]    = compute_vwap(df, window=self.vwap_window)
        roll_std      = df["close"].rolling(self.vwap_window).std()
        df["z_score"] = (df["close"] - df["vwap"]) / roll_std.replace(0, np.nan)

        # ATR
        df["atr"]     = compute_atr(df, period=self.atr_period)

        # RSI
        df["rsi"]     = compute_rsi(df["close"], period=self.rsi_period)

        # EMA50 slope
        df["ema50"]         = compute_ema(df["close"], period=50)
        df["ema50_slope"]   = compute_ema_slope(df["close"], ema_period=50, slope_window=5)

        # ADX
        adx_df        = compute_adx(df, period=self.adx_period)
        df["adx"]     = adx_df["adx"]
        df["plus_di"] = adx_df["plus_di"]
        df["minus_di"]= adx_df["minus_di"]

        return df

    def generate_signal(
        self,
        df: pd.DataFrame,
        regime: Optional[int] = None,
        session: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Run mean-reversion entry logic on the most recent closed candle.

        Returns a signal dict or None.
        """
        # ── Guard: minimum data ────────────────────────────────────────────────
        if len(df) < 60:
            return None

        # ── Guard: session must be Asian ───────────────────────────────────────
        if session is not None and session != SESSION_ASIAN:
            return None

        # ── Guard: regime must be mean-reverting (or warm-up None) ────────────
        if regime is not None and regime != REGIME_MEAN_REVERT:
            return None

        prev = df.iloc[-2]   # Always use the confirmed closed candle

        # ── Extract indicator values ───────────────────────────────────────────
        z       = prev.get("z_score", float("nan"))
        rsi     = prev.get("rsi", float("nan"))
        atr     = prev.get("atr", float("nan"))
        adx     = prev.get("adx", float("nan"))
        slope   = prev.get("ema50_slope", float("nan"))
        vwap    = prev.get("vwap", float("nan"))

        # Skip on NaN warm-up bars
        if any(map(lambda v: v != v, [z, rsi, atr, adx, slope, vwap])):
            return None

        entry = float(df.iloc[-1]["close"])

        # ── Filter 1: EMA50 must be flat ───────────────────────────────────────
        if abs(slope) > self.ema_flat_thr:
            logger.debug(
                f"MR filter: EMA50 slope {slope:.6f} > flat threshold "
                f"{self.ema_flat_thr:.6f} → skip"
            )
            return None

        # ── Filter 2: ADX must be low (no strong trend) ────────────────────────
        if adx >= self.adx_max:
            logger.debug(f"MR filter: ADX {adx:.1f} >= {self.adx_max} → skip")
            return None

        # ── Filter 3 + 4: Z-score extreme + RSI confirmation ──────────────────
        direction = None
        reason    = ""

        if z >= self.z_trigger and rsi > self.rsi_ob:
            direction = "SELL"
            reason    = (
                f"Z={z:.2f}>={self.z_trigger} | RSI={rsi:.1f}>{self.rsi_ob} | "
                f"ADX={adx:.1f} | slope={slope:.6f}"
            )

        elif z <= -self.z_trigger and rsi < self.rsi_os:
            direction = "BUY"
            reason    = (
                f"Z={z:.2f}<=-{self.z_trigger} | RSI={rsi:.1f}<{self.rsi_os} | "
                f"ADX={adx:.1f} | slope={slope:.6f}"
            )

        if direction is None:
            return None

        # ── Compute SL / TP ────────────────────────────────────────────────────
        sl_dist = atr * self.sl_mult

        if direction == "BUY":
            sl = entry - sl_dist
            tp = entry + (vwap - entry) * 0.8

            if tp <= entry:
                return None

            reward = tp - entry
            risk   = entry - sl

        else:
            sl = entry + sl_dist
            tp = entry - (entry - vwap) * 0.8

            if tp >= entry:
                return None

            reward = entry - tp
            risk   = sl - entry


        # Enforce minimum risk/reward
        rr = reward / risk if risk > 0 else 0

        if rr < self.min_rr:
            logger.debug(f"MR filter: RR {rr:.2f} < {self.min_rr} → skip")
            return None

        signal = {
            "direction": direction,
            "entry":     round(entry, 5),
            "sl":        round(sl, 5),
            "tp":        round(tp, 5),
            "atr":       round(float(atr), 6),
            "trail_sl":  round(atr * self.trail_mult, 6),
            "reason":    reason,
        }

        logger.info(
            f"[MeanReversion] {direction} signal | "
            f"Entry:{entry:.5f} SL:{sl:.5f} TP:{tp:.5f} ATR:{atr:.6f} | "
            f"{reason}"
        )
        return signal
