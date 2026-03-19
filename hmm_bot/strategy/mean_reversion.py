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
        self.min_rr       = mr.get("min_rr", 1.3)
        self.tp_atr_floor_mult = mr.get("tp_atr_floor_mult", 3.0)      

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
        df["ema50"]       = compute_ema(df["close"], period=50)
        raw_slope         = compute_ema_slope(df["close"], ema_period=50, slope_window=5)
        df["ema50_slope"] = raw_slope / df["atr"].replace(0, np.nan)

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
        bias_4h: str = "NEUTRAL",
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

        # ── Filter 3: Z-score extreme — primary entry trigger ─────────────────
        # Z-score is the clean mean-reversion signal. RSI is repurposed below
        # as a trend-state filter (not an extremity filter) — avoiding the
        # logical contradiction where RSI extreme implies directional move
        # which conflicts with the flat EMA slope requirement.
        direction = None
        reason    = ""

        if z >= self.z_trigger:
            direction = "SELL"
        elif z <= -self.z_trigger:
            direction = "BUY"

        if direction is None:
            return None

        # ── Filter 4: RSI trend-state confirmation ────────────────────────────
        # RSI is used here as a trend-direction filter only.
        # For SELL: RSI must be above midpoint — price in upper half of range.
        # For BUY:  RSI must be below midpoint — price in lower half of range.
        # Thresholds of 55/45 are intentionally mild: we need trend-state
        # agreement, not an extreme reading (that would recreate the
        # contradiction with the flat-slope filter).
        if direction == "SELL" and rsi < self.rsi_ob:
            # RSI below overbought — price already fading, no overextension
            # to sell into. Skip.
            logger.debug(
                f"MR filter: SELL skipped — RSI {rsi:.1f} < {self.rsi_ob} "
                f"(no overbought state to fade)"
            )
            return None

        if direction == "BUY" and rsi > self.rsi_os:
            # RSI above oversold — price already recovering, no overextension
            # to buy into. Skip.
            logger.debug(
                f"MR filter: BUY skipped — RSI {rsi:.1f} > {self.rsi_os} "
                f"(no oversold state to fade)"
            )
            return None

        reason = (
            f"Z={z:.2f} | RSI={rsi:.1f} | "
            f"ADX={adx:.1f} | slope={slope:.6f}"
        )

        # ── Filter: 4H bias — don't fade the higher-timeframe trend ──────────
        if bias_4h == "DOWN" and direction == "BUY":
            logger.debug("MR filter: 4H bias DOWN, skipping BUY")
            return None
        if bias_4h == "UP" and direction == "SELL":
            logger.debug("MR filter: 4H bias UP, skipping SELL")
            return None

        # ── Compute SL / TP ────────────────────────────────────────────────────
        sl_dist = atr * self.sl_mult

        # Use the configured ATR floor multiplier directly.
        # This guarantees TP >= tp_atr_floor_mult × ATR regardless of VWAP
        # proximity. With sl_atr_mult=2.0, min_rr=1.3, tp_atr_floor_mult=3.0:
        #   required for RR: 2.0 × 1.3 = 2.6 ATR
        #   actual floor:    3.0 ATR > 2.6 ATR → RR filter always passes ✓
        #   round-trip cost: ~1.85 pips vs 3.0 ATR ≈ 24-54 pips → costs covered ✓
        tp_floor_dist = atr * self.tp_atr_floor_mult

        if direction == "BUY":
            sl = entry - sl_dist

            # VWAP target: 80% reversion toward rolling mean
            vwap_tp = entry + (vwap - entry) * 0.8

            # TP = whichever is further from entry: VWAP target or ATR floor
            # If VWAP is very close (small dislocation), ATR floor wins.
            # If VWAP is far (large dislocation), VWAP target wins — larger reward.
            tp = max(vwap_tp, entry + tp_floor_dist)

            if tp <= entry:
                return None

            reward = tp - entry
            risk   = entry - sl

        else:   # SELL
            sl = entry + sl_dist

            vwap_tp = entry - (entry - vwap) * 0.8

            tp = min(vwap_tp, entry - tp_floor_dist)

            if tp >= entry:
                return None

            reward = entry - tp
            risk   = sl - entry

        # RR check — with the ATR floor above this is almost always met.
        # Kept as a final safety gate for edge cases (e.g. ATR spike between
        # signal bar and entry bar changing the effective distances).
        rr = reward / risk if risk > 0 else 0

        if rr < self.min_rr:
            logger.debug(
                f"MR filter: RR {rr:.2f} < {self.min_rr} after ATR floor → skip "
                f"(entry={entry:.5f} sl={sl:.5f} tp={tp:.5f} atr={atr:.6f})"
            )
            return None


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
