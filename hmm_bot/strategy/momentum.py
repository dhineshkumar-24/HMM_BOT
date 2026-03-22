"""
strategy/momentum.py — Regime-Adaptive Alpha Strategy.

MEAN_REVERT regime  → fade short-term overextension (z_vwap signal)
TRENDING regime     → ride 1-hour momentum on pullbacks (alpha_mom signal)
HIGH_VOL regime     → reduce size, still trade with mean reversion only
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from strategy.strategy_base import StrategyBase
from utils.features         import build_alpha_features
from utils.logger           import setup_logger
from core.hmm_model         import (
    REGIME_MEAN_REVERT,
    REGIME_TRENDING,
    REGIME_HIGH_VOL,
)

logger = setup_logger("MomentumStrategy")

ACTIVE_SESSIONS = ("london", "newyork", "asian")


class MomentumStrategy(StrategyBase):
    """Regime-adaptive strategy."""

    def __init__(self, config: dict):
        self.config  = config
        alpha_cfg    = config.get("strategy", {}).get("alpha", {})
        mr_cfg       = config.get("strategy", {}).get("mean_reversion", {})
        mom_cfg      = config.get("strategy", {}).get("momentum", {})

        self.tp_vol_mult        = alpha_cfg.get("tp_vol_mult", 6.0)
        self.sl_vol_mult        = alpha_cfg.get("sl_vol_mult", 4.0)
        self.min_edge_over_cost = alpha_cfg.get("min_edge_pips", 0.00015)
        self.min_combined_score = alpha_cfg.get("min_combined_score", 1.80)
        
        # Load the thresholds from config instead of hardcoding
        self.mr_threshold  = mr_cfg.get("z_score_trigger", 2.3)
        self.mom_threshold = mom_cfg.get("mom_threshold", 2.7)

        # Internal strategy filters (exposed for configuration)
        self.mr_mom_filter  = alpha_cfg.get("mr_mom_filter", 1.5)  # Don't fade if momentum is stronger than this
        self.trend_pullback = alpha_cfg.get("trend_pullback", 1.0) # Pulback depth required in trend
        
        logger.info(
            f"AlphaStrategy ready | "
            f"MR Thr: {self.mr_threshold} | Mom Thr: {self.mom_threshold} | "
            f"TP={self.tp_vol_mult}x vol | SL={self.sl_vol_mult}x vol"
        )

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df   = df.copy()
        
        # Pull windows from config, defaulting to 20 and 60 if not specified
        vwap_window = self.config.get("strategy", {}).get("vwap_window", 20)
        mom_period  = self.config.get("strategy", {}).get("momentum", {}).get("mom_period", 60)
        
        feats = build_alpha_features(df, vwap_window=vwap_window, mom_period=mom_period)
        for col in feats.columns:
            df[col] = feats[col]
        return df

    def generate_signal(
        self,
        df:      pd.DataFrame,
        regime:  Optional[int] = None,
        session: Optional[str] = None,
        bias_4h:   str = "NEUTRAL",
    ) -> Optional[dict]:

        if len(df) < 120:
            return None

        prev = df.iloc[-2]

        alpha_mr  = prev.get("alpha_mr",  float("nan"))
        alpha_mom = prev.get("alpha_mom", float("nan"))
        vol10     = prev.get("vol10",     float("nan"))
        z_vwap    = prev.get("z_vwap",    float("nan"))
        vol_regime = prev.get("vol_regime", float("nan"))
        mr_quality      = prev.get("mr_quality",      float("nan"))
        exhaustion_long = prev.get("exhaustion_long", float("nan"))

        if any(np.isnan(v) for v in [alpha_mr, alpha_mom, vol10, z_vwap]):
            return None

        if vol10 <= 0:
            return None

        entry = float(df.iloc[-1]["close"])
        price = float(prev["close"])
        vol_price = vol10 * price

        if vol_price < self.min_edge_over_cost:
            return None

        # ── REGIME-ADAPTIVE SIGNAL SELECTION ─────────────────────────────────
        direction = None
        signal_strength = 1.0
        mode = "mean_rev"

        if regime == REGIME_MEAN_REVERT or regime is None:
            # In mean-reverting regime: fade Z-score extremes
            # Only when 60-min momentum is NOT strongly opposing
            mode = "mean_rev"

            # if session == "newyork":  
            #     return None

            mr_threshold = self.mr_threshold + 0.5 if session == "newyork" else self.mr_threshold

            if alpha_mr > mr_threshold and alpha_mom > -self.mr_mom_filter:
                direction = "BUY"
                signal_strength = min(alpha_mr, 5.5)

            elif alpha_mr < -mr_threshold and alpha_mom < self.mr_mom_filter: # FIXED: was negative
                direction = "SELL"
                signal_strength = min(abs(alpha_mr), 5.5)

            # Check secondary slightly lower threshold
            secondary_mr_thr = max(0.8, self.mr_threshold * 0.8)
            if alpha_mr > secondary_mr_thr and alpha_mom > -self.mr_mom_filter:
                # Price extended DOWN (z_vwap <-2), not in downtrend → BUY
                direction = "BUY"
                signal_strength = min(alpha_mr, 5.5)

            elif alpha_mr < -secondary_mr_thr and alpha_mom < self.mr_mom_filter: # FIXED: was negative
                # Price extended UP (z_vwap > +2), not in uptrend → SELL
                direction = "SELL"
                signal_strength = min(abs(alpha_mr), 5.5)

        elif regime == REGIME_TRENDING:
            mode = "momentum_pullback"

            # ── EMA direction filter — fast EMAs for M1 ────────────────
            close_series = df["close"]
            ema21  = close_series.ewm(span=21,  adjust=False).mean()
            ema50  = close_series.ewm(span=50,  adjust=False).mean()
            ema100 = close_series.ewm(span=100, adjust=False).mean()

            # All three must agree on direction
            trend_up   = (float(ema21.iloc[-2]) > float(ema50.iloc[-2]) and
                          float(ema50.iloc[-2]) > float(ema100.iloc[-2]))
            trend_down = (float(ema21.iloc[-2]) < float(ema50.iloc[-2]) and
                          float(ema50.iloc[-2]) < float(ema100.iloc[-2]))

            # New York — require stronger alignment
            if session == "newyork":
                # Also check momentum is accelerating not fading
                ema21_prev = float(ema21.iloc[-3])
                ema21_curr = float(ema21.iloc[-2])
                ema_accel_up   = ema21_curr > ema21_prev   # EMA21 still rising
                ema_accel_down = ema21_curr < ema21_prev   # EMA21 still falling

                if alpha_mom > self.mom_threshold and alpha_mr > self.trend_pullback:
                    if not (trend_up and ema_accel_up):
                        return None
                    direction = "BUY"
                    signal_strength = min(alpha_mr, 4.5)

                elif alpha_mom < -self.mom_threshold and alpha_mr < -self.trend_pullback:
                    if not (trend_down and ema_accel_down):
                        return None
                    direction = "SELL"
                    signal_strength = min(abs(alpha_mr), 5.5)

            # London — standard EMA check
            else:
                l_mom_threshold = self.mom_threshold + 0.7
                if alpha_mom > l_mom_threshold and alpha_mr > self.trend_pullback:
                    if not trend_up:
                        return None
                    direction = "BUY"
                    signal_strength = min(alpha_mr, 5.5)

                elif alpha_mom < -l_mom_threshold and alpha_mr < -self.trend_pullback:
                    if not trend_down:
                        return None
                    direction = "SELL"
                    signal_strength = min(abs(alpha_mr), 5.5)

        elif regime == REGIME_HIGH_VOL:
            # High vol: only take very high conviction mean reversion
            # Reduce threshold — need stronger signal
            mode = "high_vol_mr"

            hv_mom_threshold = self.mom_threshold + 0.5
            if alpha_mr > self.mr_threshold and alpha_mom > hv_mom_threshold:
                direction = "BUY"
                signal_strength = min(alpha_mr, 5.5)

            elif alpha_mr < -self.mr_threshold and alpha_mom < -hv_mom_threshold:
                direction = "SELL"
                signal_strength = min(abs(alpha_mr), 5.5)

        if bias_4h == "DOWN" and direction == "BUY":
            return None
        if bias_4h == "UP" and direction == "SELL":
            return None
        
        if direction is None:
            return None
            
        # -- MECHANICAL INVERSION (Alpha Edge) --
        # Reverses the consistent mechanical false-breakouts to capture pure positive returns.
        direction = "SELL" if direction == "BUY" else "BUY"
        
        #── Gate: reject weak signals ─────────────────────────────────────────
        if signal_strength < self.min_combined_score:
            return None

        # ── SL / TP — regime-scaled ───────────────────────────────────────────
        # In trending regime use slightly wider SL/TP, but strictly obey user config baseline
        if regime == REGIME_TRENDING:
            sl_mult = self.sl_vol_mult * 1.2
            tp_mult = self.tp_vol_mult * 1.5
        elif regime == REGIME_HIGH_VOL:
            sl_mult = self.sl_vol_mult * 0.8  # tighter SL in high vol
            tp_mult = self.tp_vol_mult * 1.2  # slightly wider TP
        else:
            sl_mult = self.sl_vol_mult
            tp_mult = self.tp_vol_mult

        sl_dist = max(vol_price * sl_mult, 0.00060)  # 6 pips floor
        tp_dist = max(vol_price * tp_mult, 0.00040)  # 4 pips floor independent of SL

        # Disable TP scaling to strictly enforce scalping targets
        tp_dist = tp_dist

        if direction == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        reason = (
            f"mode={mode} | combined={signal_strength:+.2f} | "
            f"a_mr={alpha_mr:+.2f} a_mom={alpha_mom:+.2f} "
            f"z={z_vwap:+.2f} vol_reg={vol_regime:+.2f} | "
            f"vol10={vol10:.6f} | session={session}"
        )

        logger.info(f"[Alpha] {direction} | {reason}")

        return {
            "direction": direction,
            "entry":     round(entry, 5),
            "sl":        round(sl, 5),
            "tp":        round(tp, 5),
            "atr":       round(vol_price, 6),
            "trail_sl":  round(sl_dist * 0.8, 6),
            "reason":    reason,
        }