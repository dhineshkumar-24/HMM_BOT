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

        self.tp_vol_mult        = alpha_cfg.get("tp_vol_mult", 6.0)
        self.sl_vol_mult        = alpha_cfg.get("sl_vol_mult", 4.0)
        self.min_edge_over_cost = alpha_cfg.get("min_edge_pips", 0.00015)
        self.min_combined_score = alpha_cfg.get("min_combined_score", 1.80)
        logger.info(
            f"AlphaStrategy ready | "
            f"Threshold: 70th percentile | "
            f"TP={self.tp_vol_mult}x vol | SL={self.sl_vol_mult}x vol"
        )

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df   = df.copy()
        feats = build_alpha_features(df)
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

        from core.hmm_model import REGIME_MEAN_REVERT
        from utils.helpers import SESSION_ASIAN
        if session == SESSION_ASIAN and regime == REGIME_MEAN_REVERT:
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
        signal_strength = 0.0
        mode = "mean_rev"

        if regime == REGIME_MEAN_REVERT or regime is None:
            # Mean-reverting regime: fade Z-score extremes vs VWAP.
            #
            # alpha_mr = -z_vwap. Interpretation:
            #   alpha_mr > 0 → price BELOW VWAP → fading exhaustion → BUY
            #   alpha_mr < 0 → price ABOVE VWAP → fading exhaustion → SELL
            #
            # Two conviction tiers in a single block — no overwrite risk:
            #   Standard (|alpha_mr| >= mr_threshold): entry with base strength.
            #   High conviction (|alpha_mr| >= 3.0): entry with boosted strength.
            #     High conviction requires stronger opposing-momentum filter
            #     because a large Z-score during an actual trend is a trap.
            #
            # Momentum guard logic (alpha_mom):
            #   alpha_mom is volatility-normalised 60-bar return.
            #   For standard entries: block if momentum strongly opposes
            #     direction (threshold ±2.0 — meaningful opposition only).
            #   For high conviction: tighter filter (±1.5) because at
            #     alpha_mr > 3.0 we are already far from VWAP — if momentum
            #     is also running against us this is a trend, not a mean-rev.
            mode = "mean_rev"

            # NY session uses a tighter entry threshold — NY bars have
            # higher realised vol and wider spreads, requiring a larger
            # Z-score dislocation before the edge covers transaction costs.
            mr_threshold = 2.5 if session == "newyork" else 1.5

            # ── Single tiered direction assignment ────────────────────────────
            direction       = None
            signal_strength = 0.0

            if alpha_mr >= 3.0 and alpha_mom > -1.5:
                # High conviction BUY: price very far below VWAP,
                # momentum not strongly bearish.
                direction       = "BUY"
                signal_strength = min(alpha_mr * 1.2, 5.0)   # boosted

            elif alpha_mr <= -3.0 and alpha_mom < 1.5:
                # High conviction SELL: price very far above VWAP,
                # momentum not strongly bullish.
                direction       = "SELL"
                signal_strength = min(abs(alpha_mr) * 1.2, 5.0)   # boosted

            elif alpha_mr >= mr_threshold and alpha_mom > -2.0:
                # Standard BUY: price moderately below VWAP,
                # momentum not strongly opposing.
                direction       = "BUY"
                signal_strength = min(alpha_mr, 5.0)

            elif alpha_mr <= -mr_threshold and alpha_mom < 2.0:
                # Standard SELL: price moderately above VWAP,
                # momentum not strongly opposing.
                direction       = "SELL"
                signal_strength = min(abs(alpha_mr), 5.0)

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

                if alpha_mom > 3.0 and alpha_mr > 1.0:
                    if not (trend_up and ema_accel_up):
                        return None
                    direction = "BUY"
                    signal_strength = min(alpha_mr, 5.0)

                elif alpha_mom < -3.0 and alpha_mr < -1.0:
                    if not (trend_down and ema_accel_down):
                        return None
                    direction = "SELL"
                    signal_strength = min(abs(alpha_mr), 5.0)

            # London — standard EMA check
            else:
                if alpha_mom > 2.5 and alpha_mr > 1.0:
                    if not trend_up:
                        return None
                    direction = "BUY"
                    signal_strength = min(alpha_mr, 5.0)

                elif alpha_mom < -2.5 and alpha_mr < -1.0:
                    if not trend_down:
                        return None
                    direction = "SELL"
                    signal_strength = min(abs(alpha_mr), 5.0)

        elif regime == REGIME_HIGH_VOL:
            # High vol: only take very high conviction mean reversion
            # Reduce threshold — need stronger signal
            mode = "high_vol_mr"

            if alpha_mr > 2.0 and alpha_mom > 0:
                direction = "BUY"
                signal_strength = min(alpha_mr, 5.0)

            elif alpha_mr < -2.0 and alpha_mom < 0:
                direction = "SELL"
                signal_strength = min(abs(alpha_mr), 5.0)

        if bias_4h == "DOWN" and direction == "BUY":
            return None
        if bias_4h == "UP" and direction == "SELL":
            return None
        

        if direction is None:
            return None
        
        #── Gate: reject weak signals ─────────────────────────────────────────
        if signal_strength < self.min_combined_score:
            return None

        if len(df) >= 3:
            prev2      = df.iloc[-3]
            alpha_mr2  = prev2.get("alpha_mr",  0.0)
            alpha_mom2 = prev2.get("alpha_mom", 0.0)

            if direction == "BUY":
                # Both bars must show positive alpha_mr (price below VWAP)
                if alpha_mr2 < 0.5:
                    logger.debug(
                        f"Confirmation filter: BUY blocked — "
                        f"prev2 alpha_mr={alpha_mr2:.2f} < 0.5"
                    )
                    return None
            else:
                # Both bars must show negative alpha_mr (price above VWAP)
                if alpha_mr2 > -0.5:
                    logger.debug(
                        f"Confirmation filter: SELL blocked — "
                        f"prev2 alpha_mr={alpha_mr2:.2f} > -0.5"
                    )
                    return None

        # ── SL / TP — regime-scaled ───────────────────────────────────────────
        # In trending regime use wider SL (trend can retest)
        # In mean-revert use tighter SL (fast reversion or invalidated)
        if regime == REGIME_TRENDING:
            sl_mult = max(self.sl_vol_mult * 1.2, 4.5)
            tp_mult = max(self.tp_vol_mult * 1.5, 8.0)  # trend trades go further
        elif regime == REGIME_HIGH_VOL:
            sl_mult = 3.0   # tighter SL in high vol
            tp_mult = 7.0   # wider TP — R:R = 2.0
        else:
            sl_mult = self.sl_vol_mult
            tp_mult = self.tp_vol_mult

        sl_dist = max(vol_price * sl_mult, 0.00100)
        tp_dist = max(vol_price * tp_mult, sl_dist * 2.0)

        # Scale TP by signal strength: stronger signal = allow larger target
        tp_dist = tp_dist * (1.0 + 0.10 * (signal_strength - 2.0))
        tp_dist = max(tp_dist, sl_dist * 1.8)

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

# Alias — both names import and instantiate identically.
# MomentumStrategy kept for all existing imports and trade CSV output.
# AlphaStrategy available for any new code going forward.
# When codebase is stable, do a single search-replace to complete the rename.
AlphaStrategy = MomentumStrategy