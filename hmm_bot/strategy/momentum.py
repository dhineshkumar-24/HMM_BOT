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

logger = setup_logger("AlphaStrategy")

ACTIVE_SESSIONS = ("london", "newyork", "asian")


class MomentumStrategy(StrategyBase):
    """Regime-adaptive strategy."""

    def __init__(self, config: dict):
        self.config  = config
        alpha_cfg    = config.get("strategy", {}).get("alpha", {})

        self.tp_vol_mult        = alpha_cfg.get("tp_vol_mult", 6.0)
        self.sl_vol_mult        = alpha_cfg.get("sl_vol_mult", 4.0)
        self.min_edge_over_cost = alpha_cfg.get("min_edge_pips", 0.00010)

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
    ) -> Optional[dict]:

        if len(df) < 120:
            return None

        prev = df.iloc[-2]

        alpha_mr  = prev.get("alpha_mr",  float("nan"))
        alpha_mom = prev.get("alpha_mom", float("nan"))
        vol10     = prev.get("vol10",     float("nan"))
        z_vwap    = prev.get("z_vwap",    float("nan"))
        vol_regime = prev.get("vol_regime", float("nan"))

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

        if regime == REGIME_TRENDING:
            # In trending regime: momentum-on-pullback
            # Only enter when a_mom confirms direction AND
            # short-term alpha_mr shows a pullback opportunity
            mode = "momentum_pullback"

            if alpha_mom > 1.5 and alpha_mr > 1.0:
                # 60-min uptrend, 5-min pullback → BUY the dip
                direction = "BUY"
                signal_strength = min(alpha_mr, 5.0)

            elif alpha_mom < -1.5 and alpha_mr < -1.0:
                # 60-min downtrend, 5-min bounce → SELL the rally
                direction = "SELL"
                signal_strength = min(abs(alpha_mr), 5.0)

        elif regime == REGIME_MEAN_REVERT or regime is None:
            # In mean-reverting regime: fade Z-score extremes
            # Only when 60-min momentum is NOT strongly opposing
            mode = "mean_rev"

            if alpha_mr > 2.0 and alpha_mom > -3.0:
                # Price extended DOWN (z_vwap < -2), not in downtrend → BUY
                direction = "BUY"
                signal_strength = min(alpha_mr, 5.0)

            elif alpha_mr < -2.0 and alpha_mom < 3.0:
                # Price extended UP (z_vwap > +2), not in uptrend → SELL
                direction = "SELL"
                signal_strength = min(abs(alpha_mr), 5.0)

        elif regime == REGIME_HIGH_VOL:
            # High vol: only take very high conviction mean reversion
            # Reduce threshold — need stronger signal
            mode = "high_vol_mr"

            if alpha_mr > 3.5 and alpha_mom > -2.0:
                direction = "BUY"
                signal_strength = min(alpha_mr, 5.0)

            elif alpha_mr < -3.5 and alpha_mom < 2.0:
                direction = "SELL"
                signal_strength = min(abs(alpha_mr), 5.0)

        if direction is None:
            return None

        # ── SL / TP — regime-scaled ───────────────────────────────────────────
        # In trending regime use wider SL (trend can retest)
        # In mean-revert use tighter SL (fast reversion or invalidated)
        if regime == REGIME_TRENDING:
            sl_mult = max(self.sl_vol_mult * 1.2, 4.5)
            tp_mult = max(self.tp_vol_mult * 1.5, 8.0)  # trend trades go further
        elif regime == REGIME_HIGH_VOL:
            sl_mult = max(self.sl_vol_mult * 1.5, 5.0)  # wider SL for high vol
            tp_mult = self.tp_vol_mult
        else:
            sl_mult = self.sl_vol_mult
            tp_mult = self.tp_vol_mult

        sl_dist = max(vol_price * sl_mult, 0.00080)
        tp_dist = max(vol_price * tp_mult, sl_dist * 1.8)

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