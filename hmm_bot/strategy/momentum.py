"""
strategy/momentum.py — Statistical Alpha Strategy.

Replaces EMA/RSI/ADX indicator logic with three pure statistical alpha signals:

    Alpha 1 (Mean Reversion):  alpha_mr  = -r5 / vol10
    Alpha 2 (Momentum):        alpha_mom =  r20 / vol50
    Alpha 3 (Vol Expansion):   alpha_vol =  (vol10/vol50) - 1

    Combined: 0.35*z_mr + 0.40*z_mom + 0.25*z_vol

Entry: |combined_alpha| > 80th percentile rolling threshold
Exit:  TP = 2*vol10*price,  SL = 1*vol10*price,  Time stop = 30 bars
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from strategy.strategy_base import StrategyBase
from utils.features         import build_alpha_features
from utils.logger           import setup_logger

logger = setup_logger("AlphaStrategy")

# Sessions this strategy is active in
ACTIVE_SESSIONS = ("london", "newyork", "asian")


class MomentumStrategy(StrategyBase):
    """
    Statistical alpha strategy. Class name kept as MomentumStrategy
    so zero changes needed in strategy_router.py.
    """

    def __init__(self, config: dict):
        self.config = config
        alpha_cfg   = config.get("strategy", {}).get("alpha", {})

        self.alpha_threshold_pct = alpha_cfg.get("threshold_percentile", 80)
        self.tp_vol_mult         = alpha_cfg.get("tp_vol_mult", 2.0)
        self.sl_vol_mult         = alpha_cfg.get("sl_vol_mult", 1.0)
        self.target_risk         = config.get("trading", {}).get("risk_per_trade", 0.01)
        self.min_edge_over_cost  = alpha_cfg.get("min_edge_pips", 0.00015)  # 1.5 pips

        # Rolling alpha history for percentile threshold
        self._alpha_history: list[float] = []
        self._history_window = 200

        logger.info(
            f"AlphaStrategy ready | "
            f"Threshold: {self.alpha_threshold_pct}th percentile | "
            f"TP={self.tp_vol_mult}x vol | SL={self.sl_vol_mult}x vol"
        )

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute statistical alpha features and merge into df."""
        df   = df.copy()
        feats = build_alpha_features(df)

        # Merge alpha columns into main df
        for col in feats.columns:
            df[col] = feats[col]

        return df

    def generate_signal(
        self,
        df:      pd.DataFrame,
        regime:  Optional[int] = None,
        session: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Generate a trade signal based on combined statistical alpha.
        """
        if len(df) < 120:
            return None

        prev = df.iloc[-2]   # last closed bar

        # ── Extract alpha values ───────────────────────────────────────────────
        combined = prev.get("combined_alpha", float("nan"))
        vol10    = prev.get("vol10",          float("nan"))
        z_mr     = prev.get("z_mr",           float("nan"))
        z_mom    = prev.get("z_mom",          float("nan"))
        z_vol    = prev.get("z_vol",          float("nan"))

        if any(np.isnan(v) for v in [combined, vol10, z_mr, z_mom, z_vol]):
            return None

        if vol10 <= 0:
            return None

        # ── Dynamic threshold via rolling percentile ───────────────────────────
        self._alpha_history.append(abs(combined))
        if len(self._alpha_history) > self._history_window:
            self._alpha_history.pop(0)

        if len(self._alpha_history) < 50:
            return None   # need history to calibrate threshold

        threshold = float(np.percentile(
            self._alpha_history, self.alpha_threshold_pct
        ))

        abs_alpha = abs(combined)
        if abs_alpha <= threshold:
            logger.debug(
                f"Alpha {abs_alpha:.3f} <= threshold {threshold:.3f} — skip"
            )
            return None

        # ── Direction ─────────────────────────────────────────────────────────
        direction = "BUY" if combined > 0 else "SELL"

        # ── Volatility-based SL/TP ────────────────────────────────────────────
        entry     = float(df.iloc[-1]["close"])
        price     = float(prev["close"])

        # Convert realized vol (in log-return units) to price units
        vol_price = vol10 * price   # approximate $ move for 1-sigma

        # Enforce minimum move must exceed transaction costs
        if vol_price < self.min_edge_over_cost:
            logger.debug(f"Vol {vol_price:.5f} < min edge — skip")
            return None

        sl_dist = max(vol_price * self.sl_vol_mult, 0.00080)   # min 8 pips
        tp_dist = max(vol_price * self.tp_vol_mult, sl_dist * 1.5)

        if direction == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        reason = (
            f"combined={combined:+.3f} > thr={threshold:.3f} | "
            f"z_mr={z_mr:+.2f} z_mom={z_mom:+.2f} z_vol={z_vol:+.2f} | "
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