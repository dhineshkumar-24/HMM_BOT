import pandas as pd
import numpy as np
from typing import Dict

def linear_weighted_combiner(signals: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    """
    1. Linear weighted average
    Signals and weights must be aligned dictionaries.
    """
    combined = None
    for name, series in signals.items():
        weight = weights.get(name, 0.0)
        if combined is None:
            combined = series * weight
        else:
            combined += series * weight
    return combined

def regime_based_combiner(
    signals: Dict[str, pd.Series], 
    regime_series: pd.Series,
    regime_weights: Dict[int, Dict[str, float]]
) -> pd.Series:
    """
    2. Regime-based weighting
    regime_weights format: { regime_id: { signal_name: weight } }
    """
    combined = pd.Series(0.0, index=regime_series.index)
    
    for regime_id, weights in regime_weights.items():
        mask = regime_series == regime_id
        regime_comb = None
        for name, series in signals.items():
            weight = weights.get(name, 0.0)
            if regime_comb is None:
                regime_comb = series * weight
            else:
                regime_comb += series * weight
                
        if regime_comb is not None:
            combined.loc[mask] = regime_comb.loc[mask]
            
    return combined

def risk_adjusted_combiner(
    signals: Dict[str, pd.Series],
    window: int = 100
) -> pd.Series:
    """
    3. Risk-adjusted weighting (Inverse Volatility)
    Weights signals based on their recent variance (lower variance = higher weight).
    """
    combined = None
    inv_vols = {}
    
    for name, series in signals.items():
        vol = series.rolling(window).std()
        inv_vols[name] = 1.0 / vol.replace(0, np.nan)
        
    total_inv_vol = None
    for inv_vol in inv_vols.values():
        if total_inv_vol is None:
            total_inv_vol = inv_vol.fillna(0)
        else:
            total_inv_vol += inv_vol.fillna(0)
            
    for name, series in signals.items():
        weight = inv_vols[name] / total_inv_vol.replace(0, 1)
        if combined is None:
            combined = series * weight.fillna(0)
        else:
            combined += series * weight.fillna(0)
            
    if combined is None:
        # Fallback if dictionary is empty
        return pd.Series(dtype=float)
    return combined

# ─────────────────────────────────────────────────────────────────────────────
# SignalCombiner — class wrapper used by strategy_router.py
# ─────────────────────────────────────────────────────────────────────────────

class SignalCombiner:
    """
    Wraps the three combiner functions into a single class.
    strategy_router.py calls self._combiner.combine(df, regime=regime)
    and gets back a combined score dict.
    """

    def __init__(self, config: dict):
        self.config = config
        self.alpha_cfg = config.get("alpha", {})

    def combine(self, df: pd.DataFrame, regime: int = None) -> dict:
        """
        Run all available alpha signals and combine them.

        Returns:
            dict with keys:
                "mean_reversion" : float score  (-1 to +1)
                "momentum"       : float score  (-1 to +1)
                "combined"       : float score  (-1 to +1)
                "direction"      : "BUY" | "SELL" | None
        """
        import numpy as np

        scores = {}

        # ── Mean reversion score ──────────────────────────────────────────────
        try:
            from research.alpha.mean_reversion_alpha import volatility_adjusted_zscore
            atr = df["atr"] if "atr" in df.columns else (df["high"] - df["low"]).rolling(14).mean()
            vaz = volatility_adjusted_zscore(df["close"], atr)
            last_vaz = float(vaz.iloc[-1]) if not vaz.empty else 0.0
            # Normalise: score = clamp(vaz / 3, -1, 1) — inverted (high Z = sell)
            scores["mean_reversion"] = float(np.clip(-last_vaz / 3.0, -1.0, 1.0))
        except Exception:
            scores["mean_reversion"] = 0.0

        # ── Momentum score ────────────────────────────────────────────────────
        try:
            from research.alpha.momentum_alpha import time_series_momentum
            log_ret = np.log(df["close"] / df["close"].shift(1))
            tsm = time_series_momentum(log_ret)
            last_tsm = float(tsm.iloc[-1]) if not tsm.empty else 0.0
            # Normalise: score = clamp(tsm / 20, -1, 1)
            scores["momentum"] = float(np.clip(last_tsm / 20.0, -1.0, 1.0))
        except Exception:
            scores["momentum"] = 0.0

        # ── Regime-conditional weights ────────────────────────────────────────
        from core.hmm_model import REGIME_MEAN_REVERT, REGIME_TRENDING, REGIME_HIGH_VOL

        if regime == REGIME_MEAN_REVERT:
            weights = {"mean_reversion": 1.0, "momentum": 0.0}
        elif regime == REGIME_TRENDING:
            weights = {"mean_reversion": 0.0, "momentum": 1.0}
        else:
            weights = {"mean_reversion": 0.5, "momentum": 0.5}

        # ── Combine ───────────────────────────────────────────────────────────
        combined = sum(scores.get(k, 0.0) * w for k, w in weights.items())
        combined = float(np.clip(combined, -1.0, 1.0))

        # ── Direction decision ────────────────────────────────────────────────
        threshold = self.alpha_cfg.get("combined_threshold", 0.15)
        if combined >= threshold:
            direction = "BUY"
        elif combined <= -threshold:
            direction = "SELL"
        else:
            direction = None

        return {
            "mean_reversion": scores["mean_reversion"],
            "momentum":       scores["momentum"],
            "combined":       combined,
            "direction":      direction,
        }
