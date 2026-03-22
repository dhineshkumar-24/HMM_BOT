"""
core/regime_filters.py — Post-HMM regime validation and risk scaling v4.0.

Applies additional checks after HMM regime detection:
    1. Regime stability — has the regime been consistent for N bars?
    2. Risk scaling     — reduce position size in uncertain/volatile regimes
    3. Confidence gate  — minimum HMM confidence to allow trading
    4. Volatility band  — NEW: only trade when ATR is in P20-P80 range

v4.0 changes:
    - Added volatility_band_filter() for extreme-vol filtering
    - Refined risk scaling multipliers for better capital utilization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from core.hmm_model import REGIME_MEAN_REVERT, REGIME_TRENDING, REGIME_HIGH_VOL
from utils.logger import setup_logger

logger = setup_logger("RegimeFilters")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Regime stability check
# ─────────────────────────────────────────────────────────────────────────────

def is_regime_stable(
    regime_history: list[Optional[int]],
    current_idx: int,
    window: int = 5,
) -> bool:
    """
    Check if the HMM regime has been consistent over the last `window` bars.

    Returns True only if ALL bars in the window agree on the same regime.
    This prevents trading during regime transitions.

    Args:
        regime_history: List of regime labels (0/1/2/None) for each bar.
        current_idx:    Index of the current bar.
        window:         Number of bars to check (default 5).
    """
    if current_idx < window:
        return False

    recent = regime_history[max(0, current_idx - window + 1): current_idx + 1]
    valid  = [r for r in recent if r is not None]

    if len(valid) < window:
        return False

    return len(set(valid)) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Risk scaling by regime
# ─────────────────────────────────────────────────────────────────────────────

def apply_regime_risk_scaling(
    regime:    int,
    base_risk: float,
) -> float:
    """
    Adjust the base risk per trade based on the current HMM regime.

    v4.0 multipliers:
        MEAN_REVERT (0): 1.00× — full risk (strongest edge in ranging markets)
        TRENDING    (1): 0.85× — slightly reduced (momentum can reverse fast)
        HIGH_VOL    (2): 0.60× — significantly reduced (high-vol = high uncertainty)

    Previous v2.2 used 0.75× for trending and 0.5× for high-vol, which was too
    conservative and left alpha on the table in trending markets.

    Args:
        regime:    HMM regime label (0, 1, 2).
        base_risk: Base risk fraction from config (e.g. 0.01).

    Returns:
        Adjusted risk fraction.
    """
    SCALING = {
        REGIME_MEAN_REVERT: 1.00,
        REGIME_TRENDING:    0.85,
        REGIME_HIGH_VOL:    0.60,
    }

    scale = SCALING.get(regime, 1.0)
    adjusted = base_risk * scale

    logger.debug(
        f"Risk scaling: regime={regime} scale={scale:.2f} "
        f"base={base_risk:.4f} → adjusted={adjusted:.4f}"
    )
    return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confidence gate
# ─────────────────────────────────────────────────────────────────────────────

def passes_regime_gate(
    regime:    Optional[int],
    confidence: float,
    min_confidence: float = 0.65,
) -> bool:
    """
    Check if the HMM regime classification has sufficient confidence to trade.

    Args:
        regime:         Classified regime (0/1/2/None).
        confidence:     HMM posterior probability of the predicted regime.
        min_confidence: Minimum acceptable confidence (default 0.65 = 65%).

    Returns:
        True if the regime is known and confident enough to trade.
    """
    if regime is None:
        logger.debug("Regime gate: regime=None → FAIL (warm-up)")
        return False

    if confidence < min_confidence:
        logger.debug(
            f"Regime gate: confidence={confidence:.2f} < {min_confidence:.2f} → FAIL"
        )
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# 4. Volatility band filter (NEW v4.0)
# ─────────────────────────────────────────────────────────────────────────────

def volatility_band_filter(
    atr_percentile: float,
    low_cutoff:  float = 0.15,
    high_cutoff: float = 0.85,
) -> bool:
    """
    Only trade when ATR is within a normal volatility band.

    Rationale:
        - Extremely low vol (< P15): spreads dominate profits, poor risk-reward
        - Extremely high vol (> P85): stop-loss slippage risk, unpredictable
        - Normal vol (P15-P85): best regime for systematic strategies

    Args:
        atr_percentile: ATR's rolling percentile rank (0.0 - 1.0).
        low_cutoff:     Minimum ATR percentile to trade (default 0.15 = P15).
        high_cutoff:    Maximum ATR percentile to trade (default 0.85 = P85).

    Returns:
        True if volatility is within the acceptable band.
    """
    if np.isnan(atr_percentile):
        return True  # allow trading if percentile not computed yet

    if atr_percentile < low_cutoff:
        logger.debug(f"Vol band: pctile={atr_percentile:.2f} < {low_cutoff} → SKIP (too quiet)")
        return False

    if atr_percentile > high_cutoff:
        logger.debug(f"Vol band: pctile={atr_percentile:.2f} > {high_cutoff} → SKIP (too volatile)")
        return False

    return True
