"""
core/regime_filters.py — Post-HMM regime validation and risk scaling.

Applies additional signal confirmation on top of the raw HMM regime label.
Prevents trading during unstable regime transitions.

PLACEHOLDER: Methods are fully documented and structurally complete.
Business logic will be wired in during Phase 2 (HMM integration).
"""

from __future__ import annotations

import numpy as np
from utils.logger import setup_logger

logger = setup_logger("RegimeFilters")

# Regime label constants (matches hmm: regime_names in settings.yaml)
REGIME_MEAN_REVERT = 0
REGIME_TRENDING    = 1
REGIME_NOISY       = 2

# Risk scaling per regime — multiplied against base risk_per_trade
REGIME_RISK_SCALE: dict[int, float] = {
    REGIME_MEAN_REVERT: 1.0,   # Full risk in mean-reverting regime
    REGIME_TRENDING:    0.75,  # Reduced risk — trend strategies less certain
    REGIME_NOISY:       0.0,   # No trading in noisy/uncertain regime
}


def is_regime_stable(
    regime_history: list[int],
    window: int = 5,
) -> bool:
    """
    Check whether the regime has been consistent over the last `window` bars.

    A regime is considered stable if all bars in the window share the same
    label. This prevents trading immediately after a regime transition.

    Args:
        regime_history: List of integer regime labels, most recent last.
        window:         Number of recent bars to check for consistency.

    Returns:
        True if stable, False during transitions.
    """
    if len(regime_history) < window:
        logger.debug("Not enough regime history to evaluate stability.")
        return False

    recent = regime_history[-window:]
    stable = len(set(recent)) == 1

    if not stable:
        logger.info(
            f"Regime unstable over last {window} bars: {recent}. Skipping signal."
        )
    return stable


def apply_regime_risk_scaling(
    regime: int,
    base_risk: float,
) -> float:
    """
    Scale the base risk percentage based on the current regime.

    Args:
        regime:    Current regime label (0, 1, or 2).
        base_risk: Base risk fraction from config (e.g. 0.01 = 1%).

    Returns:
        Adjusted risk fraction. Returns 0.0 for the noisy regime.
    """
    scale = REGIME_RISK_SCALE.get(regime, 0.0)
    adjusted = base_risk * scale

    logger.debug(
        f"Regime {regime} → risk scale {scale:.2f} "
        f"| base {base_risk:.3f} → adjusted {adjusted:.3f}"
    )
    return adjusted


def passes_regime_gate(
    regime: int,
    regime_probabilities: np.ndarray,
    confidence_threshold: float = 0.65,
) -> bool:
    """
    Final gate: only act if the HMM posterior probability of the predicted
    regime exceeds the confidence threshold.

    Args:
        regime:                 Predicted regime label.
        regime_probabilities:   Posterior probability array from HMM.
        confidence_threshold:   Minimum confidence required (from config).

    Returns:
        True if confidence is sufficient to trade.
    """
    if regime >= len(regime_probabilities):
        logger.warning(f"Regime {regime} out of range for probability array.")
        return False

    confidence = float(regime_probabilities[regime])

    if confidence < confidence_threshold:
        logger.info(
            f"Regime {regime} confidence {confidence:.2%} below "
            f"threshold {confidence_threshold:.2%}. Skipping."
        )
        return False

    return True
