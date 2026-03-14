"""
strategy/strategy_router.py — HMM Regime + Session Strategy Dispatcher.

Responsibilities:
    1. Detect the active trading session from the candle time
    2. Apply the HMM regime label and confidence gate
    3. Route to the correct sub-strategy based on session + regime
    4. Return a structured signal dict or None

Routing table:
    Session=ASIAN  + Regime=MEAN_REVERT (0) → MeanReversionStrategy
    Session=LONDON + Regime=TRENDING    (1) → MomentumStrategy
    Session=NY     + Regime=TRENDING    (1) → MomentumStrategy
    any            + Regime=HIGH_VOL   (2)  → None (no trade)
    Mismatch (e.g. Asian + Trending)        → None (no trade)
    Warm-up (regime=None)                   → MeanReversion (safe default)

Signal format passed through:
    {
        "direction": "BUY" | "SELL",
        "entry":     float,
        "sl":        float,
        "tp":        float,
        "atr":       float,
        "trail_sl":  float,
        "reason":    str,
    }
"""

from __future__ import annotations

import pandas as pd
from typing import Optional
import numpy as np
from portfolio.signal_combiner import SignalCombiner

from strategy.mean_reversion import MeanReversionStrategy
from strategy.momentum       import MomentumStrategy
from core.hmm_model          import (
    REGIME_MEAN_REVERT,
    REGIME_TRENDING,
    REGIME_HIGH_VOL,
)
from research.alpha.mean_reversion_alpha import volatility_adjusted_zscore
from research.alpha.momentum_alpha import time_series_momentum
from utils.helpers import (
    detect_session,
    check_trading_session,
    SESSION_NONE,
    SESSION_ASIAN,
    SESSION_LONDON,
    SESSION_NY,
)
from utils.logger import setup_logger

logger = setup_logger("StrategyRouter")


class StrategyRouter:
    """
    Routes the current bar to the appropriate sub-strategy based on
    detected session and HMM regime label.

    All sub-strategy instances are created once and reused each bar.
    """

    def __init__(self, config: dict):
        self.config = config

        # Instantiate sub-strategies
        self._mean_rev  = MeanReversionStrategy(config)
        self._momentum  = MomentumStrategy(config)
        self._combiner = SignalCombiner(config) 

        # Routing table: (session, regime) → strategy instance
        # None regime = warm-up → use mean_reversion as safe default
        self._routing: dict[tuple, object] = {
            (SESSION_ASIAN,  REGIME_MEAN_REVERT): self._mean_rev,
            (SESSION_ASIAN,  None):               self._mean_rev,   # warm-up
            (SESSION_LONDON, REGIME_TRENDING):    self._momentum,
            (SESSION_NY,     REGIME_TRENDING):    self._momentum,
        }

        logger.info(
            "StrategyRouter ready | "
            "ASIAN+MeanRevert→MeanRev | LONDON/NY+Trending→Momentum"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def calculate_indicators(
        self,
        df: pd.DataFrame,
        session: Optional[str] = None,
        regime: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Enrich the DataFrame using the strategy relevant to the current context.

        Runs BOTH strategies' indicators so the router can switch mid-day
        without missing warm-up rows. Mean-reversion indicators are always
        computed first (superset of base indicators).

        Args:
            df:      Raw OHLCV DataFrame.
            session: Active session label (optional context hint).
            regime:  HMM regime label (optional context hint).

        Returns:
            Enriched DataFrame.
        """
        # Always compute mean-reversion indicators (they include ATR, RSI, ADX)
        df = self._mean_rev.calculate_indicators(df)
        # Overlay momentum indicators (adds EMA fast/slow — no conflicts)
        df = self._momentum.calculate_indicators(df)
        return df

    def route(
        self,
        df: pd.DataFrame,
        candle_time,
        regime: Optional[int] = None,
        regime_probabilities: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """
        Determine the active session, apply regime routing rules,
        and delegate to the correct sub-strategy.

        Args:
            df:                   Enriched DataFrame with all indicators.
            candle_time:          Datetime of the current (forming) candle.
            regime:               HMM regime label (0, 1, 2, or None = warm-up).
            regime_probabilities: Posterior probability array (not used directly
                                  here — gating is done in hmm_model.should_trade()).

        Returns:
            Signal dict from the selected strategy, or None (no trade).
        """
        # ── Weekend guard ──────────────────────────────────────────────────────
        import datetime
        if isinstance(candle_time, (pd.Timestamp, datetime.datetime)):
            if hasattr(candle_time, 'weekday') and candle_time.weekday() >= 5:
                return None

        # ── Detect active session ──────────────────────────────────────────────
        session = detect_session(self.config, candle_time)

        if session == SESSION_NONE:
            logger.debug(f"No active session at {candle_time} — no trade.")
            return None

        # ── Regime gate: never trade in high-volatility regime ─────────────────
        if regime == REGIME_HIGH_VOL:
            logger.debug(f"High-vol regime — no trade (session={session}).")
            return None

        # ── Look up routing table ──────────────────────────────────────────────
        strategy = self._routing.get((session, regime))

        if strategy is None:
            logger.debug(
                f"No strategy for session={session}, regime={regime} — no trade."
            )
            return None

        # ── Alpha Signal Filtering (via SignalCombiner) ────────────────────────
        combined_scores = self._combiner.combine(df, regime=regime)

        # If combiner has no directional view, skip this bar
        if combined_scores.get("direction") is None:
            logger.debug(
                f"Combiner score={combined_scores['combined']:.3f} — "
                f"below threshold, no trade."
            )
            return None
            atr = df['atr'] if 'atr' in df.columns else (df['high'] - df['low']).rolling(14).mean()
            vaz = volatility_adjusted_zscore(df['close'], atr)
            vaz_thresh = alpha_config.get('vaz_threshold', 1.5)
            if abs(vaz.iloc[-1]) < vaz_thresh:
                logger.debug(f"MeanReversionAlpha (VAZ {abs(vaz.iloc[-1]):.2f}) < {vaz_thresh} — blocked.")
                return None
                
        elif regime == REGIME_TRENDING:
            log_ret = np.log(df['close'] / df['close'].shift(1))
            tsm = time_series_momentum(log_ret)
            tsm_thresh = alpha_config.get('tsm_threshold', 5)
            if abs(tsm.iloc[-1]) < tsm_thresh:
                logger.debug(f"MomentumAlpha (TSM {abs(tsm.iloc[-1]):.2f}) < {tsm_thresh} — blocked.")
                return None

        # ── Generate signal ────────────────────────────────────────────────────
        strategy_name = type(strategy).__name__

        logger.debug(
            f"Routing → {strategy_name} | "
            f"session={session} | regime={regime}"
        )

        signal = strategy.generate_signal(df, regime=regime, session=session)

        if signal:
            signal["strategy"]   = strategy_name
            signal["session"]    = session
            signal["regime"]     = regime
            signal["regime_str"] = (
                {0: "mean_reverting", 1: "trending", 2: "high_vol", None: "warm_up"}
                .get(regime, "unknown")
            )
            logger.info(
                f"Signal: {signal['direction']} | {strategy_name} | "
                f"session={session} | regime={signal['regime_str']} | "
                f"{signal['reason']}"
            )

        return signal

    def reset_state(self) -> None:
        """Reset any stateful flags in sub-strategies (call on daily reset)."""
        # Currently stateless — placeholder for future pending-signal state
        logger.debug("StrategyRouter state reset.")

    # ── Convenience accessors ──────────────────────────────────────────────────

    @property
    def mean_reversion(self) -> MeanReversionStrategy:
        return self._mean_rev

    @property
    def momentum(self) -> MomentumStrategy:
        return self._momentum
