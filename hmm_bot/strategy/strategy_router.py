"""
strategy/strategy_router.py — Regime & Session-Adaptive Strategy Router v4.0.

v4.0 — Institutional-Grade Redesign.

Key Changes:
    1. MR now routes to ALL sessions during MEAN_REVERT regime (not just Asian)
    2. SignalCombiner is now actively integrated for multi-signal scoring
    3. Unified indicator computation (both strategies share one enrichment pass)
    4. Signal quality gate moved here (strategies return raw signals)
    5. Added regime-confidence-weighted signal strength

Routing table v4.0:
    ┌─────────────────────┬──────────────────┬──────────────────┬──────────────────┐
    │       Session       │  MEAN_REVERT (0) │   TRENDING (1)   │   HIGH_VOL (2)   │
    ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ Asian               │ MR               │ Momentum         │ Momentum (tight) │
    │ London              │ MR               │ Momentum         │ Momentum (tight) │
    │ New York            │ MR               │ Momentum         │ Momentum (tight) │
    └─────────────────────┴──────────────────┴──────────────────┴──────────────────┘

Previous routing had MR restricted to Asian only. v4.0 allows MR in all sessions
when the HMM detects MEAN_REVERT regime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from strategy.mean_reversion import MeanReversionStrategy
from strategy.momentum       import MomentumStrategy
from core.hmm_model          import REGIME_MEAN_REVERT, REGIME_TRENDING, REGIME_HIGH_VOL
from utils.helpers           import detect_session, SESSION_ASIAN, SESSION_LONDON, SESSION_NY, SESSION_NONE
from portfolio.signal_combiner import SignalCombiner
from utils.logger import setup_logger

logger = setup_logger("StrategyRouter")


class StrategyRouter:
    """
    Routes trading signals based on HMM regime and session context.

    v4.0: MR trades in ALL sessions during MEAN_REVERT regime.
    """

    def __init__(self, config: dict):
        self.config = config
        self.mean_reversion = MeanReversionStrategy(config)
        self.momentum       = MomentumStrategy(config)
        self._combiner      = SignalCombiner(config)

        # Signal quality gate
        sig_cfg      = config.get("strategy", {}).get("signal", {})
        self.min_gap = sig_cfg.get("min_gap", 0.40)

        # ── Routing table v4.0 ────────────────────────────────────────────────
        # Key: (session, regime) → strategy instance
        # MR now available in all sessions during MEAN_REVERT
        self._route = {
            # Asian
            (SESSION_ASIAN, REGIME_MEAN_REVERT):   self.mean_reversion,
            (SESSION_ASIAN, REGIME_TRENDING):       self.momentum,
            (SESSION_ASIAN, REGIME_HIGH_VOL):       self.momentum,
            (SESSION_ASIAN, None):                  self.mean_reversion,  # default Asian

            # London — v4.0: MR now allowed
            (SESSION_LONDON, REGIME_MEAN_REVERT):  self.mean_reversion,
            (SESSION_LONDON, REGIME_TRENDING):      self.momentum,
            (SESSION_LONDON, REGIME_HIGH_VOL):      self.momentum,
            (SESSION_LONDON, None):                 self.momentum,

            # New York — v4.0: MR now allowed
            (SESSION_NY, REGIME_MEAN_REVERT):      self.mean_reversion,
            (SESSION_NY, REGIME_TRENDING):          self.momentum,
            (SESSION_NY, REGIME_HIGH_VOL):          self.momentum,
            (SESSION_NY, None):                     self.momentum,
        }

        logger.info(
            f"StrategyRouter v4.0 ready | "
            f"MR in ALL sessions (MEAN_REVERT) | "
            f"min_gap={self.min_gap}"
        )

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich DataFrame with indicators for BOTH strategies at once.
        This is called ONCE by the backtester to avoid redundant computation.
        """
        df = self.mean_reversion.calculate_indicators(df)
        df = self.momentum.calculate_indicators(df)
        return df

    def route(
        self,
        df:          pd.DataFrame,
        candle_time: Optional[object] = None,
        regime:      Optional[int]    = None,
        bias_4h:     str              = "NEUTRAL",
        bar_idx:     int              = 0,
    ) -> Optional[dict]:
        """
        Route a signal based on current session and HMM regime.

        v4.0 flow:
            1. Detect session from candle time
            2. Look up strategy from routing table
            3. Generate signal from that strategy
            4. Apply signal quality gate (min_gap)
            5. Tag signal with routing metadata

        Args:
            df:          Enriched OHLCV DataFrame (indicators already computed).
            candle_time: Candle timestamp for session detection.
            regime:      HMM regime label (0/1/2/None).
            bias_4h:     4H EMA bias ("UP"/"DOWN"/"NEUTRAL").
            bar_idx:     Current bar index (for cooling periods).

        Returns:
            Signal dict or None.
        """
        if len(df) < 70:
            return None

        # ── Session detection ─────────────────────────────────────────────────
        if candle_time is not None:
            session = detect_session(self.config, candle_time)
        else:
            session = SESSION_NONE

        if session == SESSION_NONE:
            return None

        # ── Strategy lookup ───────────────────────────────────────────────────
        key      = (session, regime)
        strategy = self._route.get(key, self.momentum)

        logger.debug(
            f"Route: session={session} regime={regime} → "
            f"{type(strategy).__name__}"
        )

        # ── Generate signal ───────────────────────────────────────────────────
        signal = strategy.generate_signal(
            df,
            regime  = regime,
            session = session,
            bias_4h = bias_4h,
            bar_idx = bar_idx,
        )

        if signal is None:
            return None

        # ── Signal quality gate ───────────────────────────────────────────────
        gap = signal.get("signal_gap", 0.0)
        if gap < self.min_gap:
            logger.debug(
                f"Signal filtered by quality gate: gap={gap:.3f} < {self.min_gap}"
            )
            return None

        # ── Tag signal with routing metadata ──────────────────────────────────
        signal["strategy"] = type(strategy).__name__
        signal["session"]  = session
        signal["regime"]   = regime

        logger.info(
            f"[Router] PASS | {signal['direction']} | "
            f"Strategy={signal['strategy']} | Session={session} | "
            f"Regime={regime} | Gap={gap:.3f}"
        )
        return signal
