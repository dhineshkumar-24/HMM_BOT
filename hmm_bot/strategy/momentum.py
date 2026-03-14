"""
strategy/momentum.py — Momentum / Trend-Following Strategy.

Active conditions:
    Session : London OR New York  (09:00–12:00 or 14:00–21:00 broker time)
    Regime  : TRENDING (1)

Entry rules (all must pass):
    1. ADX >= adx_min (18)               (confirmed trend strength)
    2. EMA8 direction                    (price above EMA8 for BUY, below for SELL)
    3. EMA8 > EMA50                      (BUY) or EMA8 < EMA50 (SELL) — baseline trend
    4. Pullback to EMA8                  (close within ATR*0.5 of EMA8)
    5. RSI filter                        (RSI > 40 for BUY, RSI < 60 for SELL)

SL / TP / Trail:
    SL    = entry ± ATR * 1.8
    TP    = entry ± ATR * 3.0
    Trail = ATR * 1.5

Signal format returned:
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

import numpy as np
import pandas as pd
from typing import Optional

from strategy.strategy_base import StrategyBase
from core.hmm_model          import REGIME_TRENDING
from utils.helpers           import SESSION_LONDON, SESSION_NY
from utils.indicators        import (
    compute_atr,
    compute_rsi,
    compute_ema,
    compute_adx,
)
from utils.logger import setup_logger

logger = setup_logger("Momentum")


class MomentumStrategy(StrategyBase):
    """
    Trend-following strategy for trending market regimes.

    Only activates in London or NY sessions when the HMM detects regime 1.
    Requires confirmed ADX, EMA8 trend alignment, and a pullback entry
    to the EMA8 for a high-quality trend continuation entry.
    """

    def __init__(self, config: dict):
        self.config = config
        s  = config["strategy"]
        mo = s["momentum"]

        # Shared
        self.rsi_period   = s["rsi_period"]
        self.atr_period   = s["atr_period"]
        self.adx_period   = s["adx_period"]

        # Strategy-specific
        self.adx_min       = mo["adx_min"]            # ADX must be >= this
        self.ema_fast      = mo["ema_fast"]            # Fast EMA period (8)
        self.ema_slow      = mo["ema_slow"]            # Slow EMA period (50)
        self.rsi_low       = mo["rsi_trend_low"]       # BUY filter: RSI > this
        self.rsi_high      = mo["rsi_trend_high"]      # SELL filter: RSI < this
        self.sl_mult       = mo["sl_atr_mult"]         # SL = ATR * this
        self.tp_mult       = mo["tp_atr_mult"]         # TP = ATR * this
        self.trail_mult    = mo["trail_atr_mult"]      # Trailing = ATR * this

        # Pullback proximity — how close to EMA8 qualifies as a pullback
        self._pullback_atr_ratio = 0.5   # within 0.5 * ATR of EMA8

        logger.info(
            f"MomentumStrategy ready | "
            f"ADX>={self.adx_min} | EMA{self.ema_fast}/{self.ema_slow} | "
            f"RSI({self.rsi_low}/{self.rsi_high}) | "
            f"SL={self.sl_mult}x ATR | TP={self.tp_mult}x ATR"
        )

    # ── StrategyBase interface ─────────────────────────────────────────────────

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich raw OHLCV DataFrame with momentum strategy indicators."""
        df = df.copy()

        # EMAs
        df["ema_fast"]   = compute_ema(df["close"], period=self.ema_fast)
        df["ema_slow"]   = compute_ema(df["close"], period=self.ema_slow)

        # ATR and RSI
        df["atr"]        = compute_atr(df, period=self.atr_period)
        df["rsi"]        = compute_rsi(df["close"], period=self.rsi_period)

        # ADX
        adx_df           = compute_adx(df, period=self.adx_period)
        df["adx"]        = adx_df["adx"]
        df["plus_di"]    = adx_df["plus_di"]
        df["minus_di"]   = adx_df["minus_di"]

        return df

    def generate_signal(
        self,
        df: pd.DataFrame,
        regime: Optional[int] = None,
        session: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Run momentum entry logic on the most recent closed candle.

        Returns a signal dict or None.
        """
        # ── Guard: minimum data ────────────────────────────────────────────────
        if len(df) < 60:
            return None

        # ── Guard: session must be London or New York ──────────────────────────
        if session is not None and session not in (SESSION_LONDON, SESSION_NY):
            return None

        # ── Guard: regime must be trending (or warm-up None) ───────────────────
        if regime is not None and regime != REGIME_TRENDING:
            return None

        prev = df.iloc[-2]   # Confirmed closed candle

        # ── Extract indicator values ───────────────────────────────────────────
        ema_f    = prev.get("ema_fast",  float("nan"))
        ema_s    = prev.get("ema_slow",  float("nan"))
        atr      = prev.get("atr",       float("nan"))
        rsi      = prev.get("rsi",       float("nan"))
        adx      = prev.get("adx",       float("nan"))
        plus_di  = prev.get("plus_di",   float("nan"))
        minus_di = prev.get("minus_di",  float("nan"))
        close    = float(prev["close"])

        if any(map(lambda v: v != v, [ema_f, ema_s, atr, rsi, adx])):
            return None

        entry = float(df.iloc[-1]["close"])

        # ── Filter 1: ADX strength ─────────────────────────────────────────────
        if adx < self.adx_min:
            logger.debug(f"MOM filter: ADX {adx:.1f} < {self.adx_min} → skip")
            return None

        # ── Filter 2 + 3: EMA alignment ───────────────────────────────────────
        # BUY setup: price > EMA8 > EMA50 (uptrend) and +DI > -DI
        # SELL setup: price < EMA8 < EMA50 (downtrend) and -DI > +DI
        bullish_ema  = (ema_f > ema_s) and (plus_di  > minus_di)
        bearish_ema  = (ema_f < ema_s) and (minus_di > plus_di)

        if not bullish_ema and not bearish_ema:
            logger.debug("MOM filter: No EMA alignment → skip")
            return None

        # ── Filter 4: Pullback to EMA8 ─────────────────────────────────────────
        # Entry is valid when price has pulled back close to EMA8
        pullback_zone = atr * self._pullback_atr_ratio
        near_ema_fast = abs(close - ema_f) <= pullback_zone

        if not near_ema_fast:
            logger.debug(
                f"MOM filter: Price {close:.5f} not near EMA{self.ema_fast} "
                f"{ema_f:.5f} (dist={abs(close - ema_f):.5f}, zone={pullback_zone:.5f}) → skip"
            )
            return None

        # ── Filter 5: RSI directional filter ──────────────────────────────────
        direction = None
        reason    = ""

        if bullish_ema and rsi > self.rsi_low:
            direction = "BUY"
            reason    = (
                f"ADX={adx:.1f}>={self.adx_min} | "
                f"EMA{self.ema_fast}={ema_f:.5f}>EMA{self.ema_slow}={ema_s:.5f} | "
                f"Pullback={abs(close-ema_f):.5f} | RSI={rsi:.1f}>{self.rsi_low}"
            )

        elif bearish_ema and rsi < self.rsi_high:
            direction = "SELL"
            reason    = (
                f"ADX={adx:.1f}>={self.adx_min} | "
                f"EMA{self.ema_fast}={ema_f:.5f}<EMA{self.ema_slow}={ema_s:.5f} | "
                f"Pullback={abs(close-ema_f):.5f} | RSI={rsi:.1f}<{self.rsi_high}"
            )

        if direction is None:
            return None

        # ── Compute SL / TP ────────────────────────────────────────────────────
        sl_dist = atr * self.sl_mult
        tp_dist = atr * self.tp_mult

        if direction == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

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
            f"[Momentum] {direction} signal | "
            f"Entry:{entry:.5f} SL:{sl:.5f} TP:{tp:.5f} ATR:{atr:.6f} | "
            f"{reason}"
        )
        return signal
