"""
strategy/mean_reversion.py — Precision Mean Reversion Strategy v5.0

CRITICAL FIXES:
    The previous strategy NEVER traded (0 MR trades in 238 total).
    The MR regime was never detected by HMM, so all trades routed to momentum.

v5.0 — Redesigned for reliable entries and survivable SL:

    1. SL minimum 15 pips (was 10 — within EURUSD M5 noise floor)
    2. TP targets realistic 2× SL (was 4× → only 4% TP hit rate)
    3. OU half-life gating remains but with relaxed threshold (50 bars)
    4. Dual Z-score with higher trigger (2.0) for entry precision
    5. Bar confirmation + RSI confirmation
    6. Anti-noise ATR filter
    7. Achievable RR ratio (1:2 minimum, not 1:4)

Trade thesis:
    Enter when price is statistically overextended (|Z| > 2σ) in a
    confirmed mean-reverting regime, with OU half-life confirming
    fast reversion expected. Target 50% reversion to VWAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from strategy.strategy_base import StrategyBase
from core.hmm_model          import REGIME_MEAN_REVERT
from utils.features          import ou_half_life
from utils.indicators        import (
    compute_vwap,
    compute_atr,
    compute_rsi,
    compute_ema,
    compute_ema_slope,
    compute_adx,
)
from utils.logger import setup_logger

logger = setup_logger("MeanReversion")


class MeanReversionStrategy(StrategyBase):
    """
    Precision mean-reversion strategy for ranging market regimes.
    v5.0: Wider SL above noise floor, achievable TP targets.
    """

    def __init__(self, config: dict):
        self.config = config
        s  = config["strategy"]
        mr = s["mean_reversion"]

        # Shared
        self.rsi_period      = s["rsi_period"]
        self.atr_period      = s["atr_period"]
        self.adx_period      = s["adx_period"]
        self.vwap_window     = s.get("vwap_window", 21)
        self.vwap_window_slow = s.get("vwap_window_slow", 50)

        # Mean-reversion specific
        self.z_trigger       = mr.get("z_score_trigger", 2.0)    # v5.0: raised to 2.0
        self.z_slow_trigger  = mr.get("z_slow_trigger", 1.2)     # v5.0: raised to 1.2
        self.rsi_ob          = mr["rsi_overbought"]
        self.rsi_os          = mr["rsi_oversold"]
        self.ema_flat_thr    = mr["ema_slope_flat"]
        self.adx_max         = mr.get("adx_max", 22)             # v5.0: tighter (was 25)
        self.sl_mult         = mr.get("sl_atr_mult", 2.5)
        self.min_sl_pips     = mr.get("min_sl_pips", 0.00150)    # v5.0: 15-pip floor!
        self.max_sl_pips     = mr.get("max_sl_pips", 0.00350)    # 35-pip ceiling
        self.trail_mult      = mr.get("trail_atr_mult", 1.8)
        self.min_rr          = mr.get("min_rr", 2.0)             # v5.0: strict 1:2
        self.min_tp_atr_mult = mr.get("min_tp_atr_mult", 3.0)    # v5.0: raised
        self.bar_confirm     = mr.get("bar_confirm", True)
        self.cooling_bars    = mr.get("cooling_bars", 6)          # v5.0: longer cooling
        self.min_gap         = s.get("signal", {}).get("min_gap", 0.50)
        self.min_atr_ratio   = mr.get("min_atr_ratio", 0.6)     # anti-noise

        # OU half-life gate
        self.ou_max_hl       = mr.get("ou_max_half_life", 50)    # v5.0: relaxed to 50

        # Internal state
        self._last_sl_bar: dict[str, int] = {"BUY": -999, "SELL": -999}

        logger.info(
            f"MeanReversionStrategy v5.0 | "
            f"Z_fast>={self.z_trigger} | Z_slow>={self.z_slow_trigger} | "
            f"ADX<{self.adx_max} | SL_floor={self.min_sl_pips*10000:.0f}pips | "
            f"min_RR={self.min_rr} | OU_max_hl={self.ou_max_hl}"
        )

    def register_sl_hit(self, direction: str, bar_idx: int) -> None:
        """Called by backtester when a SL hit occurs → starts cooling timer."""
        self._last_sl_bar[direction] = bar_idx

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich raw OHLCV with mean-reversion indicators including OU half-life."""
        df = df.copy()

        log_ret       = np.log(df["close"] / df["close"].shift(1))
        df["returns"] = log_ret

        # ── Dual Z-score ──────────────────────────────────────────────────────
        vwap_fast        = compute_vwap(df, window=self.vwap_window)
        roll_std_fast    = df["close"].rolling(self.vwap_window).std()
        df["vwap"]       = vwap_fast
        df["z_score"]    = (df["close"] - vwap_fast) / roll_std_fast.replace(0, np.nan)

        vwap_slow        = compute_vwap(df, window=self.vwap_window_slow)
        roll_std_slow    = df["close"].rolling(self.vwap_window_slow).std()
        df["vwap_slow"]  = vwap_slow
        df["z_score_slow"] = (df["close"] - vwap_slow) / roll_std_slow.replace(0, np.nan)

        # ── ATR + rolling average ─────────────────────────────────────────────
        df["atr"]     = compute_atr(df, period=self.atr_period)
        df["atr_avg"] = df["atr"].rolling(50).mean()

        # ── RSI ───────────────────────────────────────────────────────────────
        df["rsi"] = compute_rsi(df["close"], period=self.rsi_period)

        # ── EMA50 slope ───────────────────────────────────────────────────────
        df["ema50"]       = compute_ema(df["close"], period=50)
        df["ema50_slope"] = compute_ema_slope(df["close"], ema_period=50, slope_window=5)

        # ── ADX ───────────────────────────────────────────────────────────────
        adx_df         = compute_adx(df, period=self.adx_period)
        df["adx"]      = adx_df["adx"]
        df["plus_di"]  = adx_df["plus_di"]
        df["minus_di"] = adx_df["minus_di"]

        # ── OU Half-Life ──────────────────────────────────────────────────────
        df["ou_half_life"] = ou_half_life(df["close"], window=60)

        return df

    def generate_signal(
        self,
        df:      pd.DataFrame,
        regime:  Optional[int] = None,
        session: Optional[str] = None,
        bias_4h: str = "NEUTRAL",
        bar_idx: int = 0,
    ) -> Optional[dict]:
        """
        Generate PRECISION mean-reversion entry signal.

        v5.0 entry pipeline (ALL must pass):
            1. Data sufficiency (120 bars minimum)
            2. ADX < 22 (not trending — this IS the regime filter even without HMM)
            3. EMA50 slope is flat
            4. Anti-noise: ATR > 0.6× average
            5. OU half-life < 50 bars (fast MR confirmed)
            6. |Z_fast| >= 2.0 + RSI overextension
            7. Z_slow agrees with Z_fast direction
            8. Bar confirmation (reversal candle)
            9. Post-SL cooling period
            10. Signal gap > threshold
        """
        if len(df) < 120:
            return None

        # Update bar counter
        prev  = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) >= 3 else prev
        entry = float(df.iloc[-1]["close"])

        # ── Extract values ─────────────────────────────────────────────────────
        z_fast   = float(prev.get("z_score",      float("nan")))
        z_slow   = float(prev.get("z_score_slow", float("nan")))
        rsi      = float(prev.get("rsi",          float("nan")))
        atr      = float(prev.get("atr",          float("nan")))
        atr_avg  = float(prev.get("atr_avg",      float("nan")))
        adx      = float(prev.get("adx",          float("nan")))
        slope    = float(prev.get("ema50_slope",  float("nan")))
        vwap     = float(prev.get("vwap",         float("nan")))
        ou_hl    = float(prev.get("ou_half_life", float("nan")))

        bar_open  = float(prev.get("open",  float("nan")))
        bar_close = float(prev.get("close", float("nan")))

        if any(np.isnan(v) for v in [z_fast, z_slow, rsi, atr, adx, slope, vwap]):
            return None

        if atr <= 0:
            return None

        # ── FILTER 1: EMA50 must be flat (not trending) ──────────────────────
        if abs(slope) > self.ema_flat_thr:
            return None

        # ── FILTER 2: ADX must be low (confirms ranging market) ───────────────
        if adx >= self.adx_max:
            return None

        # ── FILTER 3: Anti-noise — enough volatility for edge ─────────────────
        if atr_avg > 0 and not np.isnan(atr_avg) and atr < atr_avg * self.min_atr_ratio:
            return None

        # ── FILTER 4: OU half-life must indicate fast MR ──────────────────────
        if not np.isnan(ou_hl) and ou_hl > self.ou_max_hl:
            return None

        # ── Determine direction from Z-score + RSI ────────────────────────────
        direction = None

        if z_fast >= self.z_trigger and rsi > self.rsi_ob:
            direction = "SELL"    # overbought → expect reversion down
        elif z_fast <= -self.z_trigger and rsi < self.rsi_os:
            direction = "BUY"     # oversold → expect reversion up

        if direction is None:
            return None

        # ── FILTER 5: Slow Z-score must AGREE ─────────────────────────────────
        if direction == "SELL" and z_slow < self.z_slow_trigger:
            return None
        if direction == "BUY" and z_slow > -self.z_slow_trigger:
            return None

        # ── FILTER 6: Bar confirmation ────────────────────────────────────────
        if self.bar_confirm:
            if direction == "BUY" and bar_close < bar_open:
                return None   # need bullish reversal bar
            if direction == "SELL" and bar_close > bar_open:
                return None   # need bearish reversal bar

        # ── FILTER 7: Post-SL cooling period ──────────────────────────────────
        bars_since_sl = bar_idx - self._last_sl_bar.get(direction, -999)
        if bars_since_sl < self.cooling_bars:
            return None

        # ── Signal gap ────────────────────────────────────────────────────────
        signal_gap = abs(z_fast) - self.z_trigger
        if signal_gap < self.min_gap:
            return None

        # ── Compute SL / TP — WIDE SL above noise floor ──────────────────────
        sl_dist = max(atr * self.sl_mult, self.min_sl_pips)
        sl_dist = min(sl_dist, self.max_sl_pips)

        # TP: target reversion toward VWAP but with minimum RR enforcement
        vwap_dist = abs(entry - vwap)
        tp_ou = vwap_dist * 0.6   # target 60% reversion to mean
        tp_dist = max(tp_ou, atr * self.min_tp_atr_mult)
        tp_dist = max(tp_dist, sl_dist * self.min_rr)  # enforce 1:2 RR

        if direction == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        # Final RR enforcement
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < self.min_rr:
            return None

        trail_dist = atr * self.trail_mult
        ou_hl_str = f"{ou_hl:.1f}" if not np.isnan(ou_hl) else "N/A"

        reason = (
            f"Z_fast={z_fast:.2f} Z_slow={z_slow:.2f} | RSI={rsi:.1f} | "
            f"ADX={adx:.1f} | slope={slope:.6f} | OU_hl={ou_hl_str} | "
            f"ATR={atr*10000:.1f}pips | SL={sl_dist*10000:.1f}pips | "
            f"Gap={signal_gap:.2f}"
        )

        signal = {
            "direction":     direction,
            "entry":         round(entry, 5),
            "sl":            round(sl, 5),
            "tp":            round(tp, 5),
            "atr":           round(float(atr), 6),
            "trail_sl":      round(trail_dist, 6),
            "breakeven_atr": round(atr, 6),
            "signal_gap":    round(signal_gap, 3),
            "reason":        reason,
        }

        logger.info(
            f"[MeanReversion] {direction} | Entry:{entry:.5f} "
            f"SL:{sl:.5f}({sl_dist*10000:.0f}pips) "
            f"TP:{tp:.5f}({tp_dist*10000:.0f}pips) RR:{rr:.2f} | "
            f"OU_hl={ou_hl_str} | {reason}"
        )
        return signal
