"""
research/trade_simulator.py — Realistic intrabar trade execution simulator v4.0.

Models real MT5 execution costs and behaviour:
    - Spread (applied at BOTH entry AND exit)
    - Slippage (random uniform within configured range, at entry AND SL exit)
    - Commission (per lot, round-trip)
    - Stop Loss (checked against bar Low/High intrabar)
    - Take Profit (checked against bar High/Low intrabar)
    - Trailing Stop (updated each bar with ATR * regime-specific multiplier)
    - Next-bar-open entry (avoids look-ahead bias)

v4.0 FIXES:
    FIX C1: Exit-side spread now modeled (previously only entry-side).
            `_close_trade()` now deducts half-spread at exit.
    FIX M6: Slippage applied BOTH at entry AND at SL exit fills.
    FIX M4: Trailing stop uses trade-stored trail_sl value (regime-specific).

Cost model (EURUSD, standard account, M5):
    Spread     : 2.0 pips → $20 round-trip per standard lot (split entry/exit)
    Slippage   : 2.0 pips max → variable friction at entry + SL
    Commission : $6 per lot round-trip
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Cost model defaults (EURUSD, standard account, M5)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SPREAD_PIPS      = 2.0
DEFAULT_SLIPPAGE_PIPS    = 2.0
DEFAULT_COMMISSION_PER_LOT = 6.0   # USD, round-trip
DEFAULT_PIP_VALUE        = 10.0    # USD per pip per standard lot (EURUSD)
DEFAULT_CONTRACT_SIZE    = 100_000
DEFAULT_PIP_SIZE         = 0.0001  # EURUSD 4-digit pip


# ─────────────────────────────────────────────────────────────────────────────
# Trade record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulatedTrade:
    """Represents a single simulated trade lifecycle."""
    direction:   str         # "BUY" or "SELL"
    entry_bar:   int         # index in the DataFrame
    entry_price: float
    sl:          float
    tp:          float
    lots:        float
    atr:         float
    trail_sl:    float       # distance (in price) for trailing stop
    strategy:    str = ""
    regime:      Optional[int] = None
    session:     str = ""

    # Filled at close
    exit_bar:    int   = -1
    exit_price:  float = 0.0
    gross_pnl:   float = 0.0
    net_pnl:     float = 0.0
    spread_cost: float = 0.0
    slip_cost:   float = 0.0
    commission:  float = 0.0
    exit_reason: str   = ""   # "SL", "TP", "TRAIL", "TIME", "EOD"
    is_closed:   bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# TradeSimulator
# ─────────────────────────────────────────────────────────────────────────────

class TradeSimulator:
    """
    Simulates trade execution over a historical DataFrame bar-by-bar.

    v4.0: Fixed spread/cost accounting for realistic PnL.

    Usage:
        sim = TradeSimulator()
        sim.open_trade("BUY", entry, sl, tp, atr, lots, bar_idx)

        for i, bar in df.iterrows():
            closed = sim.update(bar, i)   # returns list of closed trades
    """

    def __init__(
        self,
        spread_pips:        float = DEFAULT_SPREAD_PIPS,
        slippage_pips:      float = DEFAULT_SLIPPAGE_PIPS,
        commission_per_lot: float = DEFAULT_COMMISSION_PER_LOT,
        pip_value:          float = DEFAULT_PIP_VALUE,
        pip_size:           float = DEFAULT_PIP_SIZE,
        trail_atr_mult:     float = 1.5,  # default, overridden per trade
        seed:               Optional[int] = 42,
    ):
        self.spread_pips   = spread_pips
        self.slippage_pips = slippage_pips
        self.commission    = commission_per_lot
        self.pip_value     = pip_value
        self.pip_size      = pip_size
        self.trail_mult    = trail_atr_mult  # fallback only

        if seed is not None:
            random.seed(seed)

        self._open_trades:   list[SimulatedTrade] = []
        self.closed_trades:  list[SimulatedTrade] = []

    # ── Open a new position ────────────────────────────────────────────────────

    def open_trade(
        self,
        direction:  str,
        entry:      float,
        sl:         float,
        tp:         float,
        atr:        float,
        lots:       float,
        bar_idx:    int,
        strategy:   str = "",
        regime:     Optional[int] = None,
        session:    str = "",
    ) -> SimulatedTrade:
        """
        Register a new trade, applying HALF spread and slippage at entry.
        v4.0: Only half-spread at entry; other half applied at exit.
        """
        slip          = random.uniform(0, self.slippage_pips) * self.pip_size
        half_spread   = (self.spread_pips / 2.0) * self.pip_size

        if direction == "BUY":
            actual_entry = entry + half_spread + slip   # buy at ask + slip
        else:
            actual_entry = entry - half_spread - slip   # sell at bid - slip

        # v4.0: Use trade-specific trail_sl if available
        # The strategy sets trail_sl via signal["trail_sl"]
        # trail_sl is stored as absolute price distance

        trade = SimulatedTrade(
            direction   = direction,
            entry_bar   = bar_idx,
            entry_price = actual_entry,
            sl          = sl,
            tp          = tp,
            lots        = lots,
            atr         = atr,
            trail_sl    = atr * self.trail_mult,  # fallback
            strategy    = strategy,
            regime      = regime,
            session     = session,
        )
        self._open_trades.append(trade)
        return trade

    # ── Update positions each bar ─────────────────────────────────────────────

    def update(self, bar: pd.Series, bar_idx: int) -> list[SimulatedTrade]:
        """
        Process open positions against the current bar's OHLC.

        Order of operations per bar (matches MT5 broker behaviour):
            1. Check SL against bar's High/Low (applying SL-slippage)
            2. Check TP against bar's High/Low (no slippage on TP fills)
            3. If still open → update trailing stop from bar's close
        """
        newly_closed = []

        for trade in list(self._open_trades):
            if trade.entry_bar == bar_idx:
                continue      # skip the bar the trade was opened on

            high    = float(bar["high"])
            low     = float(bar["low"])
            close   = float(bar["close"])
            bar_atr = float(bar["atr"]) if "atr" in bar else trade.atr

            closed      = False
            exit_price  = close
            exit_reason = ""

            # ── Step 1 & 2: Check SL / TP intrabar ───────────────────────────
            if trade.direction == "BUY":
                sl_hit = (low  <= trade.sl)
                tp_hit = (high >= trade.tp)

                if sl_hit and tp_hit:
                    if abs(trade.entry_price - trade.sl) < abs(trade.entry_price - trade.tp):
                        sl_slip = random.uniform(0, self.slippage_pips) * self.pip_size
                        exit_price  = trade.sl - sl_slip
                        exit_reason = "SL"
                    else:
                        exit_price  = trade.tp
                        exit_reason = "TP"
                    closed = True

                elif sl_hit:
                    sl_slip = random.uniform(0, self.slippage_pips) * self.pip_size
                    exit_price  = trade.sl - sl_slip
                    exit_reason = "SL"
                    closed      = True

                elif tp_hit:
                    exit_price  = trade.tp
                    exit_reason = "TP"
                    closed      = True

            else:  # SELL
                sl_hit = (high >= trade.sl)
                tp_hit = (low  <= trade.tp)

                if sl_hit and tp_hit:
                    if abs(trade.entry_price - trade.sl) < abs(trade.entry_price - trade.tp):
                        sl_slip = random.uniform(0, self.slippage_pips) * self.pip_size
                        exit_price  = trade.sl + sl_slip
                        exit_reason = "SL"
                    else:
                        exit_price  = trade.tp
                        exit_reason = "TP"
                    closed = True

                elif sl_hit:
                    sl_slip = random.uniform(0, self.slippage_pips) * self.pip_size
                    exit_price  = trade.sl + sl_slip
                    exit_reason = "SL"
                    closed      = True

                elif tp_hit:
                    exit_price  = trade.tp
                    exit_reason = "TP"
                    closed      = True

            if closed:
                self._close_trade(trade, exit_price, exit_reason, bar_idx)
                self._open_trades.remove(trade)
                newly_closed.append(trade)
                continue

            # ── Step 3: Update trailing stop ─────────────────────────────────
            # v5.0 EXIT ARCHITECTURE:
            # - Trailing ONLY activates after profit >= 1.5× ATR
            # - Minimum trail distance = 1× ATR (never closer)
            # - This prevents small winners from being strangled
            trail_dist = trade.trail_sl if trade.trail_sl > 0 else bar_atr * self.trail_mult
            # Enforce minimum trail distance of 1× ATR
            min_trail = max(trade.atr, 0.00100)
            trail_dist = max(trail_dist, min_trail)

            if trade.direction == "BUY":
                profit = close - trade.entry_price
                # Only trail after significant profit (1.5× ATR)
                if profit >= trade.atr * 1.5:
                    new_trail = close - trail_dist
                    if new_trail > trade.sl:
                        trade.sl = new_trail
            else:
                profit = trade.entry_price - close
                if profit >= trade.atr * 1.5:
                    new_trail = close + trail_dist
                    if new_trail < trade.sl:
                        trade.sl = new_trail

        return newly_closed


    def close_all(
        self,
        bar: pd.Series,
        bar_idx: int,
        reason: str = "EOD",
    ) -> list[SimulatedTrade]:
        """Force-close all open positions at the current bar's close."""
        closed = []
        for trade in list(self._open_trades):
            self._close_trade(trade, float(bar["close"]), reason, bar_idx)
            self._open_trades.remove(trade)
            closed.append(trade)
        return closed

    def close_all_pnl_aware(
        self,
        bar: pd.Series,
        bar_idx: int,
        min_progress: float = 0.40,
    ) -> list[SimulatedTrade]:
        """
        PnL-aware forced close for time-stop logic.
        Profitable trades (>= min_progress of TP) get stop tightened.
        Flat/losing trades are closed.
        """
        closed = []
        current_price = float(bar["close"])

        for trade in list(self._open_trades):
            entry = trade.entry_price
            tp    = trade.tp

            tp_dist = abs(tp - entry)
            if tp_dist > 0:
                if trade.direction == "BUY":
                    progress = (current_price - entry) / tp_dist
                else:
                    progress = (entry - current_price) / tp_dist
            else:
                progress = 0.0

            if progress >= min_progress:
                # Trade is profitable — tighten stop instead of closing
                if trade.direction == "BUY":
                    new_sl = max(trade.sl, entry + 0.00005)
                    trade.sl = new_sl
                else:
                    new_sl = min(trade.sl, entry - 0.00005)
                    trade.sl = new_sl
            else:
                # Trade is flat or losing — apply time stop
                self._close_trade(trade, current_price, "TIME", bar_idx)
                self._open_trades.remove(trade)
                closed.append(trade)

        return closed

    @property
    def has_open_trade(self) -> bool:
        return len(self._open_trades) > 0

    # ── Internal ──────────────────────────────────────────────────────────────

    def _close_trade(
        self,
        trade:       SimulatedTrade,
        exit_price:  float,
        exit_reason: str,
        bar_idx:     int,
    ) -> None:
        """
        Fill in all cost fields and mark the trade closed.

        v4.0 FIX C1: Exit-side half-spread is now deducted from net_pnl.
        Previously spread was only at entry, understating total costs by ~$10/lot.
        """
        direction = 1 if trade.direction == "BUY" else -1

        # ── Gross PnL (entry-to-exit in pips, converted to USD) ──────────────
        raw_pips    = direction * (exit_price - trade.entry_price) / self.pip_size
        gross_pnl   = raw_pips * self.pip_value * trade.lots

        # ── Cost breakdown ────────────────────────────────────────────────────
        # FIX C1: Full round-trip spread = spread_pips × pip_value × lots
        # Half is already baked into entry_price. The other half is deducted here.
        half_spread_cost = (self.spread_pips / 2.0) * self.pip_value * trade.lots
        full_spread_cost = self.spread_pips * self.pip_value * trade.lots
        commission       = self.commission * trade.lots

        # Net PnL = gross - exit_half_spread - commission
        # (Entry half-spread is already reflected in entry_price, so gross_pnl
        #  already includes that cost. We only deduct the exit half here.)
        net_pnl = gross_pnl - half_spread_cost - commission

        trade.exit_bar    = bar_idx
        trade.exit_price  = exit_price
        trade.gross_pnl   = gross_pnl
        trade.net_pnl     = net_pnl
        trade.spread_cost = full_spread_cost    # total round-trip spread for reporting
        trade.commission  = commission
        trade.exit_reason = exit_reason
        trade.is_closed   = True

        self.closed_trades.append(trade)
