"""
research/trade_simulator.py — Realistic intrabar trade execution simulator.

Models real MT5 execution costs and behaviour:
    - Spread (applied at entry)
    - Slippage (random uniform within configured range)
    - Commission (per lot, round-trip)
    - Stop Loss (checked against bar Low/High intrabar)
    - Take Profit (checked against bar High/Low intrabar)
    - Trailing Stop (updated each bar with ATR * multiplier)
    - Next-bar-open entry (avoids look-ahead bias)

Cost model defaults (Step 8):
    Spread     : 1.5 pips → $15 per standard lot
    Slippage   : 0.5 pips → $5 additional friction
    Commission : $6 per lot round-trip

All prices are in instrument points (pips * pip_value).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Cost model defaults (EURUSD, standard account, M1)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SPREAD_PIPS      = 1.5
DEFAULT_SLIPPAGE_PIPS    = 0.5
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
    exit_reason: str   = ""   # "SL", "TP", "TRAIL", "EOD"
    is_closed:   bool  = False


# ─────────────────────────────────────────────────────────────────────────────
# TradeSimulator
# ─────────────────────────────────────────────────────────────────────────────

class TradeSimulator:
    """
    Simulates trade execution over a historical DataFrame bar-by-bar.

    Usage:
        sim = TradeSimulator(config)
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
        trail_atr_mult:     float = 1.5,
        seed:               Optional[int] = 42,
    ):
        self.spread_pips   = spread_pips
        self.slippage_pips = slippage_pips
        self.commission    = commission_per_lot
        self.pip_value     = pip_value
        self.pip_size      = pip_size
        self.trail_mult    = trail_atr_mult

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
        Register a new trade, applying spread and slippage at entry.

        Args:
            direction: "BUY" or "SELL".
            entry:     Raw entry price (next bar open).
            sl:        Stop-loss level.
            tp:        Take-profit level.
            atr:       ATR at time of signal (for trailing stop).
            lots:      Position size in standard lots.
            bar_idx:   Index of the entry bar in the DataFrame.

        Returns:
            The opened SimulatedTrade.
        """
        slip = random.uniform(0, self.slippage_pips) * self.pip_size
        spread = self.spread_pips * self.pip_size

        if direction == "BUY":
            actual_entry = entry + spread / 2 + slip   # buy at ask + slip
        else:
            actual_entry = entry - spread / 2 - slip   # sell at bid - slip

        trail_dist = atr * self.trail_mult

        trade = SimulatedTrade(
            direction   = direction,
            entry_bar   = bar_idx,
            entry_price = actual_entry,
            sl          = sl,
            tp          = tp,
            lots        = lots,
            atr         = atr,
            trail_sl    = trail_dist,
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
            1. Check SL against bar's High/Low (using SL from PREVIOUS bar)
            2. Check TP against bar's High/Low
            3. If still open → update trailing stop from bar's close
               (trail takes effect from the NEXT bar)

        Args:
            bar:     Current bar (must have 'open', 'high', 'low', 'close', 'atr').
            bar_idx: Index of this bar.

        Returns:
            List of trades closed this bar.
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

            # ── Step 1 & 2: Check SL / TP against this bar using CURRENT SL
            # (BEFORE trail update — trail is applied at end of bar)
            if low <= trade.sl and high >= trade.tp:
                if abs(trade.entry_price - trade.sl) < abs(trade.entry_price - trade.tp):
                    exit_price = trade.sl
                    exit_reason = "SL"
                else:
                    exit_price = trade.tp
                    exit_reason = "TP"
                closed = True
            elif low <= trade.sl:
                exit_price = trade.sl
                exit_reason = "SL"
                closed = True
            elif high >= trade.tp:
                exit_price = trade.tp
                exit_reason = "TP"
                closed = True
            else:   # SELL
                if high >= trade.sl:
                    exit_price  = trade.sl
                    exit_reason = "SL"
                    closed = True
                elif low <= trade.tp:
                    exit_price  = trade.tp
                    exit_reason = "TP"
                    closed = True

            if closed:
                self._close_trade(trade, exit_price, exit_reason, bar_idx)
                self._open_trades.remove(trade)
                newly_closed.append(trade)
                continue

            # ── Step 3: Update trailing stop (takes effect next bar) ──────────
            if trade.direction == "BUY":
                new_trail = close - bar_atr * self.trail_mult
                if new_trail > trade.sl:
                    trade.sl = new_trail
            else:
                new_trail = close + bar_atr * self.trail_mult
                if new_trail < trade.sl:
                    trade.sl = new_trail

        return newly_closed


    def close_all(self, bar: pd.Series, bar_idx: int) -> list[SimulatedTrade]:
        """Force-close all open positions at the current bar's close (EOD)."""
        closed = []
        for trade in list(self._open_trades):
            self._close_trade(trade, float(bar["close"]), "EOD", bar_idx)
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
        """Fill in all cost fields and mark the trade closed."""
        direction = 1 if trade.direction == "BUY" else -1

        raw_pips   = direction * (exit_price - trade.entry_price) / self.pip_size
        gross_pnl  = raw_pips * self.pip_value * trade.lots
        spread_cost = self.spread_pips * self.pip_value * trade.lots
        commission  = self.commission * trade.lots
        net_pnl = gross_pnl - commission

        trade.exit_bar    = bar_idx
        trade.exit_price  = exit_price
        trade.gross_pnl   = gross_pnl
        trade.net_pnl     = net_pnl
        trade.spread_cost = spread_cost
        trade.commission  = commission
        trade.exit_reason = exit_reason
        trade.is_closed   = True

        self.closed_trades.append(trade)
