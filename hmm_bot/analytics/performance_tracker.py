"""
analytics/performance_tracker.py — Runtime trade performance analytics.

Tracks every executed trade and computes key performance metrics:

    - Total trades, wins, losses
    - Win rate
    - Average return per trade
    - Return standard deviation
    - Sharpe ratio (annualised approximation using per-trade returns)

Daily summary report emitted at end of each trading day (or on demand).

Design:
    - All data stored in-memory during runtime (no DB dependency)
    - add_trade() called whenever a position closes
    - daily_summary() called at the end of each trading day
    - reset_daily() clears daily counters only (lifetime counters persist)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import numpy as np

from utils.logger import setup_logger

logger = setup_logger("PerfTracker")


# ─────────────────────────────────────────────────────────────────────────────
# Trade record dataclass (plain dict — no external deps)
# ─────────────────────────────────────────────────────────────────────────────

def make_trade_record(
    direction:  str,
    entry:      float,
    close:      float,
    lots:       float,
    profit:     float,
    strategy:   str = "",
    regime:     Optional[int] = None,
    session:    str = "",
) -> dict:
    """Create a standardised trade record dict."""
    pct_return = (profit / (entry * lots)) if (entry > 0 and lots > 0) else 0.0
    return {
        "ts":        datetime.now().isoformat(),
        "direction": direction,
        "entry":     entry,
        "close":     close,
        "lots":      lots,
        "profit":    profit,
        "pct_ret":   pct_return,
        "win":       profit > 0,
        "strategy":  strategy,
        "regime":    regime,
        "session":   session,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PerformanceTracker
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceTracker:
    """
    Runtime trade analytics engine.

    Usage:
        tracker = PerformanceTracker(N=252)

        # after every trade close:
        tracker.add_trade(direction, entry, close, lots, profit, strategy, regime)

        # end of day:
        tracker.daily_summary()
        tracker.reset_daily()
    """

    def __init__(self, N: int = 252):
        """
        Args:
            N: Number of periods per year used in Sharpe annualisation.
               Default 252 (trading days). Use the number of trades per
               year if computing per-trade Sharpe.
        """
        self.N = N

        # Lifetime records
        self._all_trades:   list[dict] = []

        # Daily records (reset each day)
        self._daily_trades: list[dict] = []

        # Max drawdown tracking (intraday, based on running equity)
        self._equity_peak   = 0.0
        self._daily_max_dd  = 0.0

        logger.info(f"PerformanceTracker ready | Sharpe N={self.N}")

    # ── Record a trade ────────────────────────────────────────────────────────

    def add_trade(
        self,
        direction: str,
        entry:     float,
        close:     float,
        lots:      float,
        profit:    float,
        strategy:  str = "",
        regime:    Optional[int] = None,
        session:   str = "",
    ) -> None:
        """
        Record a completed trade.

        Args:
            direction: "BUY" or "SELL".
            entry:     Entry price.
            close:     Close price.
            lots:      Trade lot size.
            profit:    Realised profit in account currency (after swap & commission).
            strategy:  Strategy name ("MeanReversionStrategy", etc.).
            regime:    HMM regime at time of entry (0/1/2/None).
            session:   Session label ("asian", "london", "newyork").
        """
        record = make_trade_record(direction, entry, close, lots, profit, strategy, regime, session)
        self._all_trades.append(record)
        self._daily_trades.append(record)

        tag = "WIN" if profit > 0 else "LOSS"
        logger.info(
            f"[TRADE EXECUTED] {tag} | {direction} | "
            f"Entry:{entry:.5f} Close:{close:.5f} | "
            f"Lots:{lots:.2f} | PnL:{profit:+.2f} | "
            f"Strategy:{strategy} | Regime:{regime} | Session:{session}"
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self, trades: list[dict]) -> dict:
        """Compute all metrics for a list of trade records."""
        n = len(trades)
        if n == 0:
            return {
                "total": 0, "wins": 0, "losses": 0,
                "win_rate": 0.0, "total_profit": 0.0,
                "avg_return": 0.0, "ret_std": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0,
            }

        wins   = sum(1 for t in trades if t["win"])
        losses = n - wins
        returns = [t["pct_ret"] for t in trades]
        profits = [t["profit"]  for t in trades]

        avg_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))  if n > 1 else 0.0
        sharpe  = (avg_ret / std_ret * math.sqrt(self.N)) if std_ret > 0 else 0.0

        # Running equity drawdown on profit stream
        peak     = 0.0
        trough   = 0.0
        equity   = 0.0
        max_dd   = 0.0
        for p in profits:
            equity += p
            if equity > peak:
                peak   = equity
                trough = equity
            else:
                trough = min(trough, equity)
                dd     = (peak - trough) / abs(peak) if peak != 0 else 0.0
                max_dd = max(max_dd, dd)

        return {
            "total":        n,
            "wins":         wins,
            "losses":       losses,
            "win_rate":     wins / n,
            "total_profit": sum(profits),
            "avg_return":   avg_ret,
            "ret_std":      std_ret,
            "sharpe":       sharpe,
            "max_drawdown": max_dd,
        }

    # ── Summary output ────────────────────────────────────────────────────────

    def daily_summary(self) -> dict:
        """
        Compute and log the end-of-day performance summary.

        Returns the metrics dict for programmatic use.
        """
        m = self._compute_metrics(self._daily_trades)

        logger.info("=" * 50)
        logger.info("  DAILY SUMMARY")
        logger.info("=" * 50)
        logger.info(f"  Trades    : {m['total']}")
        logger.info(f"  Wins      : {m['wins']}")
        logger.info(f"  Losses    : {m['losses']}")
        logger.info(f"  Win Rate  : {m['win_rate']:.1%}")
        logger.info(f"  PnL       : {m['total_profit']:+.2f}")
        logger.info(f"  Avg Return: {m['avg_return']:.4%}")
        logger.info(f"  Std Dev   : {m['ret_std']:.4%}")
        logger.info(f"  Sharpe    : {m['sharpe']:.2f}")
        logger.info(f"  Max DD    : {m['max_drawdown']:.2%}")
        logger.info("=" * 50)

        return m

    def lifetime_summary(self) -> dict:
        """Compute and log the all-time performance summary."""
        m = self._compute_metrics(self._all_trades)

        logger.info("=" * 50)
        logger.info("  LIFETIME SUMMARY")
        logger.info("=" * 50)
        logger.info(f"  Trades    : {m['total']}")
        logger.info(f"  Win Rate  : {m['win_rate']:.1%}")
        logger.info(f"  Total PnL : {m['total_profit']:+.2f}")
        logger.info(f"  Sharpe    : {m['sharpe']:.2f}")
        logger.info(f"  Max DD    : {m['max_drawdown']:.2%}")
        logger.info("=" * 50)

        return m

    # ── Daily reset ───────────────────────────────────────────────────────────

    def reset_daily(self) -> None:
        """Clear daily trade list. Lifetime records are preserved."""
        self._daily_trades.clear()
        self._daily_max_dd = 0.0
        self._equity_peak  = 0.0
        logger.info("PerformanceTracker — daily counters reset.")

    # ── Quick accessors ───────────────────────────────────────────────────────

    @property
    def today_trade_count(self) -> int:
        return len(self._daily_trades)

    @property
    def lifetime_trade_count(self) -> int:
        return len(self._all_trades)

    @property
    def today_profit(self) -> float:
        return sum(t["profit"] for t in self._daily_trades)
