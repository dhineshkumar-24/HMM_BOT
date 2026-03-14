"""
risk_controls/loss_streak_monitor.py — Consecutive stop-loss protection.

Rule:
    If `max_consecutive_losses` (default 2) trades in a row hit stop loss,
    disable trading for the rest of that calendar day.

Implementation:
    - Queries MT5 deal history for the last N minutes to detect closed trades
    - If the most recent trade was a loss, increments the counter
    - Counter resets to 0 on any winning trade
    - Counter resets to 0 at the start of each new trading day

Config key (settings.yaml):
    risk:
        max_consecutive_losses: 2
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import MetaTrader5 as mt5

from utils.logger import setup_logger

logger = setup_logger("LossStreakMonitor")


class LossStreakMonitor:
    """
    Tracks consecutive losing trades and disables further trading when
    the threshold is reached.

    Usage:
        monitor = LossStreakMonitor(config, magic=MAGIC)

        # After each candle (or trade close detection):
        if monitor.check_new_closes():        # returns True if trading should stop
            trading_disabled = True
            logger.warning('CONSECUTIVE LOSS LIMIT reached')

        # At daily reset:
        monitor.reset_daily()
    """

    def __init__(self, config: dict, magic: int):
        """
        Args:
            config: Full settings dict.
            magic:  Magic number of this bot's orders (to filter other bots).
        """
        rc = config.get("risk", {})
        self.max_losses   = rc.get("max_consecutive_losses", 2)
        self.magic        = magic

        self._loss_streak      = 0
        self._last_checked_at  = datetime.min
        self._seen_tickets: set[int] = set()   # avoids double-counting same deal

        logger.info(
            f"LossStreakMonitor ready | Max consecutive losses: {self.max_losses}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def check_new_closes(self, lookback_minutes: int = 10) -> bool:
        """
        Scan recent deal history for newly closed trades by this bot.

        Updates the consecutive-loss counter for each unseen deal.

        Args:
            lookback_minutes: How far back to scan MT5 history (default 10 min).

        Returns:
            True  → trading should be disabled (streak limit reached).
            False → trading can continue.
        """
        since = datetime.now() - timedelta(minutes=lookback_minutes)
        deals = mt5.history_deals_get(since, datetime.now())

        if not deals:
            return self.is_limit_reached()

        for deal in deals:
            # Only consider our bot's closing deals
            if deal.magic != self.magic:
                continue
            if deal.entry != mt5.DEAL_ENTRY_OUT:      # exit deal only
                continue
            if deal.ticket in self._seen_tickets:
                continue

            self._seen_tickets.add(deal.ticket)
            self._process_deal(deal)

        return self.is_limit_reached()

    def is_limit_reached(self) -> bool:
        """Return True if the consecutive-loss limit has been hit."""
        return self._loss_streak >= self.max_losses

    def reset_daily(self) -> None:
        """
        Reset the loss counter at the start of a new trading day.
        Clears the seen-tickets cache too (deals from prior day irrelevant).
        """
        self._loss_streak    = 0
        self._seen_tickets.clear()
        logger.info("LossStreakMonitor — daily reset. Consecutive losses: 0")

    @property
    def current_streak(self) -> int:
        """Number of consecutive losses since last win or daily reset."""
        return self._loss_streak

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_deal(self, deal) -> None:
        """Update the loss streak based on a single closed deal."""
        profit = deal.profit + deal.swap + deal.commission
        deal_time = datetime.fromtimestamp(deal.time)

        if profit < 0:
            self._loss_streak += 1
            logger.warning(
                f"[STOP LOSS HIT] Ticket:{deal.position_id} | "
                f"PnL:{profit:.2f} | "
                f"Consecutive losses: {self._loss_streak}/{self.max_losses} | "
                f"Time:{deal_time.strftime('%H:%M:%S')}"
            )
            if self.is_limit_reached():
                logger.error(
                    f"[CONSECUTIVE LOSS LIMIT] {self._loss_streak} losses in a row. "
                    f"TRADING DISABLED FOR TODAY."
                )
        else:
            if self._loss_streak > 0:
                logger.info(
                    f"[TAKE PROFIT HIT] Ticket:{deal.position_id} | "
                    f"PnL:{profit:.2f} | Loss streak reset to 0 | "
                    f"Time:{deal_time.strftime('%H:%M:%S')}"
                )
            self._loss_streak = 0
