"""
services/trade_manager.py — Open position lifecycle management.

Responsibilities:
    - Monitor open positions for volatility spikes
    - Trail stop-loss using ATR-based logic
    - Emergency close of ALL positions (daily drawdown / max DD hard stops)
    - Detect and log newly closed trades

Migrated and extended from the original services/trade_manager.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("TradeManager")


class TradeManager:
    """Manages open position lifecycle for the HMM bot."""

    def __init__(self, magic: int):
        self.magic = magic
        self._last_log_time: Optional[datetime] = None

    # ── Position management ───────────────────────────────────────────────────

    def manage_positions(self, df: pd.DataFrame) -> None:
        """
        Monitor open positions each bar.

        Current logic:
            - Logs a warning if volatility spikes above threshold.
            - Placeholder for trailing-SL based on signal's trail_sl field.

        Args:
            df: Enriched DataFrame (must have 'atr' column populated).
        """
        positions = mt5.positions_get(magic=self.magic)
        if not positions:
            return

        last = df.iloc[-1]
        current_atr = float(last.get("atr", 0.0))

        for pos in positions:
            # Log extreme volatility
            if current_atr > 0:
                # ATR spike: current ATR more than 2× the average (rough heuristic)
                avg_atr = float(df["atr"].rolling(20).mean().iloc[-1]) if "atr" in df.columns else current_atr
                if current_atr > avg_atr * 2:
                    logger.warning(
                        f"[VOLATILITY SPIKE] {pos.symbol} | "
                        f"ATR:{current_atr:.6f} > 2× avg:{avg_atr:.6f} | "
                        f"Ticket:{pos.ticket}"
                    )

    def trail_stop_loss(self, ticket: int, new_sl: float, trade_type: int) -> bool:
        """
        Update the SL of an open position to implement a trailing stop.

        Args:
            ticket:     MT5 position ticket.
            new_sl:     New stop-loss price.
            trade_type: mt5.POSITION_TYPE_BUY or POSITION_TYPE_SELL.

        Returns:
            True if the order was accepted.
        """
        positions = mt5.positions_get()
        if not positions:
            return False

        pos = next((p for p in positions if p.ticket == ticket), None)
        if pos is None:
            return False

        # Only trail in the favourable direction
        if trade_type == mt5.POSITION_TYPE_BUY and new_sl <= pos.sl:
            return False
        if trade_type == mt5.POSITION_TYPE_SELL and new_sl >= pos.sl:
            return False

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       new_sl,
            "tp":       pos.tp,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(
                f"Trailing SL updated | Ticket:{ticket} | "
                f"New SL:{new_sl:.5f}"
            )
            return True

        logger.warning(
            f"Trailing SL failed | Ticket:{ticket} | retcode:{result.retcode}"
        )
        return False

    def monitor_closed_trades(self) -> None:
        """
        Scan last 15 seconds of deal history and log any newly closed trades.
        Called every bar to produce TAKE PROFIT / STOP LOSS log lines.
        """
        since = datetime.now() - timedelta(seconds=15)
        deals = mt5.history_deals_get(since, datetime.now())
        if not deals:
            return

        for deal in deals:
            if deal.magic != self.magic:
                continue
            if deal.entry != mt5.DEAL_ENTRY_OUT:
                continue

            profit    = deal.profit + deal.swap + deal.commission
            deal_time = datetime.fromtimestamp(deal.time)
            result    = "WIN" if profit >= 0 else "LOSS"
            tag       = "TAKE PROFIT HIT" if profit >= 0 else "STOP LOSS HIT"

            logger.info(
                f"[{tag}] {result} | Ticket:{deal.position_id} | "
                f"Price:{deal.price:.5f} | PnL:{profit:+.2f} | "
                f"Time:{deal_time.strftime('%H:%M:%S')}"
            )

    def close_all_positions(self, reason: str = "Risk rule triggered") -> None:
        """
        Emergency close of ALL positions opened by this bot.

        Args:
            reason: Human-readable reason string for the log.
        """
        positions = mt5.positions_get()
        if not positions:
            logger.info(f"[CLOSE ALL] No open positions to close. Reason: {reason}")
            return

        closed = 0
        failed = 0

        for pos in positions:
            if pos.magic != self.magic:
                continue

            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                logger.error(f"No tick for {pos.symbol} — cannot close ticket {pos.ticket}")
                failed += 1
                continue

            if pos.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price      = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price      = tick.ask

            request = {
                "action":   mt5.TRADE_ACTION_DEAL,
                "symbol":   pos.symbol,
                "position": pos.ticket,
                "volume":   pos.volume,
                "type":     order_type,
                "price":    price,
                "deviation": 30,
                "magic":    self.magic,
                "comment":  f"Risk: {reason[:20]}",
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"[TRADING DISABLED] Position closed | "
                    f"Ticket:{pos.ticket} | Symbol:{pos.symbol} | "
                    f"Reason: {reason}"
                )
                closed += 1
            else:
                logger.error(
                    f"Failed to close ticket {pos.ticket} | "
                    f"retcode:{result.retcode} | Reason: {reason}"
                )
                failed += 1

        logger.info(
            f"[CLOSE ALL] Done | Closed:{closed} | Failed:{failed} | "
            f"Reason: {reason}"
        )
