"""
risk_controls/drawdown_monitor.py — Two-level drawdown protection.

Level 1 — Daily Drawdown (3%)
    If equity drops 3% below today's opening balance:
    → Close all positions
    → Disable trading for the rest of today
    → Log DAILY DRAWDOWN LIMIT event

Level 2 — Absolute Max Drawdown (15%)
    If equity drops 15% below the initial capital at bot startup:
    → Close all positions
    → Permanently disable trading (until manual restart)
    → Log MAX ACCOUNT DRAWDOWN — CRITICAL SHUTDOWN

Both thresholds are configurable in settings.yaml:
    risk:
        max_daily_drawdown_pct:   0.03
        max_account_drawdown_pct: 0.15
"""

from __future__ import annotations

from utils.logger import setup_logger

logger = setup_logger("DrawdownMonitor")


class DrawdownMonitor:
    """
    Monitors two drawdown levels and raises flags when limits are breached.

    Usage:
        monitor = DrawdownMonitor(config, initial_capital=account.balance)

        # each bar:
        account = mt5.account_info()
        status  = monitor.check(account.equity, daily_start_balance)
        if status == DrawdownMonitor.DAILY_LIMIT:
            trade_manager.close_all_positions()
            trading_disabled = True
        elif status == DrawdownMonitor.ABSOLUTE_LIMIT:
            trade_manager.close_all_positions()
            permanent_shutdown = True
    """

    # Return codes
    OK             = "ok"
    DAILY_LIMIT    = "daily_limit"
    ABSOLUTE_LIMIT = "absolute_limit"

    def __init__(self, config: dict, initial_capital: float):
        """
        Args:
            config:          Full settings dict (reads risk sub-section).
            initial_capital: Account balance at bot startup (not daily reset).
        """
        rc = config.get("risk", {})
        self.daily_dd_pct    = rc.get("max_daily_drawdown_pct",   0.03)
        self.absolute_dd_pct = rc.get("max_account_drawdown_pct", 0.15)
        self.initial_capital = initial_capital

        self._daily_breach_logged    = False
        self._absolute_breach_logged = False

        logger.info(
            f"DrawdownMonitor ready | "
            f"Daily limit: {self.daily_dd_pct:.1%} | "
            f"Absolute limit: {self.absolute_dd_pct:.1%} | "
            f"Initial capital: {initial_capital:.2f}"
        )

    def check(self, current_equity: float, daily_start_balance: float) -> str:
        """
        Evaluate current equity against both drawdown thresholds.

        Args:
            current_equity:      Account equity right now (unrealised PnL included).
            daily_start_balance: Balance recorded at start of today's session.

        Returns:
            DrawdownMonitor.OK             — no breach
            DrawdownMonitor.DAILY_LIMIT    — daily limit breached (disable today)
            DrawdownMonitor.ABSOLUTE_LIMIT — absolute limit breached (hard shutdown)
        """
        # ── Level 2 check first (more severe) ─────────────────────────────────
        if self.initial_capital > 0:
            abs_dd = (self.initial_capital - current_equity) / self.initial_capital
            if abs_dd >= self.absolute_dd_pct:
                if not self._absolute_breach_logged:
                    logger.critical(
                        f"[MAX ACCOUNT DRAWDOWN] Equity: {current_equity:.2f} | "
                        f"Drawdown: {abs_dd:.2%} >= limit {self.absolute_dd_pct:.2%} | "
                        f"PERMANENT TRADING SHUTDOWN"
                    )
                    self._absolute_breach_logged = True
                return self.ABSOLUTE_LIMIT

        # ── Level 1 check ──────────────────────────────────────────────────────
        if daily_start_balance > 0:
            daily_dd = (daily_start_balance - current_equity) / daily_start_balance
            if daily_dd >= self.daily_dd_pct:
                if not self._daily_breach_logged:
                    logger.error(
                        f"[DAILY DRAWDOWN LIMIT] Equity: {current_equity:.2f} | "
                        f"Daily DD: {daily_dd:.2%} >= limit {self.daily_dd_pct:.2%} | "
                        f"TRADING DISABLED FOR TODAY"
                    )
                    self._daily_breach_logged = True
                return self.DAILY_LIMIT

        return self.OK

    def reset_daily(self) -> None:
        """
        Reset the daily breach flag at the start of a new trading day.
        Call this whenever daily_start_balance is refreshed.
        """
        self._daily_breach_logged = False
        logger.info("DrawdownMonitor — daily limit flag reset for new day.")

    def get_daily_drawdown(self, equity: float, daily_start: float) -> float:
        """Return the current daily drawdown as a fraction (0.0 – 1.0)."""
        if daily_start <= 0:
            return 0.0
        return max(0.0, (daily_start - equity) / daily_start)

    def get_absolute_drawdown(self, equity: float) -> float:
        """Return the drawdown vs initial capital as a fraction (0.0 – 1.0)."""
        if self.initial_capital <= 0:
            return 0.0
        return max(0.0, (self.initial_capital - equity) / self.initial_capital)
