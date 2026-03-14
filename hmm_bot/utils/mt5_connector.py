"""
utils/mt5_connector.py — MetaTrader5 connection management.

Migrated from core/mt5_connector.py. Handles init/shutdown only.
No strategy or execution logic lives here.
"""

import sys
import MetaTrader5 as mt5


class MT5Connector:
    """Manages the lifecycle of the MetaTrader5 terminal connection."""

    def __init__(self):
        self.connected = False

    def connect(self) -> bool:
        """
        Initialise the MT5 connection and verify a trading account exists.

        Returns:
            True on success. Raises SystemExit on hard failure.
        """
        if not mt5.initialize():
            print(f"[MT5Connector] Initialization failed — error: {mt5.last_error()}")
            return False

        account_info = mt5.account_info()
        if account_info is None:
            print("[MT5Connector] No trading account found. Check terminal login.")
            mt5.shutdown()
            return False

        self.connected = True
        print("=" * 50)
        print("  MT5 Connected")
        print(f"  Broker   : {account_info.company}")
        print(f"  Account  : {account_info.login}")
        print(f"  Balance  : {account_info.balance} {account_info.currency}")
        print("=" * 50)
        return True
    # ── ADD THIS FUNCTION ─────────────────────────────
    def enable_symbol(self, symbol: str) -> bool:
        """
        Ensure the symbol is enabled in MT5 Market Watch.
        """
        if not mt5.symbol_select(symbol, True):
            print(f"[MT5Connector] Failed to enable symbol: {symbol}")
            return False

        print(f"[MT5Connector] Symbol enabled: {symbol}")
        return True
    # ──────────────────────────────────────────────────

    def disconnect(self) -> None:
        """Gracefully shut down the MT5 connection."""
        mt5.shutdown()
        self.connected = False
        print("[MT5Connector] Connection closed.")
