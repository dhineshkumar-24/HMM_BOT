import MetaTrader5 as mt5
import sys


class MT5Connector:
    def __init__(self):
        self.connected = False

    def connect(self):
        """
        Initialize MT5 connection
        """
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            print("Error code:", mt5.last_error())
            sys.exit(1)

        account_info = mt5.account_info()
        if account_info is None:
            print("❌ No trading account found")
            mt5.shutdown()
            sys.exit(1)

        self.connected = True
        print("✅ MT5 connected successfully")
        print(f"Broker      : {account_info.company}")
        print(f"Account ID  : {account_info.login}")
        print(f"Balance     : {account_info.balance}")
        print(f"Currency    : {account_info.currency}")
        return True


    def enable_symbol(self, symbol: str):
        """
        Ensure symbol is enabled in Market Watch
        """
        if not mt5.symbol_select(symbol, True):
            print(f"❌ Failed to enable symbol: {symbol}")
            return False

        print(f"✅ Symbol enabled: {symbol}")
        return True


    def disconnect(self):
        """
        Shutdown MT5 connection
        """
        mt5.shutdown()
        self.connected = False
        print("🔌 MT5 connection closed")