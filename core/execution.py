import MetaTrader5 as mt5
import yaml
from utils.logger import setup_logger

logger = setup_logger("Execution")

# Load config
with open("config/settings.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

class Executor:
    def __init__(self, magic=None):
        self.magic = magic if magic is not None else CONFIG['project']['magic_number']

    def place_trade(self, symbol, signal, volume, sl, tp, price=None, order_type="MARKET"):
        """
        Executes a trade order.
        signal: BUY or SELL
        order_type: MARKET or LIMIT
        """
        action = mt5.TRADE_ACTION_DEAL
        type_op = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
        
        # Determine actual MT5 Order Type
        if order_type == "LIMIT":
            action = mt5.TRADE_ACTION_PENDING
            if signal == "BUY":
                type_op = mt5.ORDER_TYPE_BUY_LIMIT
            else:
                type_op = mt5.ORDER_TYPE_SELL_LIMIT

        # Current Price if Market
        if not price:
            tick = mt5.symbol_info_tick(symbol)
            price = tick.ask if signal == "BUY" else tick.bid
            
        request = {
            "action": action,
            "symbol": symbol,
            "volume": float(volume),
            "type": type_op,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "magic": self.magic,
            "comment": CONFIG['execution']['order_comment'],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "deviation": CONFIG['execution']['deviation'],
        }

        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order Failed: {result.retcode} - {result.comment}")
            return None
        
        logger.info(f"Order Placed: {signal} {volume} @ {price} | SL: {sl} TP: {tp}")
        return result

    def close_partial(self, ticket, volume_to_close):
        """
        Partially closes an existing position.
        """
        # Valid only for Hedging accounts mostly, depends on broker
        # Logic: Place an opposite deal for specific volume
        pass # Placeholder for advanced partial close logic
