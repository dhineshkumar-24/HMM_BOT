"""
core/execution.py — Order execution via MT5.

Handles MARKET and LIMIT order placement.
Does NOT contain strategy or risk sizing logic.
"""

import MetaTrader5 as mt5

from config import load_config
from utils.logger import setup_logger

logger = setup_logger("Execution")
_CONFIG = load_config()


class Executor:
    """Sends trade requests to the MT5 terminal."""

    def __init__(self, magic: int | None = None):
        """
        Args:
            magic: Magic number identifying this bot's orders.
                   Falls back to config value if not provided.
        """
        self.magic = magic if magic is not None else _CONFIG["project"]["magic_number"]

    def place_trade(
        self,
        symbol: str,
        signal: str,
        volume: float,
        sl: float,
        tp: float,
        price: float | None = None,
        order_type: str = "MARKET",
    ):
        """
        Execute a trade order.

        Args:
            symbol:     Instrument ticker.
            signal:     "BUY" or "SELL".
            volume:     Lot size.
            sl:         Stop-loss price.
            tp:         Take-profit price.
            price:      Limit price (LIMIT orders only; auto-fetched for MARKET).
            order_type: "MARKET" or "LIMIT".

        Returns:
            MT5 order result on success, None on failure.
        """
        action  = mt5.TRADE_ACTION_DEAL
        type_op = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL

        if order_type == "LIMIT":
            action  = mt5.TRADE_ACTION_PENDING
            type_op = mt5.ORDER_TYPE_BUY_LIMIT if signal == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT

        # ── Spread protection ─────────────────────────────────
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None

        spread = abs(tick.ask - tick.bid)
        max_spread = _CONFIG["execution"].get("max_spread", 0.00020)

        if spread > max_spread:
            logger.warning(
                f"Spread too high ({spread:.5f}) — skipping trade"
            )
            return None
        # ── Prevent duplicate trades (safety guard) ───────────
        positions = mt5.positions_get(symbol=symbol, magic=self.magic)

        if positions:
            logger.info(f"[EXECUTION BLOCKED] Position already open for {symbol}")
            return None
        # ──────────────────────────────────────────────────────
        if not price:
            price = tick.ask if signal == "BUY" else tick.bid

        request = {
            "action":       action,
            "symbol":       symbol,
            "volume":       float(volume),
            "type":         type_op,
            "price":        float(price),
            "sl":           float(sl),
            "tp":           float(tp),
            "magic":        self.magic,
            "comment":      _CONFIG["execution"]["order_comment"],
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "deviation":    _CONFIG["execution"]["deviation"],
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order FAILED — retcode {result.retcode}: {result.comment}")
            return None

        # ── Slippage logging ─────────────────────────────
        fill_price = result.price

        logger.info(
            f"Order placed | {signal} {volume} lots | "
            f"Requested:{price:.5f} | Filled:{fill_price:.5f} | "
            f"SL {sl:.5f} | TP {tp:.5f}"
        )

        return result

    def close_partial(self, ticket: int, volume_to_close: float):
        """
        Partial close of an existing position.
        Placeholder — logic depends on broker hedging mode.
        """
        raise NotImplementedError("Partial close not yet implemented.")
