import MetaTrader5 as mt5
from utils.logger import setup_logger

logger = setup_logger("TradeManager")

class TradeManager:
    def __init__(self, magic):
        self.magic = magic

    def manage_positions(self, df):
        """
        Monitors open positions for exit conditions.
        """
        positions = mt5.positions_get(magic=self.magic)
        if positions is None:
            return

        # Get latest market data
        last_candle = df.iloc[-1]
        current_vol = last_candle['volatility'] # Rolling val
        
        for pos in positions:
            symbol = pos.symbol
            # STEP 6: Partial Exit Logic
            # "Volatility Expansion: Current vol >= X * entry vol"
            # We don't store entry vol in MT5 comment easily, but we can check if current vol is extreme.
            
            # Simple Logic: If Volatility Spike, Reduce Risk.
            if current_vol > 0.0005: # Threshold example
                logger.info(f"Volatility Spike on {symbol}. Managing position...")
                # Implement partial close or tight SL here
                pass

            # Trailing SL or Breakeven logic could go here
            pass

    def monitor_closed_trades(self):
        """
        Checks for recently closed trades and logs them.
        """
        from datetime import datetime, timedelta
        
        # Check last 5 minutes history
        from_date = datetime.now() - timedelta(minutes=5)
        deals = mt5.history_deals_get(from_date, datetime.now(), group="*")
        
        if deals:
            for deal in deals:
                # Filter for Exit Deals (Entry=0, Exit=1)
                # Deal Entry In=0, Out=1, In/Out=2
                if deal.entry == mt5.DEAL_ENTRY_OUT:
                    # Avoid duplicated logging (naive check: timestamp very recent)
                    # Ideally track by ticket, but simply logging found deals is OK for low freq.
                    
                    # Only log if it happened in last 15 seconds to avoid spam on every loop
                    deal_time = datetime.fromtimestamp(deal.time)
                    if (datetime.now() - deal_time).total_seconds() < 15:
                        res = "WIN" if deal.profit > 0 else "LOSS"
                        logger.info(
                            f"{res} | Ticket: {deal.position_id} | "
                            f"Price: {deal.price:.5f} | PnL: ${deal.profit:.2f} | "
                            f"Comment: {deal.comment}"
                        )
    def close_all_positions(self):
        """
        HARD STOP: Close all open positions created by this bot
        (used for daily drawdown protection).
        """
        positions = mt5.positions_get()
        if positions is None:
            logger.warning("No open positions to close.")
            return

        for pos in positions:
            # Only close positions opened by THIS bot
            if pos.magic != self.magic:
                continue

            symbol = pos.symbol

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"No tick data for {symbol}, cannot close position.")
                continue

            # Determine close order type & price
            if pos.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "position": pos.ticket,
                "volume": pos.volume,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": self.magic,
                "comment": "Daily drawdown hard stop",
            }

            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(
                    f"Failed to close position {pos.ticket} | retcode={result.retcode}"
                )
            else:
                logger.info(
                    f"Closed position {pos.ticket} due to daily drawdown"
                )

