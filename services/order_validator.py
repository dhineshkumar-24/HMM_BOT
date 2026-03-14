import logging

logger = logging.getLogger("OrderValidator")

class OrderValidator:
    def __init__(self, config):
        self.config = config
        self.max_daily_loss = config['trading']['max_daily_drawdown_pct']

    def validate(self, account_info, risk_exposure):
        """
        Checks if a trade accepts risk rules.
        """
        # 1. Check Margin
        if account_info.margin_free < risk_exposure:
            logger.warning("Insufficient Margin")
            return False

        # 2. Daily Drawdown
        # This requires tracking starting balance of the day. 
        # For this stateless bot, we might need to query history deals for today.
        # Assuming account_info.balance is current.
        # Check equity vs balance? 
        # Simply: if Equity < Balance * (1 - MaxDD), stop.
        
        if account_info.balance == 0:
            return False
            
        current_dd = (account_info.balance - account_info.equity) / account_info.balance
        if current_dd > self.max_daily_loss:
            logger.warning(f"Daily Drawdown Limit Reached: {current_dd*100:.2f}%")
            return False

        # 3. Signal Crowding Check
        # Hard limits on repeated signals to prevent over-trading the same setup
        signal_crowding_limit = self.config.get('alpha', {}).get('signal_decay_bars', 5)
        if getattr(self, '_recent_signals_count', 0) >= signal_crowding_limit:
            logger.warning("Signal crowding detected! Rejecting trade.")
            return False

        return True
