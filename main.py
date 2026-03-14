import time
import yaml
import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd

from core.mt5_connector import MT5Connector
from core.execution import Executor
from core.risk import calculate_position_size, check_daily_drawdown
from core.data_feed import DataFeed
from strategy.mean_reversion import MeanReversionStrategy
from services.trade_manager import TradeManager
from services.order_validator import OrderValidator
from utils.logger import setup_logger
from utils.helpers import check_trading_session

# =========================
# Load Config
# =========================
with open("config/settings.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

SYMBOL = CONFIG['trading']['symbol']
TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1
}
TIMEFRAME = TF_MAP.get(CONFIG['trading']['timeframe'], mt5.TIMEFRAME_M15)
MAGIC = CONFIG['project']['magic_number']

logger = setup_logger("Main")


def main():
    logger.info("Bot Starting (Backtest-Aligned Mode)...")

    # =========================
    # 1. Connect
    # =========================
    connector = MT5Connector()
    if not connector.connect():
        logger.error("MT5 connection failed")
        return

    # =========================
    # 2. Initialize Modules
    # =========================
    strategy = MeanReversionStrategy(CONFIG)
    executor = Executor(magic=MAGIC)
    trade_manager = TradeManager(magic=MAGIC)
    validator = OrderValidator(CONFIG)
    data_feed = DataFeed(SYMBOL, timeframe=TIMEFRAME)

    # =========================
    # Startup Position Sync
    # =========================
    open_positions = mt5.positions_get(symbol=SYMBOL)
    has_open_trade = bool(open_positions)

    if has_open_trade:
        logger.warning("Existing position detected on startup. No new trades allowed until it closes.")

    # =========================
    # Drawdown Tracking
    # =========================
    account = mt5.account_info()
    daily_start_balance = account.balance
    current_day = datetime.now().day
    trading_disabled_today = False

    # =========================
    # Strategy State (Backtest Match)
    # =========================
    pending_signal = None
    wait_counter = 0

    Z_TRIGGER = CONFIG['strategy']['z_score_trigger']
    ENTRY_TARGET = CONFIG['strategy']['entry_zone_target']
    HURST_LIMIT = CONFIG['strategy']['hurst_threshold']
    RSI_OB = CONFIG['strategy']['rsi_overbought']
    RSI_OS = CONFIG['strategy']['rsi_oversold']
    RR = CONFIG['trading']['risk_reward_ratio']

    last_candle_time = None
    last_heartbeat_minute = None


    logger.info(f"Connected | Balance: {account.balance} {account.currency}")

    # =========================
    # Main Loop (Candle-Based)
    # =========================
    while True:
        try:
            now = datetime.now()

            # ----- HEARTBEAT (once per minute) -----
            if last_heartbeat_minute != now.minute:
                logger.info("Waiting for next candle...")
                last_heartbeat_minute = now.minute

            # -------- Daily Reset --------
            if now.day != current_day:
                daily_start_balance = mt5.account_info().balance
                current_day = now.day
                trading_disabled_today = False

                # 🔴 Reset strategy state for new day
                pending_signal = None
                wait_counter = 0

                logger.info("New trading day started — trading re-enabled")


            # -------- Fetch Data --------
            df = data_feed.get_candles(n=500)
            if df is None or len(df) < 120:
                time.sleep(5)
                continue

            df = strategy.calculate_indicators(df)

            current_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]

            # -------- Manage Open Trades --------
            trade_manager.manage_positions(df)
            trade_manager.monitor_closed_trades()

            # -------- New Candle Detection --------
            if last_candle_time == current_candle['time']:
                time.sleep(1)
                continue
            last_candle_time = current_candle['time']
            
            z = prev_candle['z_score']
            rsi = prev_candle['rsi']
            hurst = prev_candle['hurst']
            
            # ===== STRATEGY SNAPSHOT (WHY NO TRADE) =====
            session_ok = check_trading_session(CONFIG, candle_time=current_candle['time'])

            logger.info(
                f"Candle {current_candle['time']} | "
                f"Z: {z:.2f} | "
                f"Hurst: {hurst:.2f} | "
                f"RSI: {rsi:.1f} | "
                f"Session: {'OPEN' if session_ok else 'CLOSED'} | "
                f"Pending: {pending_signal}"
            )
            # ==========================================

            # Sync open position state
            open_positions = mt5.positions_get(symbol=SYMBOL)
            has_open_trade = bool(open_positions)

            # -------- Drawdown Check (HARD STOP) --------
            account = mt5.account_info()
            if not trading_disabled_today:
                if check_daily_drawdown(
                    daily_start_balance,
                    account.equity,
                    CONFIG['trading']['max_daily_drawdown_pct']
                ):
                    logger.error("Max daily drawdown reached. Trading DISABLED for today.")
                    trade_manager.close_all_positions()

                    trading_disabled_today = True

                    # 🔴 CRITICAL: reset strategy state
                    pending_signal = None
                    wait_counter = 0

                    continue

            # 🧪 OPTIONAL (RECOMMENDED): visibility log
            if trading_disabled_today:
                logger.info("Trading disabled for today — monitoring only")


            if trading_disabled_today or has_open_trade:
                continue
            
            # -------- Session Filter --------
            if not check_trading_session(CONFIG, candle_time=current_candle['time']):
                continue


            # -------- Volatility Regime Filter --------
            if prev_candle['vol_slope'] > 0.0001:
                pending_signal = None
                wait_counter = 0
                continue

            
            # -------- Regime Filter --------
            if hurst > HURST_LIMIT and pending_signal is None:
                continue

            # -------- Trigger --------
            if abs(z) >= Z_TRIGGER:
                if z > 0 and rsi > RSI_OB:
                    pending_signal = "SELL_WAIT"
                elif z < 0 and rsi < RSI_OS:
                    pending_signal = "BUY_WAIT"
                wait_counter = 0

            # -------- Pending Signal Logic --------
            if pending_signal:
                wait_counter += 1
                if wait_counter > 20:
                    pending_signal = None
                    wait_counter = 0

            # -------- Entry Zone --------
            signal = None
            if pending_signal == "SELL_WAIT" and z <= ENTRY_TARGET:
                signal = "SELL"
                pending_signal = None
            elif pending_signal == "BUY_WAIT" and z >= -ENTRY_TARGET:
                signal = "BUY"
                pending_signal = None

            # -------- Execute Trade --------
            if signal and not trading_disabled_today and validator.validate(account, 0):
                price = current_candle['close']

                # EXACT backtest SL logic
                vol_price = prev_candle['volatility'] * price
                sl_dist = vol_price * 2

                if signal == "BUY":
                    sl = price - sl_dist
                    tp = price + (sl_dist * RR)
                else:
                    sl = price + sl_dist
                    tp = price - (sl_dist * RR)

                lots = calculate_position_size(
                    account.balance,
                    CONFIG['trading']['risk_per_trade'],
                    sl_dist,
                    SYMBOL
                )

                if lots > 0:
                    logger.info(
                        f"{signal} | Price: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}"
                    )
                    executor.place_trade(SYMBOL, signal, lots, sl, tp)

            time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Bot stopped manually")
            break
        except Exception as e:
            tra.exception(f"Unexpected error: {e}")
            time.sleep(5)

    connector.disconnect()


if __name__ == "__main__":
    main()
