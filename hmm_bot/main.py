"""
main.py — HMM Adaptive Trading Bot — Production Entry Point.

Run from inside hmm_bot/:
    python main.py

Architecture:
    Config → MT5 → HMM (warm-up or trained) → StrategyRouter
    → DrawdownMonitor → LossStreakMonitor → Executor
    → PerformanceTracker → daily summary

Risk controls (applied every bar, in order):
    1. Max account drawdown (15%) — permanent shutdown
    2. Daily drawdown (3%)        — disable for rest of day
    3. Consecutive loss limit (2) — disable for rest of day
    4. Open position guard        — no stacking
    5. Session filter             — session + regime routing
    6. HMM confidence gate        — min 0.60 posterior probability
    7. Regime stability gate      — 5-bar stable window
"""

import sys
import time
from datetime import datetime

import MetaTrader5 as mt5

# ── Core imports ──────────────────────────────────────────────────────────────
from config                import load_config
from utils.logger          import setup_logger
from utils.mt5_connector   import MT5Connector
from utils.helpers         import detect_session, check_trading_session
from utils.features        import build_feature_matrix
from core.data_feed        import DataFeed
from core.execution        import Executor
from core.risk             import calculate_position_size
from core.regime_filters   import apply_regime_risk_scaling
from core.hmm_model        import HMMRegimeDetector, REGIME_MEAN_REVERT

# ── Strategy / Routing ────────────────────────────────────────────────────────
from strategy.strategy_router import StrategyRouter

# ── Risk controls (NEW) ───────────────────────────────────────────────────────
from risk_controls.drawdown_monitor  import DrawdownMonitor
from risk_controls.loss_streak_monitor import LossStreakMonitor

# ── Services (NEW) ────────────────────────────────────────────────────────────
from services.trade_manager import TradeManager

# ── Analytics (NEW) ──────────────────────────────────────────────────────────
from analytics.performance_tracker import PerformanceTracker

# ─────────────────────────────────────────────────────────────────────────────
# Globals loaded once at startup
# ─────────────────────────────────────────────────────────────────────────────

CONFIG    = load_config()
logger    = setup_logger("Main")

SYMBOL    = CONFIG["trading"]["symbol"]
MAGIC     = CONFIG["project"]["magic_number"]
BASE_RISK = CONFIG["trading"]["risk_per_trade"]
RR        = CONFIG["trading"]["risk_reward_ratio"]

_TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
}
TIMEFRAME = _TF_MAP.get(CONFIG["trading"]["timeframe"], mt5.TIMEFRAME_M1)

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("  HMM Adaptive Trading Bot — Starting")
    logger.info("=" * 60)

    # ── 1. MT5 Connection ──────────────────────────────────────────────────────
    connector = MT5Connector()
    if not connector.connect():
        logger.error("MT5 connection failed. Exiting.")
        sys.exit(1)
    # Ensure symbol is enabled in Market Watch
    if not connector.enable_symbol(SYMBOL):
        logger.error(f"Failed to enable trading symbol: {SYMBOL}")
        sys.exit(1)

    account          = mt5.account_info()
    initial_capital  = account.balance
    logger.info(
        f"Connected | Balance: {initial_capital:.2f} {account.currency} | "
        f"Symbol: {SYMBOL} | TF: {CONFIG['trading']['timeframe']}"
    )

    # ── 2. Module Initialisation ───────────────────────────────────────────────
    data_feed     = DataFeed(SYMBOL, timeframe=TIMEFRAME, bars=600)
    executor      = Executor(magic=MAGIC)
    router        = StrategyRouter(CONFIG)
    trade_mgr     = TradeManager(magic=MAGIC)
    performance   = PerformanceTracker(N=252)

    hmm = HMMRegimeDetector(
        n_states      = CONFIG.get("hmm", {}).get("n_components",         3),
        n_iter        = CONFIG.get("hmm", {}).get("n_iter",             300),
        n_seeds       = 3,
        model_path    = CONFIG.get("hmm", {}).get("model_path",  "models/hmm.pkl"),
        confidence_thr= CONFIG.get("hmm", {}).get("confidence_threshold", 0.60),
    )

    dd_monitor    = DrawdownMonitor(CONFIG, initial_capital=initial_capital)
    loss_monitor  = LossStreakMonitor(CONFIG, magic=MAGIC)

    # Try loading a pre-trained HMM model
    if not hmm.load():
        logger.info("No saved HMM model found — warm-up mode (MeanReversion only).")

    # ── 3. Daily state ─────────────────────────────────────────────────────────
    account              = mt5.account_info()
    daily_start_balance  = account.balance
    current_day          = datetime.now().day

    trading_disabled     = False   # daily reset each dawn
    permanent_shutdown   = False   # persists until manual restart

    last_candle_time     = None
    last_heartbeat_min   = None
    candle_counter       = 0
    retrain_interval     = CONFIG.get("hmm", {}).get("retrain_interval", 10000)

    # ── 4. Main trading loop ───────────────────────────────────────────────────
    while True:
        try:
            now = datetime.now()

            # ── Heartbeat (once per minute) ────────────────────────────────────
            if last_heartbeat_min != now.minute:
                account = mt5.account_info()
                dd_pct  = dd_monitor.get_daily_drawdown(account.equity, daily_start_balance)
                logger.info(
                    f"Heartbeat | {now.strftime('%Y-%m-%d %H:%M')} | "
                    f"Equity:{account.equity:.2f} | DailyDD:{dd_pct:.2%} | "
                    f"Losses:{loss_monitor.current_streak} | "
                    f"Trading:{'DISABLED' if (trading_disabled or permanent_shutdown) else 'ENABLED'}"
                )
                last_heartbeat_min = now.minute

            # ── Hard stop check ────────────────────────────────────────────────
            if permanent_shutdown:
                time.sleep(30)
                continue

            # ── Daily reset at new calendar day ───────────────────────────────
            if now.day != current_day:
                # End-of-day summary before resetting
                if performance.today_trade_count > 0:
                    performance.daily_summary()

                account              = mt5.account_info()
                daily_start_balance  = account.balance
                current_day          = now.day
                trading_disabled     = False

                dd_monitor.reset_daily()
                loss_monitor.reset_daily()
                performance.reset_daily()
                router.reset_state()
                hmm.clear_history()

                logger.info(
                    f"[NEW DAY] {now.date()} | "
                    f"Start balance: {daily_start_balance:.2f} — trading re-enabled"
                )

            # ── Fetch OHLCV data ───────────────────────────────────────────────
            df = data_feed.get_candles(n=600)
            if df is None or len(df) < 150:
                logger.warning("Insufficient data. Waiting...")
                time.sleep(5)
                continue

            # ── New candle guard (avoid acting on same bar twice) ──────────────
            current_candle = df.iloc[-1]
            if last_candle_time == current_candle["time"]:
                time.sleep(1)
                continue
            last_candle_time  = current_candle["time"]
            candle_counter   += 1
            candle_time       = current_candle["time"]   # broker datetime

            # ── Indicator enrichment ───────────────────────────────────────────
            df = router.calculate_indicators(df)

            # ── Monitor open positions ─────────────────────────────────────────
            trade_mgr.manage_positions(df)
            trade_mgr.monitor_closed_trades()

            # ── HMM Regime Detection ───────────────────────────────────────────
            regime       = None
            confidence   = 0.0
            regime_probs = None

            if hmm.is_trained:
                try:
                    regime, confidence, regime_probs = hmm.predict(df)
                    hmm.update_history(regime)
                    logger.info(
                        f"HMM Regime: {regime} ({hmm.regime_name(regime)}) | "
                        f"Confidence: {confidence:.2%} | "
                        f"Stable: {hmm.is_regime_stable()}"
                    )
                except Exception as e:
                    logger.warning(f"HMM predict error: {e} — warm-up default.")
                    regime = None

            # ── Periodic HMM retraining ───────────────────────────────────────
            if candle_counter % retrain_interval == 0 and candle_counter > 0:
                logger.info(f"Scheduled HMM retrain at candle {candle_counter}...")
                hmm.fit(df)
                hmm.save()

            # ═══════════════════════════════════════════════════════════════════
            # RISK GATE 1 — Drawdown checks
            # ═══════════════════════════════════════════════════════════════════
            account = mt5.account_info()
            dd_status = dd_monitor.check(account.equity, daily_start_balance)

            if dd_status == DrawdownMonitor.ABSOLUTE_LIMIT:
                trade_mgr.close_all_positions(reason="Max account drawdown 15%")
                permanent_shutdown = True
                logger.critical(
                    "[MAX ACCOUNT DRAWDOWN] Permanent shutdown activated. "
                    "Restart the bot manually after reviewing the account."
                )
                continue

            if dd_status == DrawdownMonitor.DAILY_LIMIT and not trading_disabled:
                trade_mgr.close_all_positions(reason="Daily drawdown 3%")
                trading_disabled = True
                router.reset_state()
                logger.error(
                    "[DAILY DRAWDOWN LIMIT] Trading disabled for today. "
                    f"Equity: {account.equity:.2f}"
                )

            # ═══════════════════════════════════════════════════════════════════
            # RISK GATE 2 — Consecutive loss check
            # ═══════════════════════════════════════════════════════════════════
            if not trading_disabled and loss_monitor.check_new_closes():
                trading_disabled = True
                logger.error(
                    f"[CONSECUTIVE LOSS LIMIT] {loss_monitor.current_streak} losses "
                    f"in a row — TRADING DISABLED for today."
                )

            # ── Trading disabled (either gate) ─────────────────────────────────
            if trading_disabled:
                logger.debug("Trading disabled — monitoring only.")
                time.sleep(1)
                continue

            # ═══════════════════════════════════════════════════════════════════
            # RISK GATE 3 — Open position guard (no stacking)
            # ═══════════════════════════════════════════════════════════════════
            open_positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC)
            if open_positions:
                time.sleep(1)
                continue

            # ═══════════════════════════════════════════════════════════════════
            # RISK GATE 4 — Session filter
            # ═══════════════════════════════════════════════════════════════════
            if not check_trading_session(CONFIG, candle_time=candle_time):
                time.sleep(1)
                continue

            # ═══════════════════════════════════════════════════════════════════
            # RISK GATE 5 — HMM confidence + stability gates
            # ═══════════════════════════════════════════════════════════════════
            if hmm.is_trained and regime is not None:
                if not hmm.should_trade(regime, confidence):
                    time.sleep(1)
                    continue
                if not hmm.is_regime_stable(window=5):
                    logger.info("Regime unstable — skipping bar.")
                    time.sleep(1)
                    continue

            # ── Detect active session (for routing) ────────────────────────────
            session = detect_session(CONFIG, candle_time)

            # ── 4H bias — filter signals against higher-timeframe trend ──────
            bias_4h = data_feed.get_4h_bias(
                ema_fast=CONFIG.get("mtf", {}).get("bias_ema_fast", 50),
                ema_slow=CONFIG.get("mtf", {}).get("bias_ema_slow", 200),
            )

            # ── Signal generation via router ───────────────────────────────────
            signal = router.route(
                df,
                candle_time=candle_time,
                regime=regime,
                bias_4h=bias_4h,
                regime_probabilities=regime_probs,
            )

            prev = df.iloc[-2]
            logger.info(
                f"Candle {candle_time} | "
                f"Regime:{hmm.regime_name(regime) if regime is not None else 'warm-up'} | "
                f"Session:{session} | Signal:{signal['direction'] if signal else 'NONE'} | "
                f"Z:{prev.get('z_score', float('nan')):.2f} | "
                f"ADX:{prev.get('adx', float('nan')):.1f} | "
                f"RSI:{prev.get('rsi', float('nan')):.1f}"
            )

            # ── Order execution ────────────────────────────────────────────────
            if signal and signal.get("direction") in ("BUY", "SELL"):
                entry     = signal["entry"]
                sl        = signal["sl"]
                tp        = signal["tp"]
                atr       = signal["atr"]
                direction = signal["direction"]
                sl_dist   = abs(entry - sl)

                # Regime-scaled risk
                adjusted_risk = apply_regime_risk_scaling(
                    regime if regime is not None else REGIME_MEAN_REVERT,
                    BASE_RISK,
                )

                # Position sizing from config risk budget
                lots = calculate_position_size(
                    account.balance, adjusted_risk, sl_dist, SYMBOL
                )

                if lots > 0:
                    logger.info(
                        f"[TRADE EXECUTED] {direction} | "
                        f"Entry:{entry:.5f} SL:{sl:.5f} TP:{tp:.5f} | "
                        f"ATR:{atr:.6f} Lots:{lots:.2f} | "
                        f"Risk:{adjusted_risk:.3f} | "
                        f"Reason: {signal.get('reason','')}"
                    )
                    result = executor.place_trade(SYMBOL, direction, lots, sl, tp)

            time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Bot stopped manually (Ctrl+C).")
            break
        except Exception as e:
            logger.exception(f"Unexpected error in main loop: {e}")
            time.sleep(5)

    # ── Shutdown ───────────────────────────────────────────────────────────────
    if performance.lifetime_trade_count > 0:
        performance.lifetime_summary()
    connector.disconnect()
    logger.info("HMM Bot shut down cleanly.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
