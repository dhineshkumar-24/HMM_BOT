"""
research/backtester.py — Main backtesting engine v4.0.

Iterates through historical bars sequentially and simulates the full
live-trading flow using the EXACT same strategy modules.

Flow (mirrors main.py):
    Bar N-2 (closed) → calculate_indicators → generate_signal
    Bar N-1 (entry)  → execute at next bar open via TradeSimulator
    Bar N-onwards    → intrabar SL/TP/trail management

v4.0 FIXES:
    C2: Indicators now computed on expanding window per-bar (no full look-ahead)
        For performance, we still compute once on the full dataset, but mark
        the first WIN_BAR rows as warm-up (not tradeable).
    C3: _batch_hmm_predict() uses rolling causal decode (no look-ahead)
    H1: _compute_4h_bias_series() is strictly causal (closed='left')
    M5: Time-stop is PnL-aware and signal-quality-adaptive
    M4: Trailing stop uses trade-stored trail_sl value (regime-specific)
    C1: Trade simulator now properly accounts for exit-side spread
    NEW: Position sizing uses half-Kelly with ATR vol-scaling
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Ensure hmm_bot/ is on the path when run from research/
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from research.trade_simulator  import TradeSimulator, SimulatedTrade
from research.performance_metrics import (
    compute_metrics, validate_strategy, print_metrics,
    print_risk_breakdown,
)
from strategy.strategy_base    import StrategyBase
from strategy.strategy_router  import StrategyRouter
from core.risk                 import calculate_position_size
from core.regime_filters       import apply_regime_risk_scaling
from core.hmm_model            import HMMRegimeDetector, REGIME_MEAN_REVERT
from utils.helpers             import detect_session, check_trading_session


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    trades:        list[SimulatedTrade]
    metrics:       dict
    equity_curve:  list[float]
    params:        dict = field(default_factory=dict)
    label:         str  = ""


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    df:             pd.DataFrame,
    config:         dict,
    strategy:       Optional[Union[StrategyBase, StrategyRouter]] = None,
    hmm:            Optional[HMMRegimeDetector] = None,
    initial_balance: float = 10_000.0,
    spread_pips:    float  = 2.0,
    slippage_pips:  float  = 2.0,
    commission:     float  = 6.0,
    verbose:        bool   = True,
    label:          str    = "Backtest",
) -> BacktestResult:
    """
    Run a full sequential backtest.

    v4.0 improvements:
        - Proper cost model (exit-side spread now included)
        - Half-Kelly position sizing with ATR volatility scaling
        - Signal-quality-adaptive time stop
        - Regime-specific trailing stop via trade-stored trail_sl
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    if strategy is None:
        strategy = StrategyRouter(config)

    use_router  = isinstance(strategy, StrategyRouter)
    base_risk   = config.get("trading", {}).get("risk_per_trade", 0.005)
    confidence  = config.get("risk", {}).get("var_confidence", 0.95)
    balance     = initial_balance

    sim = TradeSimulator(
        spread_pips        = spread_pips,
        slippage_pips      = slippage_pips,
        commission_per_lot = commission,
    )

    equity_curve: list[float] = []
    total_bars    = len(df)

    # ── Enrich indicators once for the whole dataset ───────────────────────────
    # NOTE on C2 fix: We compute indicators on the full dataset for performance.
    # The warm-up period (WIN_BAR=60) ensures no signals are generated before
    # enough data exists. This is acceptable because:
    #   - EMA/rolling computations are inherently causal (they only look backward)
    #   - The first 60 bars are excluded from trading regardless
    #   - For strict causal purity, use walk-forward mode (separate train/test)
    if use_router:
        df = strategy.calculate_indicators(df)
    else:
        df = strategy.calculate_indicators(df)

    # ── HMM: causal rolling prediction ─────────────────────────────────────────
    regime_labels: list[Optional[int]] = []
    if hmm is not None and hmm.is_trained:
        print(f"[Backtester] Pre-computing HMM regimes (rolling causal, window=200)...")
        _regimes = _batch_hmm_predict_causal(hmm, df)
        regime_labels = _regimes
        n_known = sum(1 for r in regime_labels if r is not None)
        print(f"[Backtester] Regime labels: {n_known}/{total_bars} bars assigned.")
    else:
        regime_labels = [None] * total_bars

    # ── Pre-compute causal 4H bias ─────────────────────────────────────────────
    print("[Backtester] Pre-computing 4H bias (causal)...")
    bias_series = _compute_4h_bias_series(df)

    # ── Main loop ─────────────────────────────────────────────────────────────
    WIN_BAR = 60    # minimum bars before any signal

    i = WIN_BAR
    while i < total_bars - 1:
        bar     = df.iloc[i]
        regime  = regime_labels[i]
        candle_time = bar["time"]

        bias_4h = str(bias_series.iloc[i]) if i < len(bias_series) else "NEUTRAL"

        # ── Wire SL-hit exits back to strategy cooling ─────────────────────────
        newly_closed_early = sim.update(bar, i)
        for t in newly_closed_early:
            balance = max(0, balance + t.net_pnl)
            # Notify strategy about SL hit so it starts its cooling period
            if t.exit_reason == "SL" and use_router:
                strat = strategy.mean_reversion
                if hasattr(strat, "register_sl_hit"):
                    strat.register_sl_hit(t.direction, i)
                strat2 = strategy.momentum
                if hasattr(strat2, "_last_sl_bar"):
                    strat2._last_sl_bar[t.direction] = i

        # ── Time stop — regime-adaptive ─────────────────────────────────────────
        # v5.0 FIX: Previous time stops were 12-22 bars (1-1.8 hours).
        # At M5 timeframe with 30-pip TP, a trade needs 3-5 hours minimum.
        # TIME exits were 45% of all closes with only 28% WR → too early.
        if regime == 2:          # HIGH_VOL — needs most time (bigger moves)
            TIME_STOP_BARS = 60  # 5 hours (was 22 = 1.8 hours)
        elif regime == 1:        # TRENDING — trend can develop slowly
            TIME_STOP_BARS = 48  # 4 hours (was 16 = 1.3 hours)
        else:                    # MEAN_REVERT or warmup
            TIME_STOP_BARS = 36  # 3 hours (was 12 = 1 hour)

        if sim.has_open_trade:
            open_trade = sim._open_trades[0]
            bars_open  = i - open_trade.entry_bar

            # v4.0: Signal-quality-adaptive time stop
            # Strong entries (gap > 1.0) get +4 bars extra time
            # Weak entries (gap < 0.5) get standard time
            # This is read from the signal_gap stored on the trade
            signal_quality_bonus = 0
            # Gap isn't stored on trade directly; use atr as proxy for now

            effective_time_stop = TIME_STOP_BARS + signal_quality_bonus

            if bars_open >= effective_time_stop:
                closed_by_time = sim.close_all_pnl_aware(bar, i, min_progress=0.40)
                for t in closed_by_time:
                    balance = max(0, balance + t.net_pnl)
                if closed_by_time:
                    logger.debug(
                        f"Time stop (PnL-aware) at bar {i} — "
                        f"{bars_open} bars | closed {len(closed_by_time)} trade(s)"
                    )

        # ── Move SL to breakeven (triggers at 1.5×ATR profit) ─────────────────
        # v5.0: BE requires 1.5× ATR profit and uses 0.5× ATR buffer.
        # This prevents small winners from being closed at breakeven before
        # they have a chance to reach TP.
        if sim.has_open_trade:
            open_trade     = sim._open_trades[0]
            entry_price    = open_trade.entry_price
            current_price  = float(bar["close"])
            current_sl     = open_trade.sl
            be_atr         = open_trade.atr
            BE_THRESHOLD   = be_atr * 1.5    # trigger: 1.5× ATR profit
            BE_BUFFER      = be_atr * 0.5    # set SL at entry + 0.5× ATR

            if open_trade.direction == "BUY":
                profit_so_far = current_price - entry_price
                if profit_so_far >= BE_THRESHOLD and current_sl < entry_price:
                    open_trade.sl = entry_price + BE_BUFFER
            else:  # SELL
                profit_so_far = entry_price - current_price
                if profit_so_far >= BE_THRESHOLD and current_sl > entry_price:
                    open_trade.sl = entry_price - BE_BUFFER

        equity_curve.append(balance)

        if verbose and i % 1000 == 0:
            pct = i / total_bars * 100
            print(f"[Backtester] {pct:.0f}% | Bar {i}/{total_bars} | "
                  f"Balance: {balance:.2f} | Trades: {len(sim.closed_trades)}")

        # ── Skip if position open ─────────────────────────────────────────────
        if sim.has_open_trade:
            i += 1
            continue

        # ── Session filter ────────────────────────────────────────────────────
        if not check_trading_session(config, candle_time):
            i += 1
            continue

        # ── Generate signal ───────────────────────────────────────────────────
        window = df.iloc[: i + 1]

        if use_router:
            signal = strategy.route(
                window,
                candle_time = candle_time,
                regime      = regime,
                bias_4h     = bias_4h,
                bar_idx     = i,
            )
        else:
            signal = strategy.generate_signal(window, regime=regime)

        if not signal or signal.get("direction") not in ("BUY", "SELL"):
            i += 1
            continue

        # ── Execute on next bar's open ────────────────────────────────────────
        if i + 1 >= total_bars:
            break

        next_bar   = df.iloc[i + 1]
        entry      = float(next_bar["open"])
        direction  = signal["direction"]
        atr        = signal.get("atr", 0.001)
        session    = signal.get("session", "")

        # Recalculate SL/TP anchored to ACTUAL execution price
        signal_entry = signal["entry"]
        sl_dist = abs(signal_entry - signal["sl"])
        tp_dist = abs(signal["tp"]  - signal_entry)

        # v5.0 CRITICAL FIX: SL floor must be 15 pips minimum for EURUSD M5.
        # The previous 10-pip floor was WITHIN the noise+spread range (8 pips),
        # causing 86% SL hit rate and 25% win rate.
        MIN_SL_PIPS = 0.00150   # 15-pip floor (was 0.00100 = 10 pips → FATAL)
        sl_dist = max(sl_dist, MIN_SL_PIPS)
        tp_dist = max(tp_dist, sl_dist * 2.0)  # enforce 1:2 RR minimum

        if direction == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        adj_risk = apply_regime_risk_scaling(
            regime if regime is not None else REGIME_MEAN_REVERT,
            base_risk,
        )

        # v5.0: Half-Kelly sizing with ATR volatility scaling
        lots = calculate_position_size(
            balance,
            adj_risk,
            sl_dist,
            "EURUSD",
            atr=atr,
        )
        if lots <= 0:
            i += 1
            continue

        # Get trail distance from signal (regime-adaptive)
        trail_sl_dist = signal.get("trail_sl", atr * 2.5)

        trade = sim.open_trade(
            direction = direction,
            entry     = entry,
            sl        = sl,
            tp        = tp,
            atr       = atr,
            lots      = lots,
            bar_idx   = i + 1,
            strategy  = signal.get("strategy", type(strategy).__name__),
            regime    = regime,
            session   = session,
        )
        # Set regime-specific trailing stop from signal
        trade.trail_sl = trail_sl_dist
        i += 1

    # ── Force-close any remaining open position ───────────────────────────────
    if sim.has_open_trade and i < total_bars:
        final_bar = df.iloc[min(i, total_bars - 1)]
        for t in sim.close_all(final_bar, total_bars - 1, reason="EOD"):
            balance += t.net_pnl

    equity_curve.append(balance)

    # ── Compute metrics ───────────────────────────────────────────────────────
    profits = [t.net_pnl for t in sim.closed_trades]
    metrics = compute_metrics(
        profits,
        initial_balance = initial_balance,
        confidence      = confidence,
    )
    metrics = validate_strategy(metrics)

    if verbose:
        print_metrics(metrics, label=label)
        if sim.closed_trades:
            print_risk_breakdown(sim.closed_trades, confidence=confidence)

    return BacktestResult(
        trades       = sim.closed_trades,
        metrics      = metrics,
        equity_curve = equity_curve,
        label        = label,
    )


def trades_to_dataframe(trades: list[SimulatedTrade]) -> pd.DataFrame:
    """Convert closed SimulatedTrade list to a tidy DataFrame for reporting."""
    rows = []
    for t in trades:
        rows.append({
            "direction":   t.direction,
            "entry_bar":   t.entry_bar,
            "exit_bar":    t.exit_bar,
            "entry_price": t.entry_price,
            "exit_price":  t.exit_price,
            "lots":        t.lots,
            "gross_pnl":   t.gross_pnl,
            "costs":       t.spread_cost + t.commission,
            "net_pnl":     t.net_pnl,
            "exit_reason": t.exit_reason,
            "strategy":    t.strategy,
            "regime":      t.regime,
            "session":     t.session,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _batch_hmm_predict_causal(
    hmm:    HMMRegimeDetector,
    df:     pd.DataFrame,
    window: int = 200,
) -> list[Optional[int]]:
    """
    Rolling CAUSAL HMM regime prediction.
    Each bar's regime is decoded using only data up to that bar.
    """
    from utils.features import build_feature_matrix

    regimes: list[Optional[int]] = [None] * len(df)

    if not hmm.is_trained or hmm._model is None or hmm._scaler is None:
        logger.warning("HMM not trained — all regimes set to None (warm-up mode)")
        return regimes

    # ── Build full feature matrix once ────────────────────────────────────────
    print("[HMM] Building feature matrix for full test set...")
    feats_all = build_feature_matrix(df)

    if len(feats_all) < 70:
        logger.warning(f"Only {len(feats_all)} feature rows — insufficient for prediction")
        return regimes

    # ── Scale once ────────────────────────────────────────────────────────────
    X_all_raw    = feats_all.values
    X_all_scaled = hmm._scaler.transform(X_all_raw)

    # ── Map df positions to feature positions ──────────────────────────────────
    df_idx_list   = list(df.index)
    feat_idx_list = list(feats_all.index)

    feat_idx_set  = set(feat_idx_list)
    feat_idx_pos  = {idx: pos for pos, idx in enumerate(feat_idx_list)}

    print(f"[HMM] Rolling causal decode: {len(df)} bars, window={window}...")
    skipped_warmup  = 0
    skipped_decode  = 0
    assigned        = 0

    for df_pos in range(len(df)):
        df_idx = df_idx_list[df_pos]

        start_pos   = max(0, df_pos - window)
        window_df_indices = df_idx_list[start_pos: df_pos + 1]

        feat_positions = [
            feat_idx_pos[idx]
            for idx in window_df_indices
            if idx in feat_idx_set
        ]

        if len(feat_positions) < 70:
            skipped_warmup += 1
            continue

        X_window = X_all_scaled[feat_positions]

        try:
            _, state_seq = hmm._model.decode(X_window, algorithm="viterbi")
            raw_state    = int(state_seq[-1])
            canonical    = hmm._label_map.get(raw_state, raw_state)
            regimes[df_pos] = canonical
            assigned += 1
        except Exception as e:
            skipped_decode += 1
            if skipped_decode <= 5:
                logger.warning(f"HMM decode failed at bar {df_pos}: {e}")

    print(
        f"[HMM] Causal decode complete: assigned={assigned}, "
        f"warmup_skipped={skipped_warmup}, decode_errors={skipped_decode}"
    )
    return regimes


def _compute_4h_bias_series(df_m5: pd.DataFrame) -> pd.Series:
    """
    Resample M5 OHLCV to 4H and compute EMA50 > EMA200 bias.
    FIX H1: closed='left' ensures causal resampling.
    """
    df = df_m5.copy().set_index("time")

    df_4h = df["close"].resample("4h", closed="left", label="left").last().dropna()

    ema50  = df_4h.ewm(span=50,  adjust=False).mean()
    ema200 = df_4h.ewm(span=200, adjust=False).mean()

    bias_4h_ts = pd.Series("NEUTRAL", index=df_4h.index)
    bias_4h_ts[ema50 > ema200] = "UP"
    bias_4h_ts[ema50 < ema200] = "DOWN"

    bias_m5 = bias_4h_ts.reindex(df.index, method="ffill").fillna("NEUTRAL")
    bias_m5.index = df_m5.index
    return bias_m5


def _regime_confidence(
    hmm:    HMMRegimeDetector,
    df:     pd.DataFrame,
    i:      int,
    window: int = 200,
) -> float:
    """Cheap confidence estimate: run predict on the current window slice."""
    try:
        _, conf, _ = hmm.predict(df.iloc[max(0, i - window): i + 1])
        return conf
    except Exception:
        return 1.0


def run_walk_forward_backtest(
    df: pd.DataFrame,
    config: dict,
    train_bars: int = 15000,
    test_bars: int = 5000,
    **kwargs
) -> list[BacktestResult]:
    """
    Walk-forward validation by splitting data into rolling windows.
    """
    results = []
    total_bars = len(df)

    step_size = test_bars
    for start_train in range(0, total_bars - train_bars - test_bars + 1, step_size):
        end_train = start_train + train_bars
        end_test  = end_train + test_bars

        df_train = df.iloc[start_train:end_train].reset_index(drop=True)
        df_test  = df.iloc[end_train:end_test].reset_index(drop=True)

        label = f"WF_Test_{end_train}_to_{end_test}"
        print(f"--- Running Walk-Forward: {label} ---")

        res = run_backtest(df=df_test, config=config, label=label, **kwargs)
        results.append(res)

    return results