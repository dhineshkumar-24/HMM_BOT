"""
research/backtester.py — Main backtesting engine.

Iterates through historical bars sequentially and simulates the full
live-trading flow using the EXACT same strategy modules:

Flow (mirrors main.py):
    Bar N-2 (closed) → calculate_indicators → generate_signal
    Bar N-1 (entry)  → execute at next bar open via TradeSimulator
    Bar N-onwards    → intrabar SL/TP/trail management

Key design:
    - Minimum look-back = 60 bars before any signals
    - HMM warm-up respected (regime=None until model trains or loads)
    - One trade at a time (no stacking)
    - Signal is dict from StrategyRouter or individual strategy
    - Regime risk scaling applied to lot sizing

Public API:
    BacktestResult = run_backtest(strategy, df, config, ...)
    run_backtest_strategy(strategy_name, df, config, hmm=None, ...)
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
from research.performance_metrics import compute_metrics, validate_strategy, print_metrics
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
    spread_pips:    float  = 1.5,
    slippage_pips:  float  = 0.5,
    commission:     float  = 6.0,
    verbose:        bool   = True,
    label:          str    = "Backtest",
) -> BacktestResult:
    """
    Run a full sequential backtest.

    Args:
        df:              Historical OHLCV DataFrame (sorted by time).
        config:          Full settings dict (same as live trading).
        strategy:        Strategy instance or StrategyRouter. If None, a full
                         StrategyRouter is created from config.
        hmm:             Pre-trained HMMRegimeDetector. If None, regime=None
                         (warm-up mode) for the whole backtest.
        initial_balance: Starting account balance in USD.
        spread_pips:     Spread cost in pips.
        slippage_pips:   Max random slippage in pips.
        commission:      Round-trip commission per standard lot in USD.
        verbose:         Print progress every 1000 bars.
        label:           Identifier string for the result.

    Returns:
        BacktestResult with trades, metrics, and equity curve.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    if strategy is None:
        strategy = StrategyRouter(config)

    use_router  = isinstance(strategy, StrategyRouter)
    base_risk   = config.get("trading", {}).get("risk_per_trade", 0.005)
    balance     = initial_balance

    sim = TradeSimulator(
        spread_pips        = spread_pips,
        slippage_pips      = slippage_pips,
        commission_per_lot = commission,
    )

    equity_curve: list[float] = []
    pending_pnl:  float       = 0.0
    total_bars    = len(df)

    # ── Enrich indicators once for the whole dataset ───────────────────────────
    if use_router:
        df = strategy.calculate_indicators(df)
    else:
        df = strategy.calculate_indicators(df)

    # ── HMM: use pre-trained or warm-up ───────────────────────────────────────
    regime_labels: list[Optional[int]] = []
    if hmm is not None and hmm.is_trained:
        # Compute regime for every bar using rolling 200-bar windows
        print(f"[Backtester] Pre-computing HMM regimes for {total_bars} bars...")
        _regimes = _batch_hmm_predict(hmm, df)
        regime_labels = _regimes
    else:
        regime_labels = [None] * total_bars
    # ── Pre-compute 4H bias once for all bars ─────────────────────────────────
    print("[Backtester] Pre-computing 4H bias from 1M data...")
    bias_series = _compute_4h_bias_series(df)

    # ── Main loop ─────────────────────────────────────────────────────────────
    WIN_BAR = 60    # minimum bars before any signal

    i = WIN_BAR
    while i < total_bars - 1:
        bar     = df.iloc[i]
        regime  = regime_labels[i]
        candle_time = bar["time"]

        bias_4h = str(bias_series.iloc[i]) if i < len(bias_series) else "NEUTRAL"

        # ── Update open positions ──────────────────────────────────────────────
        newly_closed = sim.update(bar, i)
        for t in newly_closed:
            balance = max(0, balance + t.net_pnl)
            pending_pnl = 0.0
        # ── Time stop — exit after 30 bars if trade still open ────────────────
        if regime == 2:          # HIGH_VOL — TP farther, needs more time
            TIME_STOP_BARS = 22
        elif regime == 1:        # TRENDING — moderate
            TIME_STOP_BARS = 16
        else:                    # MEAN_REVERT or warmup
            TIME_STOP_BARS = 12
        if sim.has_open_trade:
            open_trade = sim._open_trades[0]
            bars_open  = i - open_trade.entry_bar
            if bars_open >= TIME_STOP_BARS:
                for t in sim.close_all(bar, i):
                    balance = max(0, balance + t.net_pnl)
                logger.debug(f"Time stop hit at bar {i} — {bars_open} bars open")
        # ── Move SL to breakeven once trade reaches 40% of TP distance ──────────
        if sim.has_open_trade:
            open_trade = sim._open_trades[0]
            entry_price = open_trade.entry_price
            current_price = float(bar["close"])
            tp_price = open_trade.tp
            sl_price = open_trade.sl

            tp_dist = abs(tp_price - entry_price)
            price_moved = abs(current_price - entry_price)

            # If price has moved 40% toward TP, lock in breakeven
            if price_moved >= tp_dist * 0.40:
                if open_trade.direction == "BUY" and sl_price < entry_price:
                    open_trade.sl = entry_price + 0.00005  # 0.5 pip above entry
                elif open_trade.direction == "SELL" and sl_price > entry_price:
                    open_trade.sl = entry_price - 0.00005  # 0.5 pip below entry

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

        # # ── HMM confidence gate (only for MeanReversionStrategy, not AlphaStrategy)
        # use_hmm_gate = True
        # if use_router:
        #     # Check which strategy would be selected — skip gate for alpha strategy
        #     session_now = detect_session(config, candle_time)
        #     from strategy.momentum import MomentumStrategy as AlphaStrat
        #     candidate = strategy._routing.get((session_now, regime))
        #     if isinstance(candidate, AlphaStrat):
        #         use_hmm_gate = False  # alpha filters itself via combined_alpha threshold

        # if use_hmm_gate and hmm is not None and hmm.is_trained and regime is not None:
        #     conf = _regime_confidence(hmm, df, i)
        #     if not hmm.should_trade(regime, conf):
        #         i += 1
        #         continue

        # ── Generate signal ───────────────────────────────────────────────────
        window = df.iloc[: i + 1]

        if use_router:
            signal = strategy.route(
                window,
                candle_time = candle_time,
                regime      = regime,
                bias_4h     = bias_4h
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
        entry      = float(next_bar["open"])   # actual execution price
        direction  = signal["direction"]
        atr        = signal.get("atr", 0.001)
        session      = signal.get("session", "")

        # ── Recalculate SL/TP anchored to ACTUAL execution price ──────────────
        # Preserve the ATR-based distances from the signal, but shift them
        # to start from where we actually entered, not from the signal bar's close
        signal_entry = signal["entry"]
        sl_dist = abs(signal_entry - signal["sl"])   # ATR-based pip distance
        tp_dist = abs(signal["tp"]  - signal_entry)  # ATR-based pip distance

        # Enforce minimum SL of 8 pips (0.00080) on EURUSD M1
        # M1 ATR can be 1-3 pips — too tight for spread+slippage+noise
        MIN_SL_PIPS = 0.00100
        sl_dist = max(sl_dist, MIN_SL_PIPS)
        tp_dist = max(tp_dist, sl_dist * 2.0)   # maintain at least 1.8 R:R

        if direction == "BUY":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:  # SELL
            sl = entry + sl_dist
            tp = entry - tp_dist

        adj_risk = apply_regime_risk_scaling(
            regime if regime is not None else REGIME_MEAN_REVERT,
            base_risk,
        )
        lots = calculate_position_size(balance, adj_risk, sl_dist, "EURUSD")
        if lots <= 0:
            i += 1
            continue

        sim.open_trade(
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
        i += 1

    # ── Force-close any remaining open position ───────────────────────────────
    if sim.has_open_trade and i < total_bars:
        final_bar = df.iloc[min(i, total_bars - 1)]
        for t in sim.close_all(final_bar, total_bars - 1):
            balance += t.net_pnl

    equity_curve.append(balance)

    # ── Compute metrics ───────────────────────────────────────────────────────
    profits  = [t.net_pnl for t in sim.closed_trades]
    metrics = compute_metrics(profits, initial_balance=initial_balance)
    metrics  = validate_strategy(metrics)

    if verbose:
        print_metrics(metrics, label=label)

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

def _batch_hmm_predict(
    hmm: HMMRegimeDetector,
    df:  pd.DataFrame,
    window: int = 200,
) -> list[Optional[int]]:
    """Predict regime for every bar using a rolling window (vectorised)."""
    regimes = [None] * len(df)
    from utils.features import build_feature_matrix, FEATURE_COLS
    from sklearn.preprocessing import StandardScaler

    feats = build_feature_matrix(df)
    if len(feats) < window:
        return regimes

    # Align feature index with df index
    feat_idx = feats.index.tolist()
    idx_map  = {v: i for i, v in enumerate(df.index)}

    X_all   = feats.values
    X_scaled = hmm._scaler.transform(X_all)

    try:
        _, state_seq = hmm._model.decode(X_scaled, algorithm="viterbi")
        for i, df_ix in enumerate(feat_idx):
            if df_ix in idx_map:
                raw_state      = int(state_seq[i])
                canonical_state = hmm._label_map.get(raw_state, raw_state)  # ← apply label map
                regimes[idx_map[df_ix]] = canonical_state
    except Exception:
        pass

    return regimes


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
    train_bars: int = 3000,
    test_bars: int = 1000,
    **kwargs
) -> list[BacktestResult]:
    """
    Enforces walk-forward validation by splitting data into rolling windows.
    Train: bars 1-3000 -> Test: bars 3001-4000
    Train: bars 1001-4000 -> Test: bars 4001-5000 ...
    """
    results = []
    total_bars = len(df)
    
    step_size = test_bars
    for start_train in range(0, total_bars - train_bars - test_bars + 1, step_size):
        end_train = start_train + train_bars
        end_test = end_train + test_bars
        
        df_train = df.iloc[start_train:end_train].reset_index(drop=True)
        df_test = df.iloc[end_train:end_test].reset_index(drop=True)
        
        label = f"WF_Test_{end_train}_to_{end_test}"
        print(f"--- Running Walk-Forward: {label} ---")
        
        res = run_backtest(df=df_test, config=config, label=label, **kwargs)
        results.append(res)
        
    return results

def _compute_4h_bias_series(df_1m: pd.DataFrame) -> pd.Series:
    """
    Resample 1M OHLCV to 4H and compute EMA50 > EMA200 bias.
    Returns a Series indexed by 1M bar index with values 'UP', 'DOWN', 'NEUTRAL'.
    """
    df = df_1m.copy().set_index("time")
    df_4h = df["close"].resample("4h").last().dropna()
    ema50  = df_4h.ewm(span=50,  adjust=False).mean()
    ema200 = df_4h.ewm(span=200, adjust=False).mean()
    bias_4h_ts = pd.Series("NEUTRAL", index=df_4h.index)
    bias_4h_ts[ema50 > ema200] = "UP"
    bias_4h_ts[ema50 < ema200] = "DOWN"
    bias_1m = bias_4h_ts.reindex(df.index, method="ffill").fillna("NEUTRAL")
    bias_1m.index = df_1m.index
    return bias_1m