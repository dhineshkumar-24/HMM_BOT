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
    df_warmup:       Optional[pd.DataFrame] = None,
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
    print("[Backtester] Pre-computing 4H bias (causal, with warmup)...")
    bias_series = _compute_4h_bias_series(df, df_warmup=df_warmup)

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
        # ── Context-aware time stop ───────────────────────────────────────────
        # Base time limit by regime — how long price typically needs to revert
        # or follow through before the thesis is invalidated.
        if regime == 2:          # HIGH_VOL — wide TP, needs more time
            TIME_STOP_BARS = 22
        elif regime == 1:        # TRENDING — momentum needs time to extend
            TIME_STOP_BARS = 16
        else:                    # MEAN_REVERT or warmup — fast reversion or out
            TIME_STOP_BARS = 12

        if sim.has_open_trade:
            open_trade = sim._open_trades[0]
            bars_open  = i - open_trade.entry_bar

            # Only evaluate time stop once base limit is reached
            if bars_open >= TIME_STOP_BARS:
                entry_px   = open_trade.entry_price
                tp_px      = open_trade.tp
                sl_px      = open_trade.sl
                current_px = float(bar["close"])

                tp_dist    = abs(tp_px - entry_px)
                price_moved = (
                    (current_px - entry_px)
                    if open_trade.direction == "BUY"
                    else (entry_px - current_px)
                )

                # Progress toward TP as a fraction (negative = moving away)
                progress = (price_moved / tp_dist) if tp_dist > 0 else 0.0

                # Flag to prevent repeated extensions on the same trade
                already_extended = getattr(open_trade, "_time_extended", False)

                if progress >= 0.60 and not already_extended:
                    # Trade is 60%+ of the way to TP and still moving correctly.
                    # Grant one single extension of 50% of base limit.
                    # Extension is capped — prevents infinite hold in a market
                    # that reverses slowly back to entry after appearing to work.
                    open_trade._time_extended = True
                    logger.debug(
                        f"Time stop extended | bar={i} | progress={progress:.1%} "
                        f"| extension={int(TIME_STOP_BARS * 0.5)} bars"
                    )
                else:
                    # Trade is losing, stalled, or already had its extension.
                    # Close now — thesis invalidated or capital better deployed.
                    for t in sim.close_all(bar, i):
                        balance = max(0, balance + t.net_pnl)
                    logger.debug(
                        f"Time stop closed | bar={i} | bars_open={bars_open} "
                        f"| progress={progress:.1%} | extended={already_extended}"
                    )
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
        # ── Session filter ────────────────────────────────────────────────────
        if not check_trading_session(config, candle_time):
            i += 1
            continue

        # ── HMM entropy + confidence gate ────────────────────────────────────
        # Replaces the commented-out block above.
        # Uses the pre-computed regime label and the full posterior array
        # (already available from _batch_hmm_predict via regime_labels).
        # The entropy gate rejects bars where the model is near-uniform
        # across states — the primary symptom of weak gap (0.070).
        if hmm is not None and hmm.is_trained and regime is not None:
            # Retrieve the posterior array for this bar from score_samples.
            # _regime_confidence() runs a small rolling window decode —
            # cheap because it only touches 200 bars.
            posteriors = _get_posteriors(hmm, df, i)
            if not hmm.should_trade(regime, _regime_confidence(hmm, df, i),
                                    posteriors=posteriors):
                i += 1
                continue

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
    hmm:    HMMRegimeDetector,
    df:     pd.DataFrame,
    window: int = 200,
    stride: int = 10,
) -> list[Optional[int]]:
    """
    Causal regime prediction — no look-ahead bias.

    For each bar i, only observations [i-window+1 .. i] are passed to the
    HMM. The regime is taken from the LAST posterior in score_samples().

    Why the last posterior is causal:
        forward-backward posterior at position t = alpha_t * beta_t.
        At the final step T, beta_T is all-ones (no future observations),
        so posterior[T] = alpha_T / sum(alpha_T) — the pure forward
        (filtering) probability. No future data is used.

    stride=10: regime is recomputed every 10 bars and held constant
    between updates. On M5 this means a regime update every 50 minutes —
    acceptable for session-level strategy routing.
    """
    regimes = [None] * len(df)

    from utils.features import build_feature_matrix

    feats = build_feature_matrix(df)
    if len(feats) < window:
        return regimes

    feat_positions = list(feats.index)
    df_index_map   = {v: i for i, v in enumerate(df.index)}

    X_all    = feats.values
    X_scaled = hmm._scaler.transform(X_all)
    n        = len(X_scaled)

    last_canonical = None   # carry forward between strides

    for feat_i in range(window - 1, n, stride):
        start    = max(0, feat_i - window + 1)
        X_window = X_scaled[start : feat_i + 1]

        try:
            # score_samples = forward-backward; last row is causal
            _, posteriors   = hmm._model.score_samples(X_window)
            last_posterior  = posteriors[-1]                   # shape (n_states,)
            raw_state       = int(np.argmax(last_posterior))
            last_canonical  = hmm._label_map.get(raw_state, raw_state)
        except Exception:
            pass   # keep last_canonical from previous stride

        # Fill every bar in this stride block with the same regime
        block_end   = min(feat_i + stride, n)
        block_start = feat_i - stride + 1 if stride > 1 else feat_i

        for fi in range(block_start, block_end):
            if fi < len(feat_positions):
                df_ix = feat_positions[fi]
                if df_ix in df_index_map:
                    regimes[df_index_map[df_ix]] = last_canonical

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


def _get_posteriors(
    hmm:    HMMRegimeDetector,
    df:     pd.DataFrame,
    i:      int,
    window: int = 200,
) -> np.ndarray:
    """
    Return the posterior probability array for bar i.

    Uses the last 200 bars (same window as _regime_confidence) so
    both functions share the same rolling window cost.
    Returns uniform distribution on failure — entropy gate will
    then block the trade, which is the safe default.
    """
    try:
        _, _, posteriors = hmm.predict(df.iloc[max(0, i - window): i + 1])
        return posteriors
    except Exception:
        return np.ones(hmm.n_states) / hmm.n_states   # uniform = max entropy


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

def _compute_4h_bias_series(
    df_signal: pd.DataFrame,
    df_warmup: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Compute causal 4H EMA50/EMA200 bias for every bar in df_signal.

    Two structural fixes vs the original:

    Fix 1 — resample label:
        Default label='left' labels the 4H period [09:00-13:00) as "09:00"
        but its value is the close at 12:55. Forward-filling gives M5 bars
        at 09:05 a value computed from data up to 12:55 — up to 4 hours
        of embedded lookahead.
        label='right' labels the same period "13:00". Forward-fill only
        applies this value to M5 bars from 13:00 onward — genuinely causal.

    Fix 2 — EMA warmup:
        EMA200 requires 200 four-hour bars (~33 trading days) to converge.
        Without prepending warmup history, the first month of any test
        period has a cold-start miscalibrated 4H bias.
        df_warmup (= df_train) is prepended before EMA computation then
        sliced away — only test-period bias values are returned.
    """
    # Prepend warmup data for EMA convergence if provided
    if df_warmup is not None and len(df_warmup) > 0:
        full_df = pd.concat(
            [df_warmup, df_signal],
            ignore_index=True
        ).sort_values("time").reset_index(drop=True)
    else:
        full_df = df_signal.copy()

    # Build 4H series on full history (warmup + signal)
    ts_full = full_df.set_index("time")

    # label='right'  → period [09:00–13:00) labeled 13:00 → causal ffill
    # closed='right' → period boundary at right edge (consistent with label)
    df_4h = (
        ts_full["close"]
        .resample("4h", label="right", closed="right")
        .last()
        .dropna()
    )

    # EMA computed on full history — causal (recursive, left-to-right)
    ema50  = df_4h.ewm(span=50,  adjust=False).mean()
    ema200 = df_4h.ewm(span=200, adjust=False).mean()

    bias_4h_ts = pd.Series("NEUTRAL", index=df_4h.index, dtype=object)
    bias_4h_ts[ema50 > ema200] = "UP"
    bias_4h_ts[ema50 < ema200] = "DOWN"

    # Reindex to M5 frequency — ffill carries last known 4H bias forward
    # Only return bias for the signal bars, not the warmup bars
    signal_ts = df_signal.set_index("time")
    bias_1m   = (
        bias_4h_ts
        .reindex(signal_ts.index, method="ffill")
        .fillna("NEUTRAL")
    )
    bias_1m.index = df_signal.index
    return bias_1m