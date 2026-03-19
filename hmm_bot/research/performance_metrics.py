"""
research/performance_metrics.py — Quantitative performance analytics.

Computes all standard quant metrics from a list of trade PnL values
or a per-bar equity curve.

Metrics:
    Total Trades, Win Rate, Profit Factor
    Average Win, Average Loss, Expectancy
    Max Drawdown (%), Sharpe Ratio, Sortino Ratio
    Equity curve array

Formula references:
    Sharpe  = mean(returns) / std(returns) * sqrt(N)
    Sortino = mean(returns) / downside_std * sqrt(N)
    PF      = gross_profit / gross_loss
    Expect  = (win_rate * avg_win) - (loss_rate * avg_loss)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# Validation thresholds (Step 11 requirements)
MIN_WIN_RATE      = 0.45
MIN_PROFIT_FACTOR = 1.20
MAX_DRAWDOWN      = 0.05
MIN_TRADES        = 30


def compute_metrics(
    trade_profits: list[float],
    annual_factor: int = 252,
    initial_balance: float = 10_000.0,
) -> dict:
    """
    Compute full performance metrics from a list of per-trade PnL values.

    Args:
        trade_profits: Realised net PnL for every closed trade (after costs).
        annual_factor: Periods per year for Sharpe/Sortino annualisation.
                       Use 252 for daily, 1440 for M1 bars, etc.
        initial_balance: Starting account balance for normalizing drawdown.

    Returns:
        Dict with all metrics. All rates are 0.0–1.0 (not percentage).
    """
    n = len(trade_profits)
    if n == 0:
        return _empty_metrics()

    profits  = np.array(trade_profits, dtype=float)
    wins     = profits[profits > 0]
    losses   = profits[profits < 0]

    total_trades = n
    win_count    = len(wins)
    loss_count   = len(losses)
    win_rate     = win_count / n

    gross_profit = float(wins.sum())  if len(wins)   > 0 else 0.0
    gross_loss   = float(-losses.sum()) if len(losses) > 0 else 0.0
    net_profit   = float(profits.sum())

    avg_win  = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_loss = float(-losses.mean()) if len(losses) > 0 else 0.0

    profit_factor = (
        gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")
    )
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # ── Returns per trade (normalise to % of initial capital if unknown: use raw) ──
    # We use raw PnL ratios scaled to unit size for Sharpe/Sortino
    mean_ret = float(profits.mean())
    std_ret  = float(profits.std())            if n > 1 else 0.0
    neg_dev  = float(losses.std())             if len(losses) > 1 else 0.0

    sharpe   = (mean_ret / std_ret  * math.sqrt(annual_factor)) if std_ret  > 0 else 0.0
    sortino  = (mean_ret / neg_dev  * math.sqrt(annual_factor)) if neg_dev  > 0 else 0.0

    # ── Equity curve & max drawdown ────────────────────────────────────────────
    equity_curve = np.cumsum(profits)
    max_dd = _max_drawdown(equity_curve, initial_balance=initial_balance)

    # Convention: CVaR returned as a positive number representing loss magnitude.
    # A CVaR of 85.0 means: on the worst 5% of trades, the average loss is $85.
    alpha_level  = 0.05
    sorted_pnl   = np.sort(profits)   # ascending: worst losses first
    cutoff_idx   = max(1, int(np.floor(n * alpha_level)))
    tail_losses  = sorted_pnl[:cutoff_idx]
    cvar_95      = float(-tail_losses.mean()) if len(tail_losses) > 0 else 0.0

    # VaR at 95%: the loss threshold (worst 5th percentile trade)
    var_95 = float(-np.percentile(profits, 5))

    return {
        "total_trades":   total_trades,
        "win_count":      win_count,
        "loss_count":     loss_count,
        "win_rate":       win_rate,
        "net_profit":     net_profit,
        "gross_profit":   gross_profit,
        "gross_loss":     gross_loss,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "profit_factor":  profit_factor,
        "expectancy":     expectancy,
        "max_drawdown":   max_dd,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "var_95":         var_95,
        "cvar_95":        cvar_95,
        "equity_curve":   equity_curve.tolist(),
    }

def validate_strategy(metrics: dict) -> dict:
    """
    Check if a strategy meets minimum quality thresholds.

    Validation rules (Step 11):
        Win Rate         >= 45%
        Profit Factor    >= 1.20
        Max Drawdown     <= 5%
        Minimum Trades   >= 30

    Returns:
        Dict with 'passed' bool and per-rule breakdown.
    """
    rules = {
        "min_trades":      metrics["total_trades"] >= MIN_TRADES,
        "win_rate":        metrics["win_rate"]      >= MIN_WIN_RATE,
        "profit_factor":   metrics["profit_factor"] >= MIN_PROFIT_FACTOR,
        "max_drawdown":    metrics["max_drawdown"]  <= MAX_DRAWDOWN,
    }
    passed = all(rules.values())
    return {"passed": passed, "rules": rules, **metrics}


def print_metrics(metrics: dict, label: str = "BACKTEST RESULTS") -> None:
    """Print a nicely formatted metrics table to stdout."""
    w = metrics.get("win_rate", 0)
    dd = metrics.get("max_drawdown", 0)
    pf = metrics.get("profit_factor", 0)
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"

    print("=" * 52)
    print(f"  {label}")
    print("=" * 52)
    print(f"  Total Trades    : {metrics.get('total_trades', 0)}")
    print(f"  Wins / Losses   : {metrics.get('win_count',0)} / {metrics.get('loss_count',0)}")
    print(f"  Win Rate        : {w:.1%}")
    print(f"  Net Profit      : {metrics.get('net_profit', 0):+.2f}")
    print(f"  Gross Profit    : {metrics.get('gross_profit', 0):.2f}")
    print(f"  Gross Loss      : {metrics.get('gross_loss', 0):.2f}")
    print(f"  Avg Win         : {metrics.get('avg_win', 0):.2f}")
    print(f"  Avg Loss        : {metrics.get('avg_loss', 0):.2f}")
    print(f"  Profit Factor   : {pf_str}")
    print(f"  Expectancy      : {metrics.get('expectancy', 0):.2f}")
    print(f"  Max Drawdown    : {dd:.2%}")
    print(f"  Sharpe Ratio    : {metrics.get('sharpe', 0):.3f}")
    print(f"  Sortino Ratio   : {metrics.get('sortino', 0):.3f}")
    print(f"  VaR 95%         : {metrics.get('var_95', 0):+.2f}")
    print(f"  CVaR 95%        : {metrics.get('cvar_95', 0):+.2f}")
    print("=" * 52)

    if "rules" in metrics:
        print(f"  Validation      : {'PASS' if metrics['passed'] else 'FAIL'}")
        for rule, ok in metrics["rules"].items():
            print(f"    {rule:20s}: {'OK' if ok else 'FAIL'}")
        print("=" * 52)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _max_drawdown(equity_curve: np.ndarray, initial_balance: float = 10_000.0) -> float:
    """
    Maximum percentage drawdown from running peak.
    Normalized by initial account balance — not by PnL peak.
    This prevents division by near-zero when all trades lose.
    """
    if len(equity_curve) == 0:
        return 0.0

    # equity_curve is cumulative PnL — convert to absolute balance
    abs_equity = initial_balance + equity_curve   # e.g. 10000 + (-9.29) = 9990.71

    peak = np.maximum.accumulate(abs_equity)

    # peak is always >= initial_balance at start, so no division by zero
    dd = (peak - abs_equity) / peak
    return float(dd.max()) if len(dd) > 0 else 0.0

def _empty_metrics() -> dict:
    return {
        "total_trades": 0, "win_count": 0, "loss_count": 0,
        "win_rate": 0.0, "net_profit": 0.0, "gross_profit": 0.0,
        "gross_loss": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "profit_factor": 0.0, "expectancy": 0.0,
        "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "var_95": 0.0, "cvar_95": 0.0,
        "equity_curve": [],
    }
