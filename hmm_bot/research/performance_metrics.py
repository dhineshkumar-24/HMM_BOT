"""
research/performance_metrics.py — Quantitative performance analytics v4.0.

Computes all standard quant metrics from a list of trade PnL values
or a per-bar equity curve.

v4.0 additions:
    Calmar Ratio, Recovery Factor, Monte Carlo p-value
    Transaction cost sensitivity analysis

Metrics:
    Total Trades, Win Rate, Profit Factor
    Average Win, Average Loss, Expectancy
    Max Drawdown (%), Sharpe Ratio, Sortino Ratio
    Calmar Ratio, Recovery Factor (NEW)
    VaR (95%), CVaR (95%)
    Monte Carlo p-value (NEW)
    Per-strategy and per-session breakdowns

Formula references:
    Sharpe  = mean(returns) / std(returns) * sqrt(N)
    Sortino = mean(returns) / downside_std * sqrt(N)
    PF      = gross_profit / gross_loss
    Expect  = (win_rate * avg_win) - (loss_rate * avg_loss)
    Calmar  = annualized_return / max_drawdown
    VaR     = -percentile(returns, 5%) at 95% confidence
    CVaR    = mean of returns worse than VaR (expected shortfall)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd


# Validation thresholds
MIN_WIN_RATE      = 0.45
MIN_PROFIT_FACTOR = 1.20
MAX_DRAWDOWN      = 0.05
MIN_TRADES        = 30


def compute_metrics(
    trade_profits: list[float],
    annual_factor: int = 252,
    initial_balance: float = 10_000.0,
    confidence: float = 0.95,
) -> dict:
    """
    Compute full performance metrics from a list of per-trade PnL values.

    v4.0: Added Calmar ratio and recovery factor.
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

    mean_ret = float(profits.mean())
    std_ret  = float(profits.std())            if n > 1 else 0.0
    neg_dev  = float(losses.std())             if len(losses) > 1 else 0.0

    sharpe   = (mean_ret / std_ret  * math.sqrt(annual_factor)) if std_ret  > 0 else 0.0
    sortino  = (mean_ret / neg_dev  * math.sqrt(annual_factor)) if neg_dev  > 0 else 0.0

    # ── Equity curve & max drawdown ────────────────────────────────────────────
    equity_curve = np.cumsum(profits)
    max_dd = _max_drawdown(equity_curve, initial_balance=initial_balance)

    # ── Calmar Ratio (NEW v4.0) ───────────────────────────────────────────────
    # Calmar = annualized return / max drawdown
    total_return_pct = net_profit / initial_balance
    calmar = total_return_pct / max_dd if max_dd > 0.001 else 0.0

    # ── Recovery Factor (NEW v4.0) ────────────────────────────────────────────
    # Recovery Factor = total profit / max dollar drawdown
    max_dd_dollars = max_dd * initial_balance
    recovery_factor = net_profit / max_dd_dollars if max_dd_dollars > 0 else 0.0

    # ── VaR / CVaR ────────────────────────────────────────────────────────────
    var_cvar = compute_var_cvar(trade_profits, confidence=confidence)

    return {
        "total_trades":    total_trades,
        "win_count":       win_count,
        "loss_count":      loss_count,
        "win_rate":        win_rate,
        "net_profit":      net_profit,
        "gross_profit":    gross_profit,
        "gross_loss":      gross_loss,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "profit_factor":   profit_factor,
        "expectancy":      expectancy,
        "max_drawdown":    max_dd,
        "sharpe":          sharpe,
        "sortino":         sortino,
        "calmar":          calmar,
        "recovery_factor": recovery_factor,
        "equity_curve":    equity_curve.tolist(),
        **var_cvar,
    }


def compute_var_cvar(
    trade_profits: list[float],
    confidence: float = 0.95,
) -> dict:
    """
    Historical VaR and CVaR (Expected Shortfall).
    """
    key_suffix = int(confidence * 100)

    if not trade_profits or len(trade_profits) < 5:
        return {
            f"var_{key_suffix}":  0.0,
            f"cvar_{key_suffix}": 0.0,
        }

    profits = np.array(trade_profits, dtype=float)
    alpha   = 1.0 - confidence

    var_threshold = float(np.percentile(profits, alpha * 100))
    var  = -var_threshold

    tail_losses = profits[profits <= var_threshold]
    cvar = float(-tail_losses.mean()) if len(tail_losses) > 0 else var

    return {
        f"var_{key_suffix}":  round(var,  4),
        f"cvar_{key_suffix}": round(cvar, 4),
    }


def monte_carlo_pvalue(
    trade_profits: list[float],
    n_simulations: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo bootstrap p-value for strategy edge.

    Method:
        1. Randomly reshuffle the trade PnL sequence N times
        2. For each reshuffle, compute total return
        3. p-value = fraction of reshuffles with return >= actual return

    Interpretation:
        p < 0.05:  Strategy edge is statistically significant at 95% level
        p < 0.01:  Highly significant (very unlikely due to luck)
        p > 0.10:  Edge could be due to random ordering

    Args:
        trade_profits: List of net P&L per trade.
        n_simulations: Number of bootstrap iterations.
        seed:          Random seed for reproducibility.

    Returns:
        Dict with 'mc_pvalue', 'actual_return', 'mean_shuffle_return'.
    """
    if len(trade_profits) < 10:
        return {
            "mc_pvalue":             1.0,
            "actual_return":         0.0,
            "mean_shuffle_return":   0.0,
        }

    rng = np.random.default_rng(seed)
    profits = np.array(trade_profits, dtype=float)
    actual_return = float(profits.sum())

    shuffle_returns = np.zeros(n_simulations)
    for i in range(n_simulations):
        # Randomly reshuffle the trade sequence (same trades, different order)
        shuffled = rng.permutation(profits)
        # Compute max drawdown-adjusted return (Sharpe-like metric)
        equity = np.cumsum(shuffled)
        shuffle_returns[i] = float(equity[-1])

    # p-value: fraction of simulations that beat or match actual return
    p_value = float(np.mean(shuffle_returns >= actual_return))

    return {
        "mc_pvalue":           round(p_value, 4),
        "actual_return":       round(actual_return, 2),
        "mean_shuffle_return": round(float(shuffle_returns.mean()), 2),
    }


def compute_risk_by_group(
    trades,
    group_field: str,
    confidence: float = 0.95,
) -> dict:
    """
    Compute VaR and CVaR grouped by strategy or session.
    """
    groups: dict[str, list[float]] = defaultdict(list)

    for t in trades:
        key = str(getattr(t, group_field, "unknown"))
        groups[key].append(t.net_pnl)

    result = {}
    for name, pnls in groups.items():
        vc = compute_var_cvar(pnls, confidence=confidence)
        profits_arr = np.array(pnls)
        wins = profits_arr[profits_arr > 0]
        result[name] = {
            "total_trades": len(pnls),
            "win_count":    len(wins),
            "win_rate":     len(wins) / len(pnls) if pnls else 0.0,
            "net_profit":   float(profits_arr.sum()),
            **vc,
        }
    return result


def validate_strategy(metrics: dict) -> dict:
    """
    Check if a strategy meets minimum quality thresholds.

    Validation rules:
        Win Rate         >= 45%
        Profit Factor    >= 1.20
        Max Drawdown     <= 5%
        Minimum Trades   >= 30
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
    var  = metrics.get("var_95",  None)
    cvar = metrics.get("cvar_95", None)
    calmar = metrics.get("calmar", 0)
    recovery = metrics.get("recovery_factor", 0)

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
    print(f"  Calmar Ratio    : {calmar:.3f}")
    print(f"  Recovery Factor : {recovery:.2f}")
    if var is not None:
        print(f"  VaR  (95%)      : ${var:.2f}")
    if cvar is not None:
        print(f"  CVaR (95%)      : ${cvar:.2f}")
    print("=" * 52)

    if "rules" in metrics:
        print(f"  Validation      : {'PASS' if metrics['passed'] else 'FAIL'}")
        for rule, ok in metrics["rules"].items():
            print(f"    {rule:20s}: {'OK' if ok else 'FAIL'}")
        print("=" * 52)


def print_risk_breakdown(
    trades,
    confidence: float = 0.95,
) -> None:
    """Print VaR/CVaR breakdown per strategy and per session."""
    print("\n" + "=" * 52)
    print("  RISK BREAKDOWN — by Strategy")
    print("=" * 52)
    by_strategy = compute_risk_by_group(trades, "strategy", confidence)
    for name, d in by_strategy.items():
        key = int(confidence * 100)
        print(f"  {name}:")
        print(f"    Trades={d['total_trades']} WR={d['win_rate']:.1%} "
              f"Net=${d['net_profit']:+.2f}")
        print(f"    VaR_{key}=${d.get(f'var_{key}',0):.2f}  "
              f"CVaR_{key}=${d.get(f'cvar_{key}',0):.2f}")

    print("\n" + "=" * 52)
    print("  RISK BREAKDOWN — by Session")
    print("=" * 52)
    by_session = compute_risk_by_group(trades, "session", confidence)
    for name, d in by_session.items():
        key = int(confidence * 100)
        print(f"  {name}:")
        print(f"    Trades={d['total_trades']} WR={d['win_rate']:.1%} "
              f"Net=${d['net_profit']:+.2f}")
        print(f"    VaR_{key}=${d.get(f'var_{key}',0):.2f}  "
              f"CVaR_{key}=${d.get(f'cvar_{key}',0):.2f}")
    print("=" * 52 + "\n")


def transaction_cost_sensitivity(
    trade_profits: list[float],
    initial_balance: float = 10_000.0,
    cost_multipliers: list[float] = None,
) -> dict:
    """
    Test strategy robustness across different transaction cost levels.

    Computes key metrics at 1×, 1.5×, 2× cost to find the breakeven
    transaction cost multiplier.

    Args:
        trade_profits:   Existing net PnL per trade (already includes costs).
        initial_balance: Starting balance for normalization.
        cost_multipliers: List of cost multipliers to test (default [1.0, 1.5, 2.0]).

    Returns:
        Dict mapping multiplier → metrics dict.
    """
    if cost_multipliers is None:
        cost_multipliers = [1.0, 1.25, 1.5, 2.0]

    # Estimate per-trade cost from existing PnL (heuristic)
    # Assume gross_pnl = net_pnl + estimated_cost
    # This is approximate — for exact analysis, use the trade objects directly
    results = {}
    profits = np.array(trade_profits, dtype=float)

    for mult in cost_multipliers:
        # Scale the losses more aggressively (costs hit losers harder)
        adjusted = profits.copy()
        if mult > 1.0:
            # Add extra cost proportional to the average loss
            avg_loss = float(np.abs(profits[profits < 0]).mean()) if np.any(profits < 0) else 10.0
            extra_cost = avg_loss * (mult - 1.0) * 0.5  # rough approximation
            adjusted -= extra_cost

        metrics = compute_metrics(
            adjusted.tolist(),
            initial_balance=initial_balance,
        )
        results[mult] = {
            "net_profit":    metrics["net_profit"],
            "win_rate":      metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "sharpe":        metrics["sharpe"],
            "max_drawdown":  metrics["max_drawdown"],
        }

    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _max_drawdown(equity_curve: np.ndarray, initial_balance: float = 10_000.0) -> float:
    """
    Maximum percentage drawdown from running peak.
    Normalized by initial account balance.
    """
    if len(equity_curve) == 0:
        return 0.0

    abs_equity = initial_balance + equity_curve
    peak = np.maximum.accumulate(abs_equity)
    dd = (peak - abs_equity) / peak
    return float(dd.max()) if len(dd) > 0 else 0.0


def _empty_metrics() -> dict:
    return {
        "total_trades": 0, "win_count": 0, "loss_count": 0,
        "win_rate": 0.0, "net_profit": 0.0, "gross_profit": 0.0,
        "gross_loss": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "profit_factor": 0.0, "expectancy": 0.0,
        "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "calmar": 0.0, "recovery_factor": 0.0,
        "equity_curve": [],
        "var_95": 0.0, "cvar_95": 0.0,
    }
