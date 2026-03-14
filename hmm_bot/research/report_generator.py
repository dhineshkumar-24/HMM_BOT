"""
research/report_generator.py — Performance report and chart generator.

Outputs:
    1. Console summary (via performance_metrics.print_metrics)
    2. CSV trade log   → research/reports/<label>_trades.csv
    3. CSV metrics     → research/reports/<label>_metrics.csv
    4. Equity curve PNG chart
    5. Drawdown curve PNG chart

Requires: matplotlib (pip install matplotlib)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

REPORTS_DIR = os.path.join(_HERE, "reports")


def generate_report(
    result,                           # BacktestResult
    label:       str = "backtest",
    save_charts: bool = True,
    show_charts: bool = False,
) -> str:
    """
    Generate a full performance report from a BacktestResult.

    Args:
        result:      BacktestResult from run_backtest().
        label:       Report identifier for file names.
        save_charts: Save PNG charts to research/reports/.
        show_charts: Display charts interactively (requires display).

    Returns:
        Path to the generated reports folder.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug  = f"{label.replace(' ', '_')}_{ts}"

    _save_trades_csv(result, slug)
    _save_metrics_csv(result, slug)

    if save_charts:
        _plot_equity_curve(result, slug, show=show_charts)
        _plot_drawdown(result, slug, show=show_charts)
        _plot_trade_distribution(result, slug, show=show_charts)

    print(f"[Report] Saved to {REPORTS_DIR}/{slug}_*")
    return os.path.join(REPORTS_DIR, slug)


def generate_walk_forward_report(
    window_results:  list,            # list[WindowResult]
    aggregate:       dict,
    label:           str  = "walk_forward",
    save_charts:     bool = True,
    show_charts:     bool = False,
) -> str:
    """Generate a combined walk-forward report across all windows."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"wf_{label}_{ts}"

    # Aggregate metrics CSV
    agg_df = pd.DataFrame([aggregate])
    agg_df.to_csv(os.path.join(REPORTS_DIR, f"{slug}_aggregate.csv"), index=False)

    # Per-window metrics
    rows = []
    for wr in window_results:
        row = {
            "window_id":   wr.window_id,
            "test_start":  wr.test_start,
            "test_end":    wr.test_end,
        }
        row.update({k: v for k, v in wr.result.metrics.items() if k != "equity_curve"})
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(REPORTS_DIR, f"{slug}_windows.csv"), index=False
    )

    if save_charts:
        _plot_wf_equity_curves(window_results, slug, show=show_charts)

    print(f"[Report] Walk-forward report saved to {REPORTS_DIR}/{slug}_*")
    return os.path.join(REPORTS_DIR, slug)


# ─────────────────────────────────────────────────────────────────────────────
# Internal chart / CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_trades_csv(result, slug: str) -> None:
    """Save trade-by-trade log to CSV."""
    from research.backtester import trades_to_dataframe
    df = trades_to_dataframe(result.trades)
    path = os.path.join(REPORTS_DIR, f"{slug}_trades.csv")
    df.to_csv(path, index=False)
    print(f"[Report] Trades  → {path}")


def _save_metrics_csv(result, slug: str) -> None:
    """Save scalar metrics to CSV (excludes equity_curve list)."""
    m = {k: v for k, v in result.metrics.items() if k != "equity_curve"}
    df = pd.DataFrame([m])
    path = os.path.join(REPORTS_DIR, f"{slug}_metrics.csv")
    df.to_csv(path, index=False)
    print(f"[Report] Metrics → {path}")


def _plot_equity_curve(result, slug: str, show: bool) -> None:
    """Save equity curve PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Report] matplotlib not installed — skipping charts.")
        return

    ec  = result.equity_curve
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ec, color="#2196F3", linewidth=1.2, label="Equity")
    ax.axhline(ec[0], color="grey", linestyle="--", alpha=0.5, label="Start")
    ax.fill_between(range(len(ec)), ec[0], ec, alpha=0.15, color="#2196F3")
    ax.set_title(f"Equity Curve — {result.label}")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Balance (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, f"{slug}_equity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if show:
        import subprocess
        subprocess.Popen(["start", path], shell=True)
    print(f"[Report] Equity  → {path}")


def _plot_drawdown(result, slug: str, show: bool) -> None:
    """Save drawdown curve PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ec  = np.array(result.equity_curve)
    peak = np.maximum.accumulate(ec)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak != 0, (ec - peak) / np.abs(peak), 0.0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(len(dd)), dd, 0, color="#F44336", alpha=0.5, label="Drawdown")
    ax.set_title(f"Drawdown — {result.label}")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.1%}")
    )
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, f"{slug}_drawdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Report] Drawdown→ {path}")


def _plot_trade_distribution(result, slug: str, show: bool) -> None:
    """Save trade PnL histogram PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    profits = [t.net_pnl for t in result.trades]
    if not profits:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    wins  = [p for p in profits if p > 0]
    losses= [p for p in profits if p <= 0]
    ax.hist(losses, bins=30, color="#F44336", alpha=0.7, label=f"Losses ({len(losses)})")
    ax.hist(wins,   bins=30, color="#4CAF50", alpha=0.7, label=f"Wins ({len(wins)})")
    ax.axvline(0, color="black", linestyle="--")
    ax.set_title(f"Trade PnL Distribution — {result.label}")
    ax.set_xlabel("Net PnL (USD)")
    ax.set_ylabel("Frequency")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, f"{slug}_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Report] PnL dist → {path}")


def _plot_wf_equity_curves(window_results: list, slug: str, show: bool) -> None:
    """Overlay equity curves from all WF windows."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = cm.tab10(np.linspace(0, 1, len(window_results)))

    for wr, col in zip(window_results, colors):
        ec = wr.result.equity_curve
        ax.plot(ec, color=col, alpha=0.75, label=f"W{wr.window_id}")

    ax.set_title("Walk-Forward — Equity Curves per Window")
    ax.set_xlabel("Bar"); ax.set_ylabel("Balance (USD)")
    ax.legend(fontsize=7, ncol=4); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, f"{slug}_wf_equity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Report] WF equity→ {path}")
