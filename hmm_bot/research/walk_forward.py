"""
research/walk_forward.py — Rolling walk-forward validation engine.

Splits the full historical dataset into sequential train/test windows,
runs a backtest on each out-of-sample window, and aggregates results.

Default window sizes:
    train : 6 months of M1 bars ≈ 6 × 21 × 1440 = 181,440 bars
    test  : 1 month  of M1 bars ≈ 1 × 21 × 1440 = 30,240 bars

These are configurable. For daily bars use train=126, test=21.

Optional: Grid search on parameter combinations using the test window
Sharpe ratio as the objective, returning the best parameter set.

Example usage:
    from research.walk_forward import WalkForwardEngine
    engine = WalkForwardEngine(config, train_bars=10_000, test_bars=3_000)
    summary = engine.run(df)
"""

from __future__ import annotations

import sys
import os
import copy
import itertools
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from research.backtester        import run_backtest, BacktestResult
from research.performance_metrics import compute_metrics, print_metrics
from strategy.strategy_router   import StrategyRouter
from core.hmm_model             import HMMRegimeDetector


# ─────────────────────────────────────────────────────────────────────────────
# Window summary
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    window_id:    int
    train_start:  int
    train_end:    int
    test_start:   int
    test_end:     int
    result:       BacktestResult
    params:       dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward engine
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardEngine:
    """
    Anchored or rolling walk-forward validation.

    Args:
        config:       Full settings dict.
        train_bars:   Number of bars in each training window.
        test_bars:    Number of bars in each out-of-sample test window.
        anchored:     If True, training window always starts from bar 0
                      (expanding window). If False, rolling window.
        retrain_hmm:  Retrain HMM on each training window.
        hmm_path:     Optional path to a pre-trained HMM model to load.
        verbose:      Print progress information.
    """

    M1_TRAIN_DEFAULT  = 181_440   # ~6 months of M1
    M1_TEST_DEFAULT   =  30_240   # ~1 month of M1

    def __init__(
        self,
        config:      dict,
        train_bars:  int  = M1_TRAIN_DEFAULT,
        test_bars:   int  = M1_TEST_DEFAULT,
        anchored:    bool = False,
        retrain_hmm: bool = True,
        verbose:     bool = True,
    ):
        self.config      = config
        self.train_bars  = train_bars
        self.test_bars   = test_bars
        self.anchored    = anchored
        self.retrain_hmm = retrain_hmm
        self.verbose     = verbose

        self.window_results: list[WindowResult] = []

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(
        self,
        df:              pd.DataFrame,
        initial_balance: float = 10_000.0,
    ) -> dict:
        """
        Execute the walk-forward analysis.

        Returns:
            Aggregated summary dict with averaged metrics across all windows.
        """
        self.window_results.clear()
        n = len(df)

        if n < self.train_bars + self.test_bars:
            raise ValueError(
                f"DataFrame has only {n} bars but needs at least "
                f"{self.train_bars + self.test_bars} for one WF window."
            )

        # ── Generate window boundaries ────────────────────────────────────────
        windows = []
        train_start = 0
        test_start  = self.train_bars

        while test_start + self.test_bars <= n:
            windows.append((train_start, test_start - 1, test_start, test_start + self.test_bars - 1))
            if not self.anchored:
                train_start = test_start
            test_start += self.test_bars

        print(f"[WalkForward] {len(windows)} window(s) | "
              f"train={self.train_bars} bars | test={self.test_bars} bars")

        for w_id, (tr_s, tr_e, ts_s, ts_e) in enumerate(windows, 1):
            print(f"\n[WalkForward] Window {w_id}/{len(windows)} | "
                  f"Train [{tr_s}:{tr_e}] Test [{ts_s}:{ts_e}]")

            df_train = df.iloc[tr_s : tr_e + 1].reset_index(drop=True)
            df_test  = df.iloc[ts_s : ts_e + 1].reset_index(drop=True)

            # ── Optional HMM training on train window ─────────────────────────
            hmm = None
            if self.retrain_hmm:
                hmm = HMMRegimeDetector(
                    n_states       = self.config.get("hmm", {}).get("n_components", 3),
                    n_iter         = self.config.get("hmm", {}).get("n_iter", 250),
                    n_seeds        = 3,
                    confidence_thr = self.config.get("hmm", {}).get("confidence_threshold", 0.70),
                )
                print(f"  → Training HMM on {len(df_train)} bars...")
                hmm.fit(df_train)

            # ── Run out-of-sample backtest ────────────────────────────────────
            router = StrategyRouter(self.config)
            result = run_backtest(
                df              = df_test,
                config          = self.config,
                strategy        = router,
                hmm             = hmm,
                initial_balance = initial_balance,
                verbose         = False,
                label           = f"Window {w_id}",
            )

            if self.verbose:
                m = result.metrics
                print(f"  → Trades:{m['total_trades']} | "
                      f"WinRate:{m['win_rate']:.1%} | "
                      f"PF:{m['profit_factor']:.2f} | "
                      f"Sharpe:{m['sharpe']:.2f} | "
                      f"DD:{m['max_drawdown']:.2%}")

            wr = WindowResult(
                window_id   = w_id,
                train_start = tr_s,
                train_end   = tr_e,
                test_start  = ts_s,
                test_end    = ts_e,
                result      = result,
            )
            self.window_results.append(wr)

        return self._aggregate()

    # ── Grid search ───────────────────────────────────────────────────────────

    def grid_search(
        self,
        df:         pd.DataFrame,
        param_grid: dict[str, list[Any]],
        metric:     str = "sharpe",
    ) -> dict:
        """
        Grid search over parameter combinations evaluated on a single
        train/test split (first window of the dataset).

        Tested parameters should live in config['strategy'] sub-sections.

        Example param_grid:
            {
                "strategy.mean_reversion.z_score_trigger": [1.8, 2.0, 2.3],
                "strategy.mean_reversion.adx_max":         [18, 20, 22],
                "strategy.momentum.adx_min":               [18, 20, 25],
            }

        Args:
            df:         Full historical DataFrame.
            param_grid: Dict mapping dot-path config keys → list of values.
            metric:     Objective metric key from metrics dict (default: "sharpe").

        Returns:
            Dict with 'best_params' and 'best_score', plus full results table.
        """
        n = len(df)
        split = min(self.train_bars + self.test_bars, n)
        df_test = df.iloc[self.train_bars : split].reset_index(drop=True)

        keys   = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        print(f"[GridSearch] {len(combos)} combinations | "
              f"objective={metric} | test_bars={len(df_test)}")

        best_score  = -np.inf
        best_params: dict = {}
        all_results: list[dict] = []

        for idx, combo in enumerate(combos, 1):
            params    = dict(zip(keys, combo))
            cfg_copy  = _apply_params(copy.deepcopy(self.config), params)
            router    = StrategyRouter(cfg_copy)

            result = run_backtest(
                df       = df_test,
                config   = cfg_copy,
                strategy = router,
                verbose  = False,
                label    = f"GS-{idx}",
            )
            score = result.metrics.get(metric, -np.inf)
            all_results.append({"params": params, "score": score, "metrics": result.metrics})

            if score > best_score:
                best_score  = score
                best_params = params

            if idx % 10 == 0:
                print(f"  {idx}/{len(combos)} | best {metric}: {best_score:.3f}")

        print(f"[GridSearch] Best {metric}: {best_score:.3f} | Params: {best_params}")
        return {
            "best_params": best_params,
            "best_score":  best_score,
            "metric":      metric,
            "all_results": all_results,
        }

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self) -> dict:
        """Compute mean metrics across all walk-forward windows."""
        if not self.window_results:
            return {}

        all_metrics = [wr.result.metrics for wr in self.window_results]
        scalar_keys = [
            "total_trades", "win_rate", "profit_factor", "max_drawdown",
            "sharpe", "sortino", "net_profit", "expectancy",
        ]

        agg = {}
        for k in scalar_keys:
            vals = [m[k] for m in all_metrics if k in m]
            if vals:
                agg[f"avg_{k}"]    = float(np.mean(vals))
                agg[f"std_{k}"]    = float(np.std(vals))
                agg[f"min_{k}"]    = float(np.min(vals))
                agg[f"max_{k}"]    = float(np.max(vals))

        agg["n_windows"] = len(self.window_results)

        print("\n" + "=" * 52)
        print("  WALK-FORWARD AGGREGATE SUMMARY")
        print("=" * 52)
        print(f"  Windows       : {agg['n_windows']}")
        print(f"  Avg Trades    : {agg.get('avg_total_trades', 0):.1f}")
        print(f"  Avg Win Rate  : {agg.get('avg_win_rate', 0):.1%}")
        print(f"  Avg PF        : {agg.get('avg_profit_factor', 0):.2f}")
        print(f"  Avg Max DD    : {agg.get('avg_max_drawdown', 0):.2%}")
        print(f"  Avg Sharpe    : {agg.get('avg_sharpe', 0):.3f}")
        print(f"  Avg PnL/windo : {agg.get('avg_net_profit', 0):+.2f}")
        print("=" * 52)

        return agg


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_params(config: dict, params: dict) -> dict:
    """
    Apply dot-path parameter overrides to a nested config dict.

    Example: 'strategy.mean_reversion.z_score_trigger' → 2.3
    """
    for dotpath, value in params.items():
        keys = dotpath.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config
