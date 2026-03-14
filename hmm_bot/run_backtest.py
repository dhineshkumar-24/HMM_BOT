"""
run_backtest.py — CLI entry point for the backtesting and research framework.

Usage (from inside hmm_bot/):

    # Full historical backtest:
    python run_backtest.py

    # Walk-forward validation:
    python run_backtest.py --mode wf

    # Grid search parameter optimisation:
    python run_backtest.py --mode grid

    # Custom bars:
    python run_backtest.py --bars 100000

    # No charts:
    python run_backtest.py --no-charts

    # Load from CSV instead of MT5:
    python run_backtest.py --csv path/to/data.csv

Validation thresholds (Step 11):
    Win Rate         >= 45%
    Profit Factor    >= 1.20
    Max Drawdown     <= 5%
    Minimum Trades   >= 30
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Core imports ──────────────────────────────────────────────────────────────
from config              import load_config
from research.data_loader     import load_mt5_history, load_csv_history
from research.backtester      import run_backtest
from research.walk_forward    import WalkForwardEngine
from research.report_generator import generate_report, generate_walk_forward_report
from research.performance_metrics import print_metrics
from strategy.strategy_router import StrategyRouter
from core.hmm_model           import HMMRegimeDetector
import MetaTrader5 as mt5

# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HMM Bot Backtesting & Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["backtest", "wf", "grid"],
        default="backtest",
        help="Run mode: backtest / walk-forward / grid-search",
    )
    parser.add_argument(
        "--bars", type=int, default=50_000,
        help="Number of historical bars to load (default: 50,000)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.7,
        help="Fraction of data used for training in backtest mode (default: 0.7)",
    )
    parser.add_argument(
        "--train-bars", type=int, default=30_000,
        help="Training window size for WF mode (default: 30,000)",
    )
    parser.add_argument(
        "--test-bars", type=int, default=10_000,
        help="Test window size for WF mode (default: 10,000)",
    )
    parser.add_argument(
        "--balance", type=float, default=10_000.0,
        help="Starting account balance in USD (default: 10,000)",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Override symbol from config (e.g. GBPUSD)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Load data from CSV instead of MT5 terminal",
    )
    parser.add_argument(
        "--train-hmm", action="store_true", default=True,
        help="Train HMM on training data before backtest (default: True)",
    )
    parser.add_argument(
        "--load-hmm", action="store_true", default=False,
        help="Load pre-trained HMM from models/hmm.pkl",
    )
    parser.add_argument(
        "--no-hmm", action="store_true", default=False,
        help="Disable HMM — run in warm-up/mean-reversion only mode",
    )
    parser.add_argument(
        "--no-charts", action="store_true", default=False,
        help="Skip chart generation",
    )
    parser.add_argument(
        "--spread", type=float, default=1.5,
        help="Spread in pips (default: 1.5)",
    )
    parser.add_argument(
        "--slippage", type=float, default=0.5,
        help="Max slippage in pips (default: 0.5)",
    )
    parser.add_argument(
        "--commission", type=float, default=6.0,
        help="Commission per lot round-trip in USD (default: 6.0)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    CONFIG = load_config()
    SYMBOL = args.symbol or CONFIG["trading"]["symbol"]
    TF_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1":  mt5.TIMEFRAME_H1,
    }
    TIMEFRAME = TF_MAP.get(CONFIG["trading"]["timeframe"], mt5.TIMEFRAME_M1)

    print("=" * 60)
    print("  HMM Bot — Quantitative Research Framework")
    print("=" * 60)
    print(f"  Mode         : {args.mode}")
    print(f"  Symbol       : {SYMBOL}")
    print(f"  Bars         : {args.bars:,}")
    print(f"  Balance      : ${args.balance:,.2f}")
    print(f"  HMM          : {'disabled' if args.no_hmm else ('load' if args.load_hmm else 'train')}")
    print(f"  Costs        : spread={args.spread}p | slip={args.slippage}p | comm=${args.commission}")
    print("=" * 60)

    # ── Load historical data ───────────────────────────────────────────────────
    if args.csv:
        df = load_csv_history(args.csv)
    else:
        print(f"\nLoading {args.bars:,} bars of {SYMBOL} from MT5...")
        if not mt5.initialize():
            print("ERROR: MT5 failed to initialize. Open MT5 terminal and retry.")
            sys.exit(1)
        df = load_mt5_history(SYMBOL, TIMEFRAME, bars=args.bars)
        mt5.shutdown()

    print(f"Data range: {df['time'].iloc[0]} → {df['time'].iloc[-1]} ({len(df):,} bars)")

    # ── HMM setup ───────────────────────────────────────────────────────────────
    hmm = None
    if not args.no_hmm:
        hmm = HMMRegimeDetector(
            n_states       = CONFIG.get("hmm", {}).get("n_components", 3),
            n_iter         = CONFIG.get("hmm", {}).get("n_iter", 100),
            n_seeds        = 3,
            confidence_thr = CONFIG.get("hmm", {}).get("confidence_threshold", 0.62),
            model_path     = CONFIG.get("hmm", {}).get("model_path", "models/hmm.pkl"),
        )
        if args.load_hmm:
            if hmm.load():
                print("HMM model loaded from disk.")
            else:
                print("No saved model found — will train on the training set.")
                args.train_hmm = True

    # ── Run requested mode ────────────────────────────────────────────────────
    if args.mode == "backtest":
        _run_single_backtest(df, CONFIG, hmm, args, SYMBOL)

    elif args.mode == "wf":
        _run_walk_forward(df, CONFIG, args)

    elif args.mode == "grid":
        _run_grid_search(df, CONFIG, args)


# ─────────────────────────────────────────────────────────────────────────────
# Mode handlers
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_backtest(df, CONFIG, hmm, args, symbol):
    """Standard chronological backtest with optional HMM training."""
    n     = len(df)
    split = int(n * args.train_frac)

    df_train = df.iloc[:split].reset_index(drop=True)
    df_test  = df.iloc[split:].reset_index(drop=True)

    print(f"\nTrain: {len(df_train):,} bars | Test: {len(df_test):,} bars")

    # Train HMM on training portion
    if hmm is not None and not hmm.is_trained and args.train_hmm:
        print("Training HMM on training data...")
        hmm.fit(df_train)
        hmm.save()
        print("HMM saved.")

    router = StrategyRouter(CONFIG)
    result = run_backtest(
        df              = df_test,
        config          = CONFIG,
        strategy        = router,
        hmm             = hmm,
        initial_balance = args.balance,
        spread_pips     = args.spread,
        slippage_pips   = args.slippage,
        commission      = args.commission,
        verbose         = True,
        label           = f"{symbol}_backtest",
    )

    generate_report(
        result,
        label       = f"{symbol}_backtest",
        save_charts = not args.no_charts,
    )

    # Validation gate
    if result.metrics.get("passed"):
        print("\n✅ STRATEGY PASSES validation thresholds.")
    else:
        print("\n❌ STRATEGY FAILS validation thresholds.")
        failed = [k for k, v in result.metrics.get("rules", {}).items() if not v]
        print(f"   Failed rules: {failed}")


def _run_walk_forward(df, CONFIG, args):
    """Walk-forward analysis."""
    engine = WalkForwardEngine(
        config      = CONFIG,
        train_bars  = args.train_bars,
        test_bars   = args.test_bars,
        retrain_hmm = not args.no_hmm,
        verbose     = True,
    )
    summary = engine.run(df, initial_balance=args.balance)
    generate_walk_forward_report(
        engine.window_results,
        summary,
        label       = "wf",
        save_charts = not args.no_charts,
    )


def _run_grid_search(df, CONFIG, args):
    """Grid search parameter optimisation."""
    engine = WalkForwardEngine(
        config     = CONFIG,
        train_bars = args.train_bars,
        test_bars  = args.test_bars,
    )

    # Default parameter grid (user can edit directly)
    param_grid = {
        "strategy.mean_reversion.z_score_trigger": [1.8, 2.0, 2.3, 2.5],
        "strategy.mean_reversion.adx_max":         [18, 20, 22],
        "strategy.momentum.adx_min":               [18, 20, 25],
        "strategy.momentum.rsi_trend_low":         [35, 40, 45],
    }

    gs_result = engine.grid_search(df, param_grid, metric="sharpe")
    print(f"\nBEST PARAMETERS: {gs_result['best_params']}")
    print(f"BEST SHARPE    : {gs_result['best_score']:.3f}")

    # Save grid search results
    os.makedirs(os.path.join(_HERE, "research", "reports"), exist_ok=True)
    import json
    out_path = os.path.join(_HERE, "research", "reports", "grid_search_results.json")
    # Serialise (remove non-serialisable nested metrics)
    for r in gs_result.get("all_results", []):
        r["metrics"] = {k: v for k, v in r["metrics"].items() if k != "equity_curve"}
    with open(out_path, "w") as fh:
        json.dump(gs_result, fh, indent=2, default=str)
    print(f"Results saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
