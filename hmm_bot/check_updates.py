"""
check_updates.py  —  Run this FIRST before any backtest.
Verifies that all Phase 1-9 updates are actually in the codebase.

FIX L4: All path checks now use _HERE (script directory) as base
         instead of CWD. Run from any directory safely:
             python hmm_bot/check_updates.py

Run from inside hmm_bot/:
    python check_updates.py
"""

import os, sys, importlib

# FIX L4: Use script-relative base path — works regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))

PASS = "  ✅"
FAIL = "  ❌"
WARN = "  ⚠️ "

results = []

def check(label, condition, fix):
    status = PASS if condition else FAIL
    results.append((status, label, fix if not condition else ""))
    print(f"{status}  {label}")
    if not condition:
        print(f"       FIX → {fix}\n")

print("\n" + "="*60)
print("  HMM BOT — Update Verification Checklist")
print("="*60 + "\n")

# FIX L1+L4: Use _HERE base and updated REGIME_HIGH_VOL name
print("📁  Folder Structure")
check("research/alpha/ folder exists",
      os.path.isdir(os.path.join(_HERE, "research", "alpha")),
      "mkdir research\\alpha")

check("research/validation/ folder exists",
      os.path.isdir(os.path.join(_HERE, "research", "validation")),
      "mkdir research\\validation")

check("portfolio/ folder exists",
      os.path.isdir(os.path.join(_HERE, "portfolio")),
      "mkdir portfolio")

check("risk_controls/ folder exists",
      os.path.isdir(os.path.join(_HERE, "risk_controls")),
      "mkdir risk_controls")

check("analytics/ folder exists",
      os.path.isdir(os.path.join(_HERE, "analytics")),
      "mkdir analytics")

# ── 2. ALPHA FILES ────────────────────────────────────────────────
print("\n📊  Alpha Signal Files")
check("research/alpha/mean_reversion_alpha.py exists",
      os.path.isfile(os.path.join(_HERE, "research", "alpha", "mean_reversion_alpha.py")),
      "This file was supposed to be created in Phase 3 — recreate it")

check("research/alpha/momentum_alpha.py exists",
      os.path.isfile(os.path.join(_HERE, "research", "alpha", "momentum_alpha.py")),
      "Phase 3 file missing — recreate it")

check("research/alpha/microstructure_alpha.py exists",
      os.path.isfile(os.path.join(_HERE, "research", "alpha", "microstructure_alpha.py")),
      "Phase 3 file missing — recreate it")

check("research/alpha/regime_alpha.py exists",
      os.path.isfile(os.path.join(_HERE, "research", "alpha", "regime_alpha.py")),
      "Phase 3 file missing — recreate it")

# ── 3. VALIDATION + PORTFOLIO ─────────────────────────────────────
print("\n🔬  Validation & Portfolio")
check("research/validation/signal_validator.py exists",
      os.path.isfile(os.path.join(_HERE, "research", "validation", "signal_validator.py")),
      "Phase 5 file missing — recreate it")

check("portfolio/signal_combiner.py exists",
      os.path.isfile(os.path.join(_HERE, "portfolio", "signal_combiner.py")),
      "Phase 7 file missing — recreate it")

check("research/experiment_tracker.py exists",
      os.path.isfile(os.path.join(_HERE, "research", "experiment_tracker.py")),
      "Phase 9 file missing — recreate it")

# ── 4. RISK CONTROLS ─────────────────────────────────────────────
print("\n🛡️   Risk Controls (CRITICAL)")
check("risk_controls/drawdown_monitor.py exists",
      os.path.isfile(os.path.join(_HERE, "risk_controls", "drawdown_monitor.py")),
      "CRITICAL: main.py imports this — bot will crash without it")

check("risk_controls/loss_streak_monitor.py exists",
      os.path.isfile(os.path.join(_HERE, "risk_controls", "loss_streak_monitor.py")),
      "CRITICAL: main.py imports this — bot will crash without it")

# ── 5. FEATURE COLS CHECK ─────────────────────────────────────────
print("\n🧠  Feature Engineering (CRITICAL)")
try:
    from utils.features import FEATURE_COLS, build_feature_matrix
    check("FEATURE_COLS has more than 7 features (new ones added)",
          len(FEATURE_COLS) > 7,
          f"CRITICAL: FEATURE_COLS still has only {len(FEATURE_COLS)} features. "
          f"Add skewness, kurtosis, drawdown_pct, hurst_rolling to FEATURE_COLS "
          f"AND to build_feature_matrix() in utils/features.py")
    print(f"       Current features: {FEATURE_COLS}")
except Exception as e:
    print(f"{FAIL}  Could not import utils/features.py — {e}")

# ── 6. IMPORTS CHECK ──────────────────────────────────────────────
print("\n🔗  Import Chain (checks if files are importable)")

modules_to_check = [
    ("config", "load_config"),
    ("utils.features", "build_feature_matrix"),
    ("core.hmm_model", "HMMRegimeDetector"),
    ("strategy.strategy_router", "StrategyRouter"),
    ("research.backtester", "run_backtest"),
    ("research.walk_forward", "WalkForwardEngine"),
]

for mod_name, attr in modules_to_check:
    try:
        mod = importlib.import_module(mod_name)
        has_attr = hasattr(mod, attr)
        check(f"{mod_name}.{attr} importable",
              has_attr,
              f"File exists but {attr} is missing from {mod_name}")
    except ImportError as e:
        check(f"{mod_name} importable",
              False,
              f"ImportError: {e}")

# ── 7. RISK CONTROLS IMPORTABLE ───────────────────────────────────
print("\n🛡️   Risk Controls Importable")
for mod_name, cls in [
    ("risk_controls.drawdown_monitor", "DrawdownMonitor"),
    ("risk_controls.loss_streak_monitor", "LossStreakMonitor"),
]:
    try:
        mod = importlib.import_module(mod_name)
        check(f"{cls} importable",
              hasattr(mod, cls),
              f"{cls} class is missing from {mod_name}")
    except ImportError as e:
        check(f"{mod_name} importable", False, f"ImportError: {e}")

# ── 8. SIGNAL COMBINER WIRING ───────────────────────────────────────
print("\n🔌  Router → Combiner Wiring")
try:
    with open(os.path.join(_HERE, "strategy", "strategy_router.py"), "r") as f:
        router_code = f.read()
    check("strategy_router.py imports signal_combiner",
          "signal_combiner" in router_code or "SignalCombiner" in router_code,
          "CRITICAL: Router is not wired to combiner. "
          "Alpha modules are dead code until this is connected.")
    check("strategy_router.py imports alpha modules",
          "mean_reversion_alpha" in router_code or "research.alpha" in router_code,
          "Router does not call any alpha module — alpha signals have no effect")
except FileNotFoundError:
    print(f"{FAIL}  strategy/strategy_router.py not found")

# FIX L1: Verify REGIME_HIGH_VOL is used (not old REGIME_NOISY)
print("\n🏷️   Regime Naming Consistency")
try:
    with open(os.path.join(_HERE, "core", "hmm_model.py"), "r") as f:
        hmm_code = f.read()
    with open(os.path.join(_HERE, "models", "hmm_model.py"), "r") as f:
        stub_code = f.read()
    check("core/hmm_model.py uses REGIME_HIGH_VOL (not REGIME_NOISY)",
          "REGIME_HIGH_VOL" in hmm_code and "REGIME_NOISY" not in hmm_code,
          "Replace REGIME_NOISY with REGIME_HIGH_VOL in core/hmm_model.py")
    check("models/hmm_model.py stub uses REGIME_HIGH_VOL (not REGIME_NOISY)",
          "REGIME_HIGH_VOL" in stub_code,
          "FIX L1: Rename REGIME_NOISY=2 to REGIME_HIGH_VOL=2 in models/hmm_model.py")
except FileNotFoundError as e:
    print(f"{WARN}  Could not check regime naming: {e}")

# ── SUMMARY ───────────────────────────────────────────────────────
fails = [r for r in results if r[0] == FAIL]
passes = [r for r in results if r[0] == PASS]

print("\n" + "="*60)
print(f"  RESULT:  {len(passes)} passed   {len(fails)} failed")
print("="*60)

if fails:
    print("\n🔴  FIX THESE BEFORE RUNNING ANY BACKTEST:\n")
    for _, label, fix in fails:
        print(f"  • {label}")
        print(f"    → {fix}\n")
else:
    print("\n🟢  All checks passed. Safe to proceed to Run 1.")