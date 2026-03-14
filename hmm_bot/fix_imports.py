"""
fix_imports.py — Scans ALL python files in hmm_bot/ and removes
the 'hmm_bot.' prefix from any imports.
Run once from inside hmm_bot/:  python fix_imports.py
"""
import os, re

ROOT = os.path.dirname(os.path.abspath(__file__))
fixed = []

print("\n🔍 Scanning all .py files for bad 'hmm_bot.' imports...\n")

for dirpath, dirnames, filenames in os.walk(ROOT):
    # Skip venv and __pycache__
    dirnames[:] = [d for d in dirnames if d not in ('venv', '__pycache__', '.git')]
    for filename in filenames:
        if not filename.endswith('.py'):
            continue
        filepath = os.path.join(dirpath, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()

        # Fix both styles of bad import
        patched = re.sub(r'from hmm_bot\.', 'from ', original)
        patched = re.sub(r'import hmm_bot\.', 'import ', patched)

        if patched != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(patched)
            rel = os.path.relpath(filepath, ROOT)
            print(f"  ✅ Fixed: {rel}")
            fixed.append(rel)

print(f"\n{'='*50}")
print(f"  Done. Fixed {len(fixed)} file(s).")
if not fixed:
    print("  No bad imports found — problem is something else.")
    print("\n  Run this to see the exact error line:")
    print('  python -c "from research.alpha.mean_reversion_alpha import volatility_adjusted_zscore"')
print(f"{'='*50}\n")