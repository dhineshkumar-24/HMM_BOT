"""
models/__init__.py

This package stores serialised model artefacts (.pkl files).
Python source code does NOT belong here.

The HMM implementation lives in core/hmm_model.py.
Always import from there:

    from core.hmm_model import HMMRegimeDetector
    from core.hmm_model import REGIME_MEAN_REVERT, REGIME_TRENDING, REGIME_HIGH_VOL

models/hmm_model.py previously existed as a placeholder stub and defined
REGIME_NOISY = 2 (conflicting with REGIME_HIGH_VOL = 2 in core/hmm_model.py).
It has been deleted. If you see an ImportError referencing models.hmm_model,
update that import to core.hmm_model.
"""