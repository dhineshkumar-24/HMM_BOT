"""
core/hmm_model.py — HMM Regime Detection Engine.

Implements a Gaussian Hidden Markov Model with 3 hidden states:
    0 = Mean Reverting  (low vol, negative autocorr)
    1 = Trending        (persistent momentum, high autocorr)
    2 = High Volatility (elevated ATR, high vol-of-vol)

Training:
    - Uses last `n_bars` candles (default 5000)
    - Trains 3 times with different random seeds; keeps the best model
      (scored by log-likelihood on the full training data)
    - Features are normalized with StandardScaler (fit on training data,
      reused for live prediction)
    - Model and scaler are persisted together to models/hmm.pkl via joblib

Live Prediction (every bar):
    - Fetch last 200 bars → compute 7 features → normalize → HMM decode
    - Returns (regime: int, confidence: float, probs: ndarray)
    - Built-in gates:
        * confidence < CONFIDENCE_THRESHOLD  → trade should be skipped
        * regime == REGIME_HIGH_VOL          → trade should be skipped

Dependencies:
    hmmlearn >= 0.3.0
    scikit-learn >= 1.3.0
    joblib >= 1.3.0
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Optional
from hmmlearn.hmm import GaussianHMM
from utils.logger   import setup_logger
from utils.features import build_feature_matrix, FEATURE_COLS

logger = setup_logger("HMMModel")

# ─────────────────────────────────────────────────────────────────────────────
# Regime constants
# ─────────────────────────────────────────────────────────────────────────────
REGIME_MEAN_REVERT: int = 0   # Low vol, mean-reverting behaviour
REGIME_TRENDING:    int = 1   # Directional momentum / trend
REGIME_HIGH_VOL:    int = 2   # High volatility — avoid trading

REGIME_NAMES: dict[int, str] = {
    REGIME_MEAN_REVERT: "mean_reverting",
    REGIME_TRENDING:    "trending",
    REGIME_HIGH_VOL:    "high_volatility",
}

CONFIDENCE_THRESHOLD: float = 0.70   # Minimum posterior probability to act
N_STATES:             int   = 3
N_TRAINING_BARS:      int   = 20000
N_PREDICTION_BARS:    int   = 200
N_TRAINING_SEEDS:     int   = 3


# ─────────────────────────────────────────────────────────────────────────────
# Helper — safe hmmlearn import
# ─────────────────────────────────────────────────────────────────────────────
def _require_hmmlearn():
    """Raise a clear error if hmmlearn is not installed."""
    try:
        from hmmlearn import hmm            # noqa: F401
        return hmm
    except ImportError:
        raise ImportError(
            "hmmlearn is required for regime detection.\n"
            "Install it with:  pip install hmmlearn scikit-learn joblib"
        )


def _require_sklearn():
    """Raise a clear error if scikit-learn is not installed."""
    try:
        from sklearn.preprocessing import StandardScaler   # noqa: F401
        return StandardScaler
    except ImportError:
        raise ImportError(
            "scikit-learn is required for feature normalization.\n"
            "Install it with:  pip install scikit-learn"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HMMRegimeDetector
# ─────────────────────────────────────────────────────────────────────────────
class HMMRegimeDetector:
    """
    Gaussian HMM regime detector.

    Usage — training:
        detector = HMMRegimeDetector()
        detector.fit(df)           # df must have OHLCV + tick_volume columns
        detector.save()

    Usage — live prediction (every bar):
        detector = HMMRegimeDetector()
        detector.load()
        regime, confidence, probs = detector.predict(df_last_200)
        if detector.should_trade(regime, confidence):
            # pass regime to StrategyRouter

    Usage — check before routing:
        if not detector.should_trade(regime, confidence):
            continue   # skip this bar
    """

    def __init__(
        self,
        n_states:      int   = N_STATES,
        n_iter:        int   = 250,
        n_seeds:       int   = N_TRAINING_SEEDS,
        model_path:    str   = "models/hmm.pkl",
        n_train_bars:  int   = N_TRAINING_BARS,
        n_pred_bars:   int   = N_PREDICTION_BARS,
        confidence_thr: float = CONFIDENCE_THRESHOLD,
    ):
        self.n_states        = n_states
        self.n_iter          = n_iter
        self.n_seeds         = n_seeds
        self.model_path      = model_path
        self.n_train_bars    = n_train_bars
        self.n_pred_bars     = n_pred_bars
        self.confidence_thr  = confidence_thr

        self._model: Optional[GaussianHMM] = None   # hmmlearn.GaussianHMM — best seed
        self._scaler         = None   # sklearn.StandardScaler
        self._is_trained     = False
        self._regime_history: list[int] = []
        self._label_map: dict[int, int]  = {0: 0, 1: 1, 2: 2}  # identity until trained

        self._warmup_logged = False
        self._last_gate_reason = None
        self._last_gate_regime = None
        self._low_conf_skip_count = 0
        self._high_vol_skip_count = 0

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, n_bars: int | None = None) -> None:
        """
        Train the HMM on the most recent `n_bars` candles of `df`.

        Trains `n_seeds` times with different random seeds and keeps the
        model with the highest log-likelihood score on the training data.

        Args:
            df:     Full OHLCV DataFrame (needs at least n_train_bars rows).
            n_bars: Override the number of training bars (default 5000).
        """
        hmm_lib = _require_hmmlearn()
        StandardScaler = _require_sklearn()

        n = n_bars or self.n_train_bars
        df_train = df.tail(n).copy()

        logger.info(f"Building feature matrix from {len(df_train)} bars...")
        features = build_feature_matrix(df_train)

        if len(features) < 200:
            logger.error(
                f"Only {len(features)} valid feature rows after NaN drop. "
                "Need at least 200. Cannot train."
            )
            return

        X = features.values   # (n_samples, 7)
        logger.info(f"Feature matrix shape: {X.shape}")

        # ── Normalize ─────────────────────────────────────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── Multi-seed training ───────────────────────────────────────────────
        best_model = None
        best_score = -np.inf
        seeds = [42, 7, 123][:self.n_seeds]

        for seed in seeds:
            logger.info(f"Training HMM — seed={seed}...")
            try:
                model = hmm_lib.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="diag",
                    n_iter=self.n_iter,
                    tol=1e-4,
                    random_state=seed,
                    verbose=False,
                )
                model.fit(X_scaled)
                score = model.score(X_scaled)
                logger.info(f"  Seed {seed} → log-likelihood: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                logger.warning(f"  Seed {seed} failed: {e}")

        if best_model is None:
            logger.error("All training seeds failed. HMM not trained.")
            return

        self._model      = best_model
        self._scaler     = scaler
        self._is_trained = True

        logger.info(
            f"HMM training complete. Best log-likelihood: {best_score:.4f} "
            f"| Converged: {best_model.monitor_.converged}"
        )
        # ── Align FIRST — then log with correct economic labels ────────────────
        # _log_model_summary() uses self.regime_name(i) which reads _label_map.
        # If alignment runs after summary, the summary prints wrong label names
        # because _label_map is still the identity map {0:0,1:1,2:2}.
        self._label_map = self._align_state_labels()
        self._log_model_summary()

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        df: pd.DataFrame,
        n_bars: int | None = None,
    ) -> tuple[int, float, np.ndarray]:
        """
        Predict the current market regime from recent candles.

        Uses the last `n_pred_bars` bars of `df` (or `n_bars` if provided).
        The Viterbi-decoded hidden state sequence is extracted; the
        LAST state in the sequence is the current bar's regime.
        Posterior probabilities come from the forward-backward algorithm.

        Args:
            df:     DataFrame with at least n_pred_bars rows.
            n_bars: Override n_pred_bars for this call.

        Returns:
            (regime, confidence, probabilities)
            - regime:        int — 0, 1, or 2
            - confidence:    float — P(regime | observations)
            - probabilities: ndarray of shape (n_states,)

        Raises:
            RuntimeError: If the model is not trained/loaded.
        """
        if not self._is_trained or self._model is None or self._scaler is None:
            raise RuntimeError(
                "HMM model is not trained. Call fit() or load() first."
            )

        n = n_bars or self.n_pred_bars
        df_slice = df.tail(n).copy()

        features = build_feature_matrix(df_slice)
        if len(features) < 70:
            if not self._warmup_logged:
                logger.info(
                    f"HMM warming up: {len(features)}/70 valid feature rows "
                    f"— defaulting to {self.regime_name(REGIME_MEAN_REVERT)}."
                )
                self._warmup_logged = True

            probs = np.zeros(self.n_states)
            probs[REGIME_MEAN_REVERT] = 1.0
            return REGIME_MEAN_REVERT, 1.0, probs

        self._warmup_logged = False

        X = features.values
        X_scaled = self._scaler.transform(X)

        # Viterbi decode → regime labels for every bar in the window
        try:
            _, state_sequence = self._model.decode(X_scaled, algorithm="viterbi")
        except Exception as e:
            logger.error(f"HMM decode failed: {e}")
            probs = np.ones(self.n_states) / self.n_states
            return REGIME_MEAN_REVERT, 1.0 / self.n_states, probs

        # Forward-backward posteriors for the full sequence
        try:
            _, posteriors = self._model.score_samples(X_scaled)
        except Exception as e:
            logger.warning(f"Posterior calculation failed: {e}. Using Viterbi only.")
            posteriors = np.zeros((len(X_scaled), self.n_states))
            posteriors[np.arange(len(X_scaled)), state_sequence] = 1.0

        # Current bar = last observation
        raw_state     = int(state_sequence[-1])
        current_probs = posteriors[-1]                          # shape (n_states,)
        confidence    = float(current_probs[raw_state])

        # Translate learned state integer → canonical economic regime label
        current_state = self._label_map.get(raw_state, raw_state)

        return current_state, confidence, current_probs

    # ── Trade gate ────────────────────────────────────────────────────────────

    def should_trade(
        self,
        regime:     int,
        confidence: float,
        posteriors: np.ndarray = None,
    ) -> bool:
        """
        Return False if the bot should skip trading this bar.

        Conditions for SKIP (applied in order):
            1. confidence < confidence_threshold
            2. posterior entropy > 0.65 (model genuinely uncertain)
            3. regime == REGIME_HIGH_VOL

        The entropy gate (condition 2) is the new addition.
        It catches bars where the model has a plurality winner
        but is still near-uniform across states — the symptom
        of weak regime separation (gap = 0.070).

        Entropy interpretation for 3 states:
            Max entropy = log(3) = 1.099  (perfectly uniform)
            Normalised entropy = entropy / log(n_states)
            0.0 = certain (one state = 1.0)
            1.0 = uniform (all states = 0.333)
            We skip when normalised entropy > 0.65
        """
        # Case 1: low confidence
        if confidence < self.confidence_thr:
            self._low_conf_skip_count += 1
            if self._last_gate_reason != "low_confidence":
                logger.info(
                    f"[Gate] Low confidence {confidence:.2%} < "
                    f"{self.confidence_thr:.2%} — skipping."
                )
            elif self._low_conf_skip_count % 100 == 0:
                logger.info(
                    f"[Gate] Low-confidence skips: {self._low_conf_skip_count}"
                )
            self._last_gate_reason = "low_confidence"
            self._last_gate_regime = regime
            return False

        # Case 2: posterior entropy gate
        if posteriors is not None:
            p            = np.array(posteriors) + 1e-9   # avoid log(0)
            entropy      = float(-np.sum(p * np.log(p)))
            max_entropy  = np.log(self.n_states)
            norm_entropy = entropy / max_entropy

            if norm_entropy > 0.65:
                if self._last_gate_reason != "high_entropy":
                    logger.info(
                        f"[Gate] Posterior entropy {norm_entropy:.2f} > 0.65 "
                        f"— model uncertain across states, skipping. "
                        f"Posteriors: {[f'{p:.2f}' for p in posteriors]}"
                    )
                self._last_gate_reason = "high_entropy"
                self._last_gate_regime = regime
                return False

        # Case 3: high-volatility regime
        if regime == REGIME_HIGH_VOL:
            self._high_vol_skip_count += 1
            if self._last_gate_reason != "high_volatility":
                logger.info("[Gate] High-volatility regime detected — skipping.")
            elif self._high_vol_skip_count % 100 == 0:
                logger.info(
                    f"[Gate] High-volatility skips: {self._high_vol_skip_count}"
                )
            self._last_gate_reason = "high_volatility"
            self._last_gate_regime = regime
            return False

        # All gates passed
        self._last_gate_reason = "pass"
        self._last_gate_regime = regime
        return True

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        """
        Persist the trained model + scaler to disk as a single .pkl file.

        Args:
            path: Override the default model_path.
        """
        if not self._is_trained:
            logger.error("Cannot save — model not trained.")
            return

        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required. Install with: pip install joblib")

        target = path or self.model_path
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)

        payload = {
            "model":      self._model,
            "scaler":     self._scaler,
            "n_states":   self.n_states,
            "features":   FEATURE_COLS,
            "label_map":  self._label_map,   # ← persist alignment
        }
        joblib.dump(payload, target)
        logger.info(f"HMM model saved to: {target}")

    def load(self, path: str | None = None) -> bool:
        """
        Load a previously saved model + scaler from disk.

        Args:
            path: Override the default model_path.

        Returns:
            True if successfully loaded, False if file not found.
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required. Install with: pip install joblib")

        target = path or self.model_path
        if not os.path.exists(target):
            logger.info(f"No saved model found at: {target}")
            return False

        try:
            payload = joblib.load(target)
            self._model      = payload["model"]
            self._scaler     = payload["scaler"]
            self.n_states    = payload.get("n_states", self.n_states)
            self._label_map  = payload.get("label_map", {0:0, 1:1, 2:2})  # ← restore
            self._is_trained = True
            logger.info(f"HMM model loaded from: {target}")
            return True
        except Exception as e:
            logger.error(f"Failed to load HMM model: {e}")
            return False

    # ── Regime history helpers ─────────────────────────────────────────────────

    def update_history(self, regime: int) -> None:
        """Append to the rolling regime history (used for stability checks)."""
        self._regime_history.append(regime)
        if len(self._regime_history) > 50:
            self._regime_history.pop(0)

    def is_regime_stable(self, window: int = 5) -> bool:
        """
        Return True if the last `window` predicted regimes are all the same.
        Prevents acting during rapid regime transitions.
        """
        if len(self._regime_history) < window:
            return False
        recent = self._regime_history[-window:]
        return len(set(recent)) == 1

    def regime_name(self, regime: int) -> str:
        """Human-readable regime label."""
        return REGIME_NAMES.get(regime, f"unknown_{regime}")

    def clear_history(self) -> None:
        """Reset rolling regime history (call on daily reset)."""
        self._regime_history.clear()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def regime_history(self) -> list[int]:
        return list(self._regime_history)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _log_model_summary(self) -> None:
        """Log the trained model's transition matrix and mean vectors.

        Called AFTER _align_state_labels() so regime_name() shows the
        correct economic label (mean_reverting / trending / high_volatility)
        rather than the raw hmmlearn integer state index.
        """
        if self._model is None:
            return
        logger.info("── HMM Model Summary ─────────────────")
        tm = self._model.transmat_
        for i in range(self.n_states):
            row = "  ".join(f"{p:.3f}" for p in tm[i])
            # _label_map[i] converts raw hmmlearn state i to canonical regime int
            # regime_name() then converts canonical int to human label
            canonical = self._label_map.get(i, i)
            logger.info(
                f"  State {i} ({self.regime_name(canonical):18s}) "
                f"transitions: [{row}]"
            )
        logger.info("── Feature Means per State ───────────")
        for i, mean_vec in enumerate(self._model.means_):
            vals = "  ".join(f"{v:+.4f}" for v in mean_vec)
            canonical = self._label_map.get(i, i)
            logger.info(
                f"  State {i} ({self.regime_name(canonical):12s}) "
                f"means: [{vals}]"
            )
        logger.info("──────────────────────────────────────")

    def _align_state_labels(self) -> dict:
        """
        Map learned HMM states to economic regime labels.
        Step 1 — identify HIGH_VOL
        Step 2 — define remaining immediately after Step 1
        Step 3 — attempt enhanced feature lookup
        Step 4 — compute trending_scores
        Step 5 — assign labels and log
        """
        if self._model is None:
            return {0: 0, 1: 1, 2: 2}

        means = self._model.means_

        # ── Step 1: Core vol features ─────────────────────────────────────────
        try:
            IDX_RVOL = FEATURE_COLS.index("realized_vol")
            IDX_VOV  = FEATURE_COLS.index("vol_of_vol")
            IDX_ATR  = FEATURE_COLS.index("atr_norm")
        except ValueError as exc:
            logger.error(
                f"Core volatility feature missing: {exc}. "
                f"Returning identity map."
            )
            return {0: 0, 1: 1, 2: 2}

        # ── Step 2: HIGH_VOL + remaining ──────────────────────────────────────
        combined_vol   = means[:, IDX_RVOL] + means[:, IDX_ATR] + means[:, IDX_VOV]
        high_vol_state = int(np.argmax(combined_vol))
        remaining      = [i for i in range(self.n_states) if i != high_vol_state]

        # ── Step 3: Feature availability check ───────────────────────────────
        enhanced_available = False
        legacy_available   = False

        try:
            IDX_ER    = FEATURE_COLS.index("efficiency_ratio")
            IDX_VR    = FEATURE_COLS.index("variance_ratio")
            IDX_HURST = FEATURE_COLS.index("hurst_rolling")
            IDX_MLAC  = FEATURE_COLS.index("multi_lag_autocorr")
            enhanced_available = True
        except ValueError:
            pass

        if not enhanced_available:
            try:
                IDX_AUTOCR = FEATURE_COLS.index("autocorr")
                IDX_MOM    = FEATURE_COLS.index("momentum")
                legacy_available = True
            except ValueError:
                pass

        # ── Step 4: Trending scores ───────────────────────────────────────────
        if enhanced_available:
            trending_scores = {
                state: (
                    means[state, IDX_ER]      * 0.40
                    + means[state, IDX_VR]    * 0.35
                    + means[state, IDX_HURST] * 0.15
                    + means[state, IDX_MLAC]  * 0.10
                )
                for state in remaining
            }
            method = "enhanced (ER+VR+Hurst+MLAC)"

        elif legacy_available:
            trending_scores = {
                state: (
                    means[state, IDX_AUTOCR] * 0.7
                    + means[state, IDX_MOM]  * 0.3
                )
                for state in remaining
            }
            method = "legacy (autocorr+momentum)"

        else:
            try:
                IDX_DD = FEATURE_COLS.index("drawdown_pct")
                trending_scores = {
                    state: -means[state, IDX_DD]
                    for state in remaining
                }
                method = "last-resort (drawdown proxy)"
                logger.warning(
                    "Neither enhanced nor legacy trend features found. "
                    f"FEATURE_COLS = {FEATURE_COLS}."
                )
            except ValueError:
                logger.error(
                    f"No usable trend features. FEATURE_COLS = {FEATURE_COLS}"
                )
                return {0: 0, 1: 1, 2: 2}

        # ── Step 5: Assign labels ─────────────────────────────────────────────
        trending_state = max(trending_scores, key=trending_scores.get)
        mean_rev_state = [s for s in remaining if s != trending_state][0]

        label_map = {
            mean_rev_state: REGIME_MEAN_REVERT,
            trending_state: REGIME_TRENDING,
            high_vol_state: REGIME_HIGH_VOL,
        }

        score_gap = abs(
            trending_scores[trending_state] - trending_scores[mean_rev_state]
        )

        if score_gap < 0.10:
            logger.warning(
                f"Alignment confidence LOW (gap={score_gap:.4f}). "
                f"Method: {method}."
            )

        logger.info("── State Label Alignment ──────────────────────────")
        logger.info(
            f"  Learned state {high_vol_state} → HIGH_VOL    "
            f"| vol_score={combined_vol[high_vol_state]:+.4f}"
        )
        logger.info(
            f"  Learned state {trending_state} → TRENDING    "
            f"| trend_score={trending_scores[trending_state]:+.4f}"
        )
        logger.info(
            f"  Learned state {mean_rev_state} → MEAN_REVERT "
            f"| trend_score={trending_scores[mean_rev_state]:+.4f}"
        )
        logger.info(
            f"  Gap={score_gap:.4f} "
            f"({'GOOD' if score_gap >= 0.10 else 'WEAK'}) "
            f"| Method={method}"
        )
        logger.info("───────────────────────────────────────────────────")

        return label_map