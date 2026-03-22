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

CONFIDENCE_THRESHOLD: float = 0.65   # Minimum posterior probability to act
N_STATES:             int   = 3
N_TRAINING_BARS:      int   = 40000   # Use more data for better regime sep.
N_PREDICTION_BARS:    int   = 200
N_TRAINING_SEEDS:     int   = 5      # More seeds = better likelihood surface coverage


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
        n_iter:        int   = 300,
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

        self._model: Optional[GaussianHMM] = None
        self._scaler         = None
        self._is_trained     = False
        self._regime_history: list[int] = []
        self._label_map: dict[int, int]  = {0: 0, 1: 1, 2: 2}
        self._posterior_history: list[np.ndarray] = []   # for EMA smoothing

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
                    covariance_type="full",
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
        self._log_model_summary()
        # Reset smoothing buffer then align with composite scoring
        self._posterior_history = []
        self._label_map = self._align_state_labels(X_scaled)

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
        raw_state = int(state_sequence[-1])

        # ── Posterior smoothing (regime persistence filter) ────────────────────────
        # Apply EMA-5 smoothing to posteriors BEFORE argmax.
        # This adds effective regime persistence: prevents flickering between
        # regimes on consecutive bars (e.g. regime stays stable for 15-30 bars
        # even if raw posteriors bounce). No look-ahead bias: we smooth only the
        # historical posterior sequence, then take the LAST smoothed value.
        last_posterior  = posteriors[-1]    # shape (n_states,)
        smooth_posterior = self._smooth_posterior(last_posterior)

        # Use the smoothed argmax as the regime label
        raw_state_smooth  = int(np.argmax(smooth_posterior))
        confidence        = float(smooth_posterior[raw_state_smooth])

        # Translate learned state integer → canonical economic regime label
        current_state = self._label_map.get(raw_state_smooth, raw_state_smooth)

        return current_state, confidence, smooth_posterior

    # ── Trade gate ────────────────────────────────────────────────────────────

    def should_trade(self, regime: int, confidence: float) -> bool:
        """
        Return False if the bot should skip trading this bar.

        Conditions for SKIP:
            1. confidence < confidence_threshold
            2. regime == REGIME_HIGH_VOL (2)

        Args:
            regime:     Predicted regime label (0, 1, or 2).
            confidence: Posterior probability of the predicted regime.

        Returns:
            bool — True = safe to trade, False = skip this bar.
        """
        # Case 1: low confidence
        if confidence < self.confidence_thr:
            self._low_conf_skip_count += 1

            # log only on state change, then every 100 skips
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

        # Case 2: high-volatility regime
        if regime == REGIME_HIGH_VOL:
            self._high_vol_skip_count += 1

            # log only on transition into high-vol regime, then every 100 skips
            if self._last_gate_reason != "high_volatility":
                logger.info("[Gate] High-volatility regime detected — skipping.")
            elif self._high_vol_skip_count % 100 == 0:
                logger.info(
                    f"[Gate] High-volatility skips: {self._high_vol_skip_count}"
                )

            self._last_gate_reason = "high_volatility"
            self._last_gate_regime = regime
            return False

        # Gate passed → reset gate state
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
            "label_map":  self._label_map,
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
        """Log the trained model's transition matrix and mean vectors."""
        if self._model is None:
            return
        logger.info("── HMM Model Summary ─────────────────")
        tm = self._model.transmat_
        for i in range(self.n_states):
            row = "  ".join(f"{p:.3f}" for p in tm[i])
            logger.info(f"  State {i} ({self.regime_name(i):18s}) transitions: [{row}]")
        logger.info("── Feature Means per State ───────────")
        for i, mean_vec in enumerate(self._model.means_):
            vals = "  ".join(f"{v:+.4f}" for v in mean_vec)
            logger.info(f"  State {i} means: [{vals}]")
        logger.info("──────────────────────────────────────")

    def _align_state_labels(self, X_scaled: np.ndarray = None) -> dict:
        """
        Map HMM integer states to economic regime labels using a 5-feature
        composite TRENDING score (v3.0).

        OLD approach (gap=0.0077): used single lag-1 autocorr.
        NEW approach (target gap>0.10): weighted composite of 5 independent
        statistical signals that research shows discriminate trending vs MR.

        TRENDING_SCORE = 0.30*ER + 0.25*VR + 0.20*Hurst + 0.15*AC_multi + 0.10*TC

        Feature index map (FEATURE_COLS v3.0):
            0: log_return       1: atr_norm          2: vol_ratio
            3: vol_of_vol_rel   4: efficiency_ratio   5: variance_ratio
            6: hurst_approx    7: autocorr_multi    8: momentum_scaled
            9: trend_consistency 10: ema_slope       11: drawdown_pct
            12: skewness
        """
        if self._model is None:
            return {0: 0, 1: 1, 2: 2}

        means  = self._model.means_   # shape (n_states, n_features)
        n_feat = means.shape[1]

        # Helper: safe index access with fallback
        def idx(i, fallback=0):
            return i if n_feat > i else fallback

        IDX_ATR_NORM = idx(1)
        IDX_VOL_RATIO = idx(2)
        IDX_VOV_REL   = idx(3)
        IDX_ER        = idx(4)   # Efficiency Ratio:  HIGH = trending
        IDX_VR        = idx(5)   # Variance Ratio:    HIGH = trending
        IDX_HURST     = idx(6)   # Hurst:             HIGH = trending
        IDX_AC_MULTI  = idx(7)   # Multi-lag autocorr: HIGH = trending
        IDX_TC        = idx(9)   # Trend consistency: HIGH = trending

        # ── Step 1: HIGH_VOL = highest combined volatility ────────────────────
        vol_score = (means[:, IDX_ATR_NORM] +
                     np.clip(means[:, IDX_VOL_RATIO], 0, None) * 2.0 +
                     np.clip(means[:, IDX_VOV_REL], 0, None))
        high_vol_state = int(np.argmax(vol_score))

        # ── Step 2: TRENDING vs MEAN_REVERT via composite score ───────────────
        remaining = [i for i in range(self.n_states) if i != high_vol_state]

        def trending_composite(s: int) -> float:
            er_s  = float(means[s, IDX_ER])          # [0, 1], already bounded
            vr_s  = float(means[s, IDX_VR])          # [-1.5, 1.5]
            h_s   = float(means[s, IDX_HURST])       # [-0.5, 0.5]
            ac_s  = float(means[s, IDX_AC_MULTI])    # [-0.5, 0.5]
            tc_s  = float(means[s, IDX_TC]) - 0.5    # [0,1] → center at 0
            return (0.30 * er_s + 0.25 * vr_s +
                    0.20 * h_s  + 0.15 * ac_s + 0.10 * tc_s)

        scores         = {i: trending_composite(i) for i in remaining}
        trending_state = max(scores, key=lambda k: scores[k])
        mean_rev_state = [i for i in remaining if i != trending_state][0]

        # ── Gap score diagnostic ──────────────────────────────────────────────
        gap    = abs(scores[trending_state] - scores[mean_rev_state])
        status = "✅" if gap >= 0.10 else ("⚠️" if gap >= 0.03 else "❌")

        label_map = {
            mean_rev_state: REGIME_MEAN_REVERT,
            trending_state: REGIME_TRENDING,
            high_vol_state: REGIME_HIGH_VOL,
        }

        logger.info("── State Label Alignment (v3.0 composite) ──────────────")
        logger.info(
            f"  State {high_vol_state} → HIGH_VOL    "
            f"(vol_score={vol_score[high_vol_state]:+.4f})"
        )
        logger.info(
            f"  State {trending_state} → TRENDING    "
            f"(composite={scores[trending_state]:+.4f})"
        )
        logger.info(
            f"  State {mean_rev_state} → MEAN_REVERT "
            f"(composite={scores[mean_rev_state]:+.4f})"
        )
        logger.info(
            f"  {status} Gap score: {gap:.4f} "
            f"({'GOOD' if gap >= 0.10 else 'WEAK' if gap >= 0.03 else 'CRITICAL'})"
            f"  (target > 0.10)"
        )
        logger.info("────────────────────────────────────────────────────────")

        return label_map

    def _smooth_posterior(
        self,
        new_probs: np.ndarray,
        alpha: float = 0.25,
    ) -> np.ndarray:
        """
        EMA-smooth posterior probability vector for regime persistence.

        Prevents bar-by-bar flickering between TRENDING and MEAN_REVERT.
        With alpha=0.25 the effective window is ~4 bars = 20 minutes on M5.

        P_smooth = alpha * P_new + (1 - alpha) * P_previous

        No look-ahead: smoothing uses only historical (already-computed) posteriors.
        """
        if not self._posterior_history:
            self._posterior_history.append(new_probs.copy())
            return new_probs

        prev_smooth = self._posterior_history[-1]
        smoothed    = alpha * new_probs + (1.0 - alpha) * prev_smooth

        total = smoothed.sum()
        if total > 0:
            smoothed = smoothed / total

        self._posterior_history.append(smoothed.copy())
        if len(self._posterior_history) > 10:
            self._posterior_history.pop(0)

        return smoothed
