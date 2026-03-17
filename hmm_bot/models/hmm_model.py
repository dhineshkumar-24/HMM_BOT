"""
models/hmm_model.py — Hidden Markov Model regime detector.

Wraps hmmlearn's GaussianHMM to:
  1. Train on a feature matrix (fit)
  2. Predict the current market regime (predict_regime)
  3. Return posterior probabilities (get_probabilities)
  4. Persist/load the trained model to disk (save / load)

PLACEHOLDER: All methods are structurally complete and documented.
Training and inference will be wired in Phase 2.

Dependencies (add to requirements.txt):
    hmmlearn>=0.3.0
    scikit-learn>=1.3.0
    joblib>=1.3.0
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


from utils.logger import setup_logger

logger = setup_logger("HMMModel")

# Regime label constants — must match settings.yaml hmm.regime_names
REGIME_MEAN_REVERT = 0
REGIME_TRENDING    = 1
REGIME_NOISY       = 2


class HMMModel:
    """
    Gaussian Hidden Markov Model for market regime detection.

    Regimes:
        0 → Mean-reverting (low vol, low Hurst)
        1 → Trending       (high Hurst, momentum)
        2 → Noisy / Flat   (chaotic, avoid trading)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full settings dict (reads from config['hmm']).
        """
        hmm_cfg = config.get("hmm", {})

        self.n_components       = hmm_cfg.get("n_components", 3)
        self.n_iter             = hmm_cfg.get("n_iter", 150)
        self.covariance_type    = hmm_cfg.get("covariance_type", "full")
        self.model_path         = hmm_cfg.get("model_path", "models/hmm_state.pkl")
        self.confidence_threshold = hmm_cfg.get("confidence_threshold", 0.65)

        self._model = None        # hmmlearn.GaussianHMM instance
        self._is_trained = False  # Guard flag

        logger.info(
            f"HMMModel created — {self.n_components} states, "
            f"{self.covariance_type} covariance, {self.n_iter} EM iterations."
        )

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray | pd.DataFrame) -> None:
        """
        Train the HMM on a feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
               Use utils.features.build_feature_matrix() to generate.

        TODO (Phase 2):
            from hmmlearn import hmm
            self._model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=42,
            )
            self._model.fit(X)
            self._is_trained = True
            logger.info(f"HMM trained on {len(X)} samples.")
        """
        logger.warning("HMMModel.fit() called — PLACEHOLDER. Training not yet implemented.")
        self._is_trained = False

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_regime(self, X: np.ndarray | pd.DataFrame) -> int:
        """
        Predict the most likely regime for the most recent observation.

        Args:
            X: Feature vector of shape (1, n_features) or (n_samples, n_features).
               The last row is used as the current observation.

        Returns:
            Integer regime label (0, 1, or 2).
            Returns REGIME_MEAN_REVERT (0) during warm-up.

        TODO (Phase 2):
            if not self._is_trained:
                return REGIME_MEAN_REVERT
            hidden_states = self._model.predict(X)
            return int(hidden_states[-1])
        """
        logger.debug("HMMModel.predict_regime() — returning warm-up default (0).")
        return REGIME_MEAN_REVERT

    def get_probabilities(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Return the posterior state probabilities for the most recent observation.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_components,) with probabilities summing to 1.
            Returns uniform distribution during warm-up.

        TODO (Phase 2):
            _, posteriors = self._model.score_samples(X)
            return posteriors[-1]
        """
        uniform = np.ones(self.n_components) / self.n_components
        return uniform

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        """
        Persist the trained model to disk using joblib.

        TODO (Phase 2):
            import joblib
            joblib.dump(self._model, path or self.model_path)
        """
        logger.warning("HMMModel.save() — PLACEHOLDER. Persistence not yet implemented.")

    def load(self, path: str | None = None) -> bool:
        """
        Load a previously trained model from disk.

        Returns:
            True if loaded successfully, False otherwise.

        TODO (Phase 2):
            import joblib
            p = path or self.model_path
            if not os.path.exists(p):
                return False
            self._model = joblib.load(p)
            self._is_trained = True
            return True
        """
        logger.warning("HMMModel.load() — PLACEHOLDER. Persistence not yet implemented.")
        return False

    @property
    def is_trained(self) -> bool:
        return self._is_trained
