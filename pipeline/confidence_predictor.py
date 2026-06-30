"""
Confidence Predictor
====================
Learned confidence predictor replacing the rule-based confidence gate.

Uses an MLPClassifier (scikit-learn) trained on retrieval quality features
to predict whether retrieved context is sufficient for accurate generation.

Features (8-dim):
  1. top_score      - maximum retrieval score
  2. score_gap      - difference between top score and 5th score
  3. score_std      - standard deviation of top-10 scores
  4. coverage       - estimated relevance coverage
  5. num_results    - number of retrieved chunks
  6. avg_score      - mean of retrieval scores
  7. score_entropy  - entropy of score distribution
  8. query_length   - number of words in query

Training data can be generated from evaluation runs where ground-truth
relevance labels are available, or from heuristic labels based on
whether the top score exceeds a threshold.
"""
import sys
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class ConfidencePredictor:
    """Learned confidence predictor using MLPClassifier.

    Replaces the static 3-rule heuristic (top_score > τ, gap > δ, coverage > γ)
    with a trained neural network that learns to predict retrieval quality
    from 8-dimensional feature vectors.

    Attributes:
        model: sklearn MLPClassifier (hidden=(32, 16), max_iter=1000).
        is_trained: Whether the model has been fit on training data.
        scaler: StandardScaler for feature normalization.
    """

    def __init__(self):
        """Initialize the confidence predictor with MLP model."""
        self.model = None
        self.scaler = None
        self.is_trained = False

        try:
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler

            self.model = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
            )
            self.scaler = StandardScaler()
            logger.info("ConfidencePredictor: MLPClassifier initialized (32→16)")
        except Exception as e:
            logger.warning("ConfidencePredictor: sklearn unavailable: %s", e)

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------

    def extract_features(self, results: List[Dict], query: str) -> np.ndarray:
        """Extract 8-dimensional feature vector from retrieval results.

        Args:
            results: List of retrieval result dicts with 'score' key.
            query: The original query string.

        Returns:
            8-dim numpy array of features.
        """
        if not results:
            return np.zeros(8, dtype=np.float64)

        # Extract scores
        scores = np.array([r.get('score', 0.0) for r in results], dtype=np.float64)

        # Feature 1: Top score
        top_score = float(np.max(scores)) if len(scores) > 0 else 0.0

        # Feature 2: Score gap (top - 5th, or top if fewer than 5)
        if len(scores) >= 5:
            sorted_scores = np.sort(scores)[::-1]
            score_gap = float(sorted_scores[0] - sorted_scores[4])
        else:
            score_gap = top_score  # Use top_score as gap when few results

        # Feature 3: Score standard deviation (top-10)
        top_n = scores[:min(10, len(scores))]
        score_std = float(np.std(top_n)) if len(top_n) > 1 else 0.0

        # Feature 4: Coverage (estimated from score distribution)
        # Higher coverage = more scores above mean
        if len(scores) > 0:
            mean_score = float(np.mean(scores))
            coverage = float(np.sum(scores > mean_score * 0.5) / len(scores))
        else:
            coverage = 0.0

        # Feature 5: Number of results (normalized by 30)
        num_results = min(float(len(results)) / 30.0, 1.0)

        # Feature 6: Average score
        avg_score = float(np.mean(scores)) if len(scores) > 0 else 0.0

        # Feature 7: Score entropy (normalized)
        if len(scores) > 1 and np.sum(scores) > 0:
            probs = scores / (np.sum(scores) + 1e-8)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            max_entropy = np.log(len(scores))
            score_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        else:
            score_entropy = 0.0

        # Feature 8: Query length (normalized by 30)
        query_length = min(float(len(query.split())) / 30.0, 1.0)

        features = np.array([
            top_score, score_gap, score_std, coverage,
            num_results, avg_score, score_entropy, query_length
        ], dtype=np.float64)

        return features

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_confidence(self, results: List[Dict], query: str) -> Dict:
        """Predict confidence score for retrieval results.

        Uses trained MLP if available, otherwise falls back to heuristic.

        Args:
            results: List of retrieval result dicts.
            query: The original query string.

        Returns:
            Dictionary with 'confident' (bool), 'score' (float),
            'passed' (bool), 'confidence' (float), 'features' (list).
        """
        features = self.extract_features(results, query)

        # Try MLP prediction if trained
        if self.is_trained and self.model is not None and self.scaler is not None:
            try:
                features_scaled = self.scaler.transform([features])
                proba = self.model.predict_proba(features_scaled)[0]

                if len(proba) > 1:
                    score = float(proba[1])  # Probability of class 1 (confident)
                else:
                    score = float(proba[0])

                passed = score >= 0.5

                return {
                    'confident': passed,
                    'passed': passed,
                    'score': score,
                    'confidence': score,
                    'features': features.tolist(),
                    'method': 'mlp',
                }
            except Exception as e:
                logger.warning("MLP prediction failed: %s. Using heuristic.", e)

        # Fallback: heuristic scoring
        score = self._heuristic_score(features)
        passed = score >= config.CONFIDENCE_THRESHOLD

        return {
            'confident': passed,
            'passed': passed,
            'score': float(score),
            'confidence': float(score),
            'features': features.tolist(),
            'method': 'heuristic',
        }

    def _heuristic_score(self, features: np.ndarray) -> float:
        """Compute heuristic confidence score from features.

        Weighted combination: 0.4 * top_score + 0.3 * score_gap + 0.3 * coverage.

        Args:
            features: 8-dim feature vector.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        top_score = features[0]
        score_gap = features[1]
        coverage = features[3]

        # Normalize to [0, 1]
        score = 0.4 * min(top_score, 1.0) + 0.3 * min(score_gap, 1.0) + 0.3 * coverage
        return min(max(score, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the MLP classifier on labeled data.

        Args:
            X: Feature matrix of shape (n_samples, 8).
            y: Binary labels of shape (n_samples,). 1 = confident, 0 = not.

        Returns:
            Training metrics dictionary.
        """
        if self.model is None or self.scaler is None:
            logger.error("Cannot train: model or scaler not initialized.")
            return {"error": "Model not initialized"}

        if len(X) < 10:
            logger.warning("Too few training samples (%d). Need at least 10.", len(X))
            return {"error": "Insufficient training data"}

        try:
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)

            # Train
            self.model.fit(X_scaled, y)
            self.is_trained = True

            # Compute training accuracy
            train_acc = self.model.score(X_scaled, y)

            logger.info(
                "ConfidencePredictor trained on %d samples. Train accuracy: %.3f",
                len(X), train_acc
            )

            return {
                "n_samples": len(X),
                "train_accuracy": float(train_acc),
                "n_iter": self.model.n_iter_,
                "classes": self.model.classes_.tolist(),
            }

        except Exception as e:
            logger.error("Training failed: %s", e)
            return {"error": str(e)}

    def train_from_results(
        self,
        results_list: List[Dict],
        quality_threshold: float = 0.5,
    ) -> Dict:
        """Train on a list of query results with heuristic labels.

        Labels are generated heuristically: confident=1 if top_score >= threshold.

        Args:
            results_list: List of result dicts from pipeline queries.
            quality_threshold: Score threshold for positive label.

        Returns:
            Training metrics dictionary.
        """
        X_list = []
        y_list = []

        for result in results_list:
            retrieval_results = result.get('reranked_results', [])
            query = result.get('query', '')

            if not retrieval_results:
                continue

            features = self.extract_features(retrieval_results, query)
            X_list.append(features)

            # Heuristic label: confident if top score >= threshold
            top_score = features[0]
            label = 1 if top_score >= quality_threshold else 0
            y_list.append(label)

        if not X_list:
            logger.warning("No valid training samples found.")
            return {"error": "No training samples"}

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int32)

        return self.train(X, y)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save trained model and scaler to disk.

        Args:
            path: File path for the saved model (without extension).
        """
        import joblib

        if self.model is not None and self.is_trained:
            joblib.dump(self.model, f"{path}_model.joblib")
            joblib.dump(self.scaler, f"{path}_scaler.joblib")
            logger.info("ConfidencePredictor saved to %s", path)
        else:
            logger.warning("Cannot save: model not trained.")

    def load(self, path: str) -> bool:
        """Load trained model and scaler from disk.

        Args:
            path: File path for the saved model (without extension).

        Returns:
            True if loaded successfully, False otherwise.
        """
        import joblib

        try:
            self.model = joblib.load(f"{path}_model.joblib")
            self.scaler = joblib.load(f"{path}_scaler.joblib")
            self.is_trained = True
            logger.info("ConfidencePredictor loaded from %s", path)
            return True
        except Exception as e:
            logger.warning("Failed to load ConfidencePredictor: %s", e)
            return False