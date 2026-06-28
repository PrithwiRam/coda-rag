"""
Query Classifier Module
========================
Classifies user queries into retrieval strategy categories
(LEXICAL, SEMANTIC, AMBIGUOUS) using a two-stage approach:

1. **Rule-Based Classification**: Fast, interpretable rules based on
   extracted query features. Used when confidence exceeds threshold.
2. **ML Classification**: LogisticRegression fallback trained on labeled
   query data, used when rule-based confidence is below threshold.

Classification Labels (from config):
- LEXICAL:   Query is best served by keyword/BM25 retrieval.
- SEMANTIC:  Query is best served by dense/embedding retrieval.
- AMBIGUOUS: Query could benefit from hybrid retrieval.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

import config
from pipeline.query_feature_extractor import QueryFeatureExtractor

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Two-stage query classifier that routes queries to the optimal
    retrieval strategy (LEXICAL, SEMANTIC, or AMBIGUOUS).

    Stage 1: Rule-based classification with confidence scoring.
    Stage 2: ML-based classification (LogisticRegression) as fallback.

    Attributes:
        feature_extractor: QueryFeatureExtractor instance for feature extraction.
        confidence_threshold: Minimum confidence for rule-based results to be accepted.
        ml_model: Trained LogisticRegression model (or None if not loaded/trained).
    """

    def __init__(
        self,
        feature_extractor: Optional[QueryFeatureExtractor] = None,
        confidence_threshold: float = config.CLASSIFIER_CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize the QueryClassifier.

        Args:
            feature_extractor: A QueryFeatureExtractor instance. If None,
                               a new one will be created.
            confidence_threshold: Minimum confidence for rule-based classification
                                  to be accepted without ML fallback.
                                  Default from config.CLASSIFIER_CONFIDENCE_THRESHOLD.
        """
        self.feature_extractor = feature_extractor or QueryFeatureExtractor()
        self.confidence_threshold = confidence_threshold
        self.ml_model: Optional[LogisticRegression] = None

        # Attempt to load a pre-trained ML model if it exists
        model_path = config.CLASSIFIER_MODEL_PATH
        if Path(model_path).exists():
            self.load_model(model_path)
            logger.info("Loaded ML classifier model from %s", model_path)
        else:
            logger.info(
                "No ML classifier model found at %s. "
                "Using rule-based classification only.",
                model_path,
            )

        logger.info(
            "QueryClassifier initialized (confidence_threshold=%.2f, ml_model=%s)",
            self.confidence_threshold,
            "loaded" if self.ml_model is not None else "not loaded",
        )

    def classify(self, query: str) -> Dict:
        """
        Classify a query into a retrieval strategy.

        Classification proceeds in priority order:
        1. Extract features from the query.
        2. Apply rule-based classification.
        3. If rule-based confidence >= threshold, return that result.
        4. Otherwise, try ML classification if a model is loaded.
        5. If no ML model, return AMBIGUOUS as fallback.

        Args:
            query: The input query string.

        Returns:
            A dictionary with:
                - 'label' (str): One of LEXICAL, SEMANTIC, AMBIGUOUS.
                - 'confidence' (float): Classification confidence (0.0 to 1.0).
                - 'method' (str): 'rule_based', 'ml', or 'fallback'.
                - 'features' (dict): The extracted feature dictionary.
        """
        if not query or not query.strip():
            logger.warning("Empty query received, returning AMBIGUOUS fallback.")
            return {
                "label": config.QUERY_TYPE_AMBIGUOUS,
                "confidence": 0.0,
                "method": "fallback",
                "features": {},
            }

        # Step 1: Extract features
        features = self.feature_extractor.extract_features(query)

        # Step 2: Try rule-based classification
        rule_result = self._rule_based_classify(features)

        # Step 3: Check if rule-based confidence is sufficient
        if rule_result["confidence"] >= self.confidence_threshold:
            logger.info(
                "Rule-based classification: '%s' -> %s (confidence=%.2f)",
                query[:60],
                rule_result["label"],
                rule_result["confidence"],
            )
            return {
                "label": rule_result["label"],
                "confidence": rule_result["confidence"],
                "method": "rule_based",
                "features": features,
            }

        # Step 4: Try ML classification if model is available
        if self.ml_model is not None:
            ml_result = self._ml_classify(features)
            logger.info(
                "ML classification: '%s' -> %s (confidence=%.2f)",
                query[:60],
                ml_result["label"],
                ml_result["confidence"],
            )
            return {
                "label": ml_result["label"],
                "confidence": ml_result["confidence"],
                "method": "ml",
                "features": features,
            }

        # Step 5: Fallback to AMBIGUOUS
        logger.info(
            "Fallback classification: '%s' -> AMBIGUOUS (no ML model, "
            "rule confidence=%.2f below threshold=%.2f)",
            query[:60],
            rule_result["confidence"],
            self.confidence_threshold,
        )
        return {
            "label": config.QUERY_TYPE_AMBIGUOUS,
            "confidence": rule_result["confidence"],
            "method": "fallback",
            "features": features,
        }

    def _rule_based_classify(self, features: Dict) -> Dict:
        """
        Apply deterministic rules to classify a query based on its features.

        Rules are evaluated in priority order (first match wins):

        1. has_identifiers == True -> LEXICAL (0.95)
        2. entity_count >= 3 AND question_type == 'factual_wh' -> LEXICAL (0.90)
        3. abstractness_score > 0.6 AND question_type == 'conceptual_wh' -> SEMANTIC (0.90)
        4. query_length > 15 AND keyword_density < 0.3 -> SEMANTIC (0.85)
        5. entity_count == 0 AND question_type == 'conceptual_wh' -> SEMANTIC (0.85)
        6. Otherwise -> AMBIGUOUS (0.50)

        Args:
            features: Feature dictionary from QueryFeatureExtractor.

        Returns:
            A dictionary with 'label' and 'confidence' keys.
        """
        # Rule 1: Identifiers present -> LEXICAL (high confidence)
        if features.get("has_identifiers", False):
            return {
                "label": config.QUERY_TYPE_LEXICAL,
                "confidence": 0.95,
            }

        # Rule 2: Many entities + factual question -> LEXICAL
        if (
            features.get("entity_count", 0) >= 3
            and features.get("question_type") == "factual_wh"
        ):
            return {
                "label": config.QUERY_TYPE_LEXICAL,
                "confidence": 0.90,
            }

        # Rule 3: Highly abstract + conceptual question -> SEMANTIC
        if (
            features.get("abstractness_score", 0.0) > 0.6
            and features.get("question_type") == "conceptual_wh"
        ):
            return {
                "label": config.QUERY_TYPE_SEMANTIC,
                "confidence": 0.90,
            }

        # Rule 4: Long query with low keyword density -> SEMANTIC
        if (
            features.get("query_length", 0) > 15
            and features.get("keyword_density", 0.0) < 0.3
        ):
            return {
                "label": config.QUERY_TYPE_SEMANTIC,
                "confidence": 0.85,
            }

        # Rule 5: No entities + conceptual question -> SEMANTIC
        if (
            features.get("entity_count", 0) == 0
            and features.get("question_type") == "conceptual_wh"
        ):
            return {
                "label": config.QUERY_TYPE_SEMANTIC,
                "confidence": 0.85,
            }

        # Rule 6: Default fallback -> AMBIGUOUS
        return {
            "label": config.QUERY_TYPE_AMBIGUOUS,
            "confidence": 0.50,
        }

    def _ml_classify(self, features: Dict) -> Dict:
        """
        Classify a query using the trained ML model (LogisticRegression).

        Converts features to a numerical vector, runs prediction,
        and returns the predicted label with its probability.

        Args:
            features: Feature dictionary from QueryFeatureExtractor.

        Returns:
            A dictionary with 'label' and 'confidence' keys.

        Raises:
            RuntimeError: If no ML model is loaded.
        """
        if self.ml_model is None:
            raise RuntimeError("No ML model loaded for classification.")

        # Convert features to numerical vector
        feature_vector = self.feature_extractor.to_feature_vector(features)
        X = np.array([feature_vector])

        # Predict class and probability
        prediction = self.ml_model.predict(X)[0]
        probabilities = self.ml_model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))

        return {
            "label": prediction,
            "confidence": confidence,
        }

    def train(self, queries: List[str], labels: List[str]) -> None:
        """
        Train the ML classifier on labeled query data.

        Extracts features from each query, converts to feature vectors,
        and trains a LogisticRegression model.

        Args:
            queries: A list of query strings.
            labels: A list of corresponding labels (LEXICAL, SEMANTIC, AMBIGUOUS).

        Raises:
            ValueError: If queries and labels have different lengths or are empty.
        """
        if len(queries) != len(labels):
            raise ValueError(
                f"Mismatch: {len(queries)} queries but {len(labels)} labels."
            )
        if not queries:
            raise ValueError("Cannot train on empty data.")

        logger.info("Training ML classifier on %d labeled queries...", len(queries))

        # Extract features and build feature matrix
        feature_vectors = []
        for query in queries:
            features = self.feature_extractor.extract_features(query)
            vector = self.feature_extractor.to_feature_vector(features)
            feature_vectors.append(vector)

        X = np.array(feature_vectors)
        y = np.array(labels)

        # Train LogisticRegression
        self.ml_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class="multinomial",
            solver="lbfgs",
        )
        self.ml_model.fit(X, y)

        # Log training accuracy
        train_accuracy = self.ml_model.score(X, y)
        logger.info(
            "ML classifier trained successfully. Training accuracy: %.4f",
            train_accuracy,
        )

    def save_model(self, path: Optional[str] = None) -> None:
        """
        Save the trained ML model to disk using joblib.

        Args:
            path: File path to save the model. If None, uses
                  config.CLASSIFIER_MODEL_PATH.

        Raises:
            RuntimeError: If no ML model has been trained.
        """
        if self.ml_model is None:
            raise RuntimeError("No ML model to save. Train a model first.")

        save_path = Path(path) if path else config.CLASSIFIER_MODEL_PATH

        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.ml_model, save_path)
        logger.info("ML classifier model saved to %s", save_path)

    def load_model(self, path: Optional[str] = None) -> None:
        """
        Load a trained ML model from disk using joblib.

        Args:
            path: File path to load the model from. If None, uses
                  config.CLASSIFIER_MODEL_PATH.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        load_path = Path(path) if path else config.CLASSIFIER_MODEL_PATH

        if not load_path.exists():
            raise FileNotFoundError(
                f"ML model file not found at {load_path}"
            )

        self.ml_model = joblib.load(load_path)
        logger.info("ML classifier model loaded from %s", load_path)
