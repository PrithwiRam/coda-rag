"""
Query Feature Extractor Module
================================
Extracts linguistic and structural features from user queries
to support adaptive retrieval routing in the ACORA pipeline.

Features Extracted (8 total):
1. entity_count       - Number of named entities (spaCy NER)
2. keyword_density    - Ratio of NOUN + PROPN tokens to total tokens
3. query_length       - Number of tokens (spaCy tokenization)
4. has_identifiers    - Presence of IDs, codes, serial numbers
5. has_numbers        - Presence of digits, dates, quantities
6. question_type      - 'factual_wh', 'conceptual_wh', or 'other'
7. avg_word_frequency - Ratio of stopwords to total words (proxy for word commonality)
8. abstractness_score - Ratio of abstract/conceptual terms to content words
"""

import re
import logging
from typing import Dict, List

import spacy

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants for feature extraction
# --------------------------------------------------------------------------- #

# Factual WH-words (seeking specific facts)
FACTUAL_WH_WORDS = {"who", "what", "when", "where", "which"}

# Conceptual WH-words and verbs (seeking explanation/understanding)
CONCEPTUAL_WH_WORDS = {"why", "how", "explain", "describe", "discuss"}

# Abstract/conceptual terms for abstractness scoring
ABSTRACT_TERMS = {
    # Verbs
    "explain", "describe", "analyze", "analyse", "discuss", "compare",
    "contrast", "evaluate", "assess", "interpret", "summarize", "summarise",
    "define", "illustrate", "elaborate", "clarify", "justify",
    # Nouns / conceptual words
    "impact", "effect", "influence", "relationship", "significance",
    "importance", "consequence", "implication", "principle", "theory",
    "concept", "mechanism", "process", "role", "purpose", "function",
    "difference", "similarity", "advantage", "disadvantage", "benefit",
    "limitation", "challenge", "approach", "strategy", "method",
    "perspective", "context", "framework", "paradigm", "correlation",
    "causation", "trend", "pattern", "overview",
}

# POS tags considered as content words
CONTENT_POS_TAGS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


class QueryFeatureExtractor:
    """
    Extracts 8 linguistic and structural features from a query string
    to enable adaptive retrieval routing.

    Uses spaCy for NER, POS tagging, and tokenization, plus regex
    patterns for identifier and number detection.
    """

    def __init__(self):
        """
        Initialize the QueryFeatureExtractor.

        Loads the spaCy English model (en_core_web_sm) and compiles
        regex patterns for identifier and number detection.
        """
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Install it with: python -m spacy download en_core_web_sm"
            )
            raise

        # Compile regex patterns for identifier detection
        self._identifier_patterns = [
            # Alphanumeric codes with hyphens/underscores (e.g., ABC-123, XY_456)
            re.compile(r"\b[A-Z]{2,}[-_]\d{2,}\b", re.IGNORECASE),
            # Hash-prefixed IDs (e.g., #12345)
            re.compile(r"#\d{3,}"),
            # ISBN patterns (e.g., ISBN 978-3-16-148410-0)
            re.compile(r"\bISBN[\s:-]?\d", re.IGNORECASE),
            # DOI patterns (e.g., 10.1000/xyz123)
            re.compile(r"\b10\.\d{4,}/\S+"),
            # Version numbers (e.g., v2.3.1)
            re.compile(r"\bv?\d+\.\d+\.\d+\b", re.IGNORECASE),
            # Alphanumeric IDs with mixed case (e.g., PROJ123, ID456)
            re.compile(r"\b[A-Z]{2,}\d{3,}\b"),
            # UUID-like patterns
            re.compile(
                r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                re.IGNORECASE,
            ),
        ]

        # Pattern for detecting numbers, dates, quantities
        self._number_patterns = [
            re.compile(r"\d"),  # Any digit
        ]

        logger.info("QueryFeatureExtractor initialized with %d identifier patterns.",
                     len(self._identifier_patterns))

    def extract_features(self, query: str) -> Dict:
        """
        Extract all 8 features from a query string.

        Args:
            query: The input query string.

        Returns:
            A dictionary containing all 8 features:
                - entity_count (int): Number of named entities.
                - keyword_density (float): Ratio of NOUN+PROPN to total tokens.
                - query_length (int): Number of tokens.
                - has_identifiers (bool): Whether IDs/codes are present.
                - has_numbers (bool): Whether digits/dates are present.
                - question_type (str): 'factual_wh', 'conceptual_wh', or 'other'.
                - avg_word_frequency (float): Stopword ratio (proxy for commonality).
                - abstractness_score (float): Ratio of abstract terms to content words.
        """
        if not query or not query.strip():
            logger.warning("Empty query received, returning default features.")
            return {
                "entity_count": 0,
                "keyword_density": 0.0,
                "query_length": 0,
                "has_identifiers": False,
                "has_numbers": False,
                "question_type": "other",
                "avg_word_frequency": 0.0,
                "abstractness_score": 0.0,
            }

        # Process with spaCy
        doc = self.nlp(query)

        # Filter out whitespace and punctuation-only tokens for meaningful counts
        tokens = [token for token in doc if not token.is_space]
        total_tokens = len(tokens)

        # --- Feature 1: Entity Count ---
        entity_count = len(doc.ents)

        # --- Feature 2: Keyword Density ---
        # Ratio of (NOUN + PROPN) POS tags to total tokens
        if total_tokens > 0:
            noun_propn_count = sum(
                1 for token in tokens if token.pos_ in {"NOUN", "PROPN"}
            )
            keyword_density = noun_propn_count / total_tokens
        else:
            keyword_density = 0.0

        # --- Feature 3: Query Length ---
        query_length = total_tokens

        # --- Feature 4: Has Identifiers ---
        has_identifiers = self._has_identifiers(query)

        # --- Feature 5: Has Numbers ---
        has_numbers = any(
            pattern.search(query) for pattern in self._number_patterns
        )

        # --- Feature 6: Question Type ---
        question_type = self._detect_question_type(query)

        # --- Feature 7: Average Word Frequency ---
        # Heuristic: ratio of stopwords to total words
        # High ratio => common/frequent words => more semantic query
        words = [token for token in tokens if token.is_alpha]
        if len(words) > 0:
            stopword_count = sum(1 for word in words if word.is_stop)
            avg_word_frequency = stopword_count / len(words)
        else:
            avg_word_frequency = 0.0

        # --- Feature 8: Abstractness Score ---
        abstractness_score = self._compute_abstractness(doc)

        features = {
            "entity_count": entity_count,
            "keyword_density": round(keyword_density, 4),
            "query_length": query_length,
            "has_identifiers": has_identifiers,
            "has_numbers": has_numbers,
            "question_type": question_type,
            "avg_word_frequency": round(avg_word_frequency, 4),
            "abstractness_score": round(abstractness_score, 4),
        }

        logger.debug("Extracted features for query '%s': %s", query[:60], features)
        return features

    def _detect_question_type(self, query: str) -> str:
        """
        Classify the question type based on leading words.

        The classification checks the first few words of the query
        (case-insensitive) against known WH-word sets.

        Args:
            query: The input query string.

        Returns:
            - 'factual_wh' for who/what/when/where/which queries.
            - 'conceptual_wh' for why/how/explain/describe/discuss queries.
            - 'other' for all other query forms.
        """
        # Normalize and tokenize for matching
        query_lower = query.lower().strip()
        words = query_lower.split()

        if not words:
            return "other"

        # Check the first few words for question type indicators
        # (some queries start with "can you explain..." or "please describe...")
        check_words = set(words[:3])

        # Check for conceptual WH-words (check first to catch "how" before "what")
        if check_words & CONCEPTUAL_WH_WORDS:
            return "conceptual_wh"

        # Check for factual WH-words
        if check_words & FACTUAL_WH_WORDS:
            return "factual_wh"

        return "other"

    def _has_identifiers(self, query: str) -> bool:
        """
        Check if the query contains identifiers, codes, or serial numbers.

        Uses compiled regex patterns to detect patterns like:
        - ABC-123, XY_456 (alphanumeric codes)
        - #12345 (hash-prefixed IDs)
        - ISBN numbers
        - DOI patterns
        - Version numbers (v2.3.1)
        - UUID patterns

        Args:
            query: The input query string.

        Returns:
            True if any identifier pattern matches, False otherwise.
        """
        return any(
            pattern.search(query) for pattern in self._identifier_patterns
        )

    def _compute_abstractness(self, doc) -> float:
        """
        Compute the abstractness score of a query.

        The score is the ratio of abstract/conceptual terms (from the
        ABSTRACT_TERMS set) to the total number of content words
        (NOUN, PROPN, VERB, ADJ, ADV).

        Args:
            doc: A spaCy Doc object of the processed query.

        Returns:
            A float between 0.0 and 1.0, where higher values indicate
            more abstract/conceptual queries.
        """
        # Get content words (non-stop, non-punct, relevant POS)
        content_words = [
            token for token in doc
            if token.pos_ in CONTENT_POS_TAGS
            and not token.is_stop
            and not token.is_punct
            and not token.is_space
        ]

        if not content_words:
            return 0.0

        # Count how many content words are in the abstract terms set
        abstract_count = sum(
            1 for token in content_words
            if token.lemma_.lower() in ABSTRACT_TERMS
            or token.text.lower() in ABSTRACT_TERMS
        )

        return abstract_count / len(content_words)

    def to_feature_vector(self, features: Dict) -> List[float]:
        """
        Convert a feature dictionary to a numerical vector for ML classification.

        The vector ordering is fixed and consistent:
        [entity_count, keyword_density, query_length, has_identifiers,
         has_numbers, question_type_factual, question_type_conceptual,
         avg_word_frequency, abstractness_score]

        Note: Boolean and categorical features are encoded numerically:
        - has_identifiers: True -> 1.0, False -> 0.0
        - has_numbers: True -> 1.0, False -> 0.0
        - question_type: one-hot encoded as two binary features
          (factual_wh, conceptual_wh); 'other' = (0, 0)

        Args:
            features: A feature dictionary from extract_features().

        Returns:
            A list of floats representing the feature vector (length 9).
        """
        # Encode question type as two binary features
        question_type = features.get("question_type", "other")
        is_factual_wh = 1.0 if question_type == "factual_wh" else 0.0
        is_conceptual_wh = 1.0 if question_type == "conceptual_wh" else 0.0

        vector = [
            float(features.get("entity_count", 0)),
            float(features.get("keyword_density", 0.0)),
            float(features.get("query_length", 0)),
            1.0 if features.get("has_identifiers", False) else 0.0,
            1.0 if features.get("has_numbers", False) else 0.0,
            is_factual_wh,
            is_conceptual_wh,
            float(features.get("avg_word_frequency", 0.0)),
            float(features.get("abstractness_score", 0.0)),
        ]

        return vector
