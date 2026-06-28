"""
Confidence Gate Module
======================

Evaluates retrieval result quality and determines whether the results are
confident enough to proceed to context building, or whether a fallback
retrieval strategy should be triggered.

The confidence score is a weighted combination of:
  - **top_score** (40%): The highest rerank/retrieval score, indicating the
    best individual match quality.
  - **score_gap** (30%): The difference between the top-1 and top-5 scores,
    reflecting how clearly the best results separate from the rest.
  - **coverage** (30%): The fraction of non-stopword query terms that appear
    in the top-3 results, measuring how well the results address all aspects
    of the query.

Usage:
    from pipeline.confidence_gate import ConfidenceGate

    gate = ConfidenceGate()
    result = gate.evaluate(query="What is GDP of India?", results=ranked_results)
    if result['passed']:
        # proceed to ACOA
    else:
        # trigger fallback retrieval
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common English stop-words (lightweight set; avoids NLTK dependency)
# ---------------------------------------------------------------------------
STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "shall", "may", "might", "can", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each", "every",
    "all", "any", "few", "more", "most", "other", "some", "such", "no",
    "only", "own", "same", "than", "too", "very", "just", "because",
    "about", "up", "down", "if", "while", "how", "what", "which", "who",
    "whom", "this", "that", "these", "those", "it", "its", "i", "me", "my",
    "we", "our", "you", "your", "he", "him", "his", "she", "her", "they",
    "them", "their",
})


class ConfidenceGate:
    """Evaluates whether retrieval results are confident enough to proceed.

    The gate uses a composite confidence score to decide if the retrieved
    results are sufficiently relevant and well-separated, or if a fallback
    retrieval strategy should be triggered.

    Attributes:
        threshold (float): Minimum confidence score to pass the gate.
    """

    def __init__(self, threshold: float = config.CONFIDENCE_THRESHOLD) -> None:
        """Initialise the ConfidenceGate.

        Args:
            threshold: Minimum confidence score required to pass the gate.
                Results below this threshold trigger the fallback path.
                Default from ``config.CONFIDENCE_THRESHOLD``.
        """
        self.threshold = threshold
        logger.info(
            "ConfidenceGate initialised  |  threshold=%.2f", self.threshold
        )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def evaluate(self, query: str, results: List[Dict]) -> Dict:
        """Evaluate the confidence of retrieval results for a given query.

        Args:
            query: The original user query string.
            results: Ranked list of retrieval/reranking results. Each result
                is a dictionary that should contain:
                - ``rerank_score`` or ``score`` (float): relevance score
                - ``chunk`` (dict): with at least ``text`` (str)

        Returns:
            Dictionary with the following keys::

                {
                    "passed":     bool,   # True if confidence >= threshold
                    "confidence": float,  # composite confidence score [0, 1]
                    "top_score":  float,  # highest result score
                    "score_gap":  float,  # score difference between #1 and #5
                    "coverage":   float,  # fraction of query terms covered
                }
        """
        if not results:
            logger.warning("ConfidenceGate.evaluate called with empty results.")
            return {
                "passed": False,
                "confidence": 0.0,
                "top_score": 0.0,
                "score_gap": 0.0,
                "coverage": 0.0,
            }

        # ------------------------------------------------------------------
        # 1. Extract scores
        # ------------------------------------------------------------------
        scores = []
        for r in results:
            score = r.get("rerank_score", r.get("score", 0.0))
            scores.append(float(score))

        scores_sorted = sorted(scores, reverse=True)

        # ------------------------------------------------------------------
        # 2. Compute component signals
        # ------------------------------------------------------------------

        # (a) top_score – highest individual score
        top_score = scores_sorted[0]

        # (b) score_gap – separation between rank-1 and rank-5
        rank5_score = scores_sorted[4] if len(scores_sorted) >= 5 else scores_sorted[-1]
        score_gap = top_score - rank5_score

        # (c) coverage – fraction of query entities in top results
        coverage = self._compute_entity_coverage(query, results)

        # ------------------------------------------------------------------
        # 3. Normalise signals to [0, 1]
        # ------------------------------------------------------------------

        # top_score normalisation: scores from cross-encoder rerankers are
        # typically in [-10, 10] or [0, 1].  We use a sigmoid-like mapping
        # for robustness.  If scores are already in [0, 1], this is ~identity.
        if top_score > 1.0:
            # Likely raw logit from cross-encoder – normalise to [0, 1]
            import math
            normalized_top_score = 1.0 / (1.0 + math.exp(-top_score))
        elif top_score < 0.0:
            import math
            normalized_top_score = 1.0 / (1.0 + math.exp(-top_score))
        else:
            normalized_top_score = top_score

        # score_gap normalisation: cap at 1.0 (gap of 1+ is excellent)
        normalized_gap = min(score_gap, 1.0) if score_gap > 0 else 0.0

        # coverage is already in [0, 1]

        # ------------------------------------------------------------------
        # 4. Compute composite confidence
        # ------------------------------------------------------------------
        confidence = (
            0.4 * normalized_top_score
            + 0.3 * normalized_gap
            + 0.3 * coverage
        )

        passed = confidence >= self.threshold

        logger.info(
            "ConfidenceGate  |  confidence=%.3f  passed=%s  "
            "(top=%.3f  gap=%.3f  cov=%.3f)",
            confidence, passed, normalized_top_score, normalized_gap, coverage,
        )

        return {
            "passed": passed,
            "confidence": confidence,
            "top_score": top_score,
            "score_gap": score_gap,
            "coverage": coverage,
        }

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _compute_entity_coverage(
        self, query: str, results: List[Dict]
    ) -> float:
        """Compute the fraction of non-stopword query terms found in top results.

        Uses simple word-overlap: for each significant word in the query,
        check whether it appears (case-insensitive) in the concatenated text
        of the top-3 results.

        Args:
            query: The original user query.
            results: Ranked list of retrieval results.

        Returns:
            Coverage ratio in [0.0, 1.0].
        """
        # Extract significant query terms (non-stopwords, length > 1)
        query_terms = {
            w.lower()
            for w in query.split()
            if w.lower() not in STOP_WORDS and len(w) > 1
        }

        if not query_terms:
            # Query consists entirely of stop-words – treat as full coverage
            return 1.0

        # Collect text from top-3 results
        top_n = min(3, len(results))
        top_text_parts = []
        for r in results[:top_n]:
            chunk = r.get("chunk", {})
            text = chunk.get("text", "") if isinstance(chunk, dict) else ""
            top_text_parts.append(text)

        combined_text = " ".join(top_text_parts).lower()

        # Count how many query terms appear in the combined text
        found = sum(1 for term in query_terms if term in combined_text)

        coverage = found / len(query_terms)
        return coverage
