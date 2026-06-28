"""
Evaluation Metrics Module
=========================

Provides the ``EvaluationMetrics`` class with static methods for computing
standard Information Retrieval and Natural Language Generation metrics.

**Retrieval metrics** (operate on document/chunk IDs):
  - Precision@K
  - Recall@K
  - Mean Reciprocal Rank (MRR)
  - Normalised Discounted Cumulative Gain (nDCG@K)
  - Hit Rate@K

**Generation metrics** (operate on answer strings):
  - Exact Match (EM)
  - Token-level F1 Score

All methods are static to allow convenient use without instantiation.

Usage:
    from evaluation.metrics import EvaluationMetrics

    p5 = EvaluationMetrics.precision_at_k(retrieved, relevant, k=5)
    f1 = EvaluationMetrics.f1_score(predicted_answer, reference_answer)
    all_metrics = EvaluationMetrics.compute_all_metrics(...)
"""

import re
import sys
import math
import logging
from pathlib import Path
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Collection of static methods for retrieval and generation evaluation.

    All metrics follow the convention of returning a float in [0.0, 1.0].
    """

    # ===================================================================
    # Retrieval Metrics
    # ===================================================================

    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        """Compute Precision@K.

        Precision@K = |{relevant docs in top-K retrieved}| / K

        Args:
            retrieved_ids: Ordered list of retrieved document/chunk IDs.
            relevant_ids: Set of ground-truth relevant IDs.
            k: Cutoff rank.

        Returns:
            Precision@K value in [0.0, 1.0].
        """
        if k <= 0:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        hits = sum(1 for doc_id in top_k if doc_id in relevant_set)

        return hits / k

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        """Compute Recall@K.

        Recall@K = |{relevant docs in top-K retrieved}| / |{all relevant docs}|

        Args:
            retrieved_ids: Ordered list of retrieved document/chunk IDs.
            relevant_ids: Set of ground-truth relevant IDs.
            k: Cutoff rank.

        Returns:
            Recall@K value in [0.0, 1.0].
        """
        if not relevant_ids or k <= 0:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        hits = sum(1 for doc_id in top_k if doc_id in relevant_set)

        return hits / len(relevant_set)

    @staticmethod
    def mrr(
        retrieved_ids: List[str],
        relevant_ids: List[str],
    ) -> float:
        """Compute Mean Reciprocal Rank (MRR).

        MRR = 1 / rank_of_first_relevant_result

        If no relevant result is found, returns 0.0.

        Args:
            retrieved_ids: Ordered list of retrieved document/chunk IDs.
            relevant_ids: Set of ground-truth relevant IDs.

        Returns:
            MRR value in [0.0, 1.0].
        """
        relevant_set = set(relevant_ids)

        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        """Compute Normalised Discounted Cumulative Gain at K (nDCG@K).

        Uses binary relevance: rel(doc) = 1 if relevant, else 0.

        DCG@K   = Σ_{i=1}^{K} rel(i) / log₂(i + 1)
        IDCG@K  = Σ_{i=1}^{min(K, |relevant|)} 1 / log₂(i + 1)
        nDCG@K  = DCG@K / IDCG@K

        Args:
            retrieved_ids: Ordered list of retrieved document/chunk IDs.
            relevant_ids: Set of ground-truth relevant IDs.
            k: Cutoff rank.

        Returns:
            nDCG@K value in [0.0, 1.0].
        """
        if k <= 0 or not relevant_ids:
            return 0.0

        relevant_set = set(relevant_ids)

        # DCG@K
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in relevant_set:
                dcg += 1.0 / math.log2(i + 1)

        # Ideal DCG@K (all relevant docs ranked at the top)
        ideal_hits = min(k, len(relevant_set))
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def hit_rate_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int,
    ) -> float:
        """Compute Hit Rate at K (a.k.a. Success@K).

        Returns 1.0 if at least one relevant document appears in the top-K
        retrieved results, otherwise 0.0.

        Args:
            retrieved_ids: Ordered list of retrieved document/chunk IDs.
            relevant_ids: Set of ground-truth relevant IDs.
            k: Cutoff rank.

        Returns:
            1.0 or 0.0.
        """
        if k <= 0:
            return 0.0

        relevant_set = set(relevant_ids)
        top_k = retrieved_ids[:k]

        return 1.0 if any(doc_id in relevant_set for doc_id in top_k) else 0.0

    # ===================================================================
    # Generation (Answer) Metrics
    # ===================================================================

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalise answer text for comparison.

        Steps:
            1. Lowercase
            2. Remove punctuation
            3. Remove articles (a, an, the)
            4. Collapse whitespace

        Args:
            text: Raw answer text.

        Returns:
            Normalised text string.
        """
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def exact_match(predicted: str, reference: str) -> float:
        """Compute Exact Match (EM) score.

        Both strings are normalised (lowercased, punctuation removed,
        articles removed) before comparison.

        Args:
            predicted: Model-generated answer.
            reference: Ground-truth reference answer.

        Returns:
            1.0 if normalised strings are identical, else 0.0.
        """
        norm_predicted = EvaluationMetrics._normalize_answer(predicted)
        norm_reference = EvaluationMetrics._normalize_answer(reference)

        return 1.0 if norm_predicted == norm_reference else 0.0

    @staticmethod
    def f1_score(predicted: str, reference: str) -> float:
        """Compute token-level F1 score between predicted and reference answers.

        Both strings are normalised, then split into token bags.
        F1 = 2 * precision * recall / (precision + recall)

        Args:
            predicted: Model-generated answer.
            reference: Ground-truth reference answer.

        Returns:
            Token-level F1 score in [0.0, 1.0].
        """
        norm_predicted = EvaluationMetrics._normalize_answer(predicted)
        norm_reference = EvaluationMetrics._normalize_answer(reference)

        pred_tokens = norm_predicted.split()
        ref_tokens = norm_reference.split()

        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        # Count common tokens (order-independent, with multiplicity)
        from collections import Counter
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)

        common = sum((pred_counter & ref_counter).values())

        if common == 0:
            return 0.0

        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    # ===================================================================
    # Aggregate
    # ===================================================================

    @staticmethod
    def compute_all_metrics(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        predicted_answer: str,
        reference_answer: str,
        k_values: Optional[List[int]] = None,
    ) -> Dict:
        """Compute all retrieval and generation metrics at once.

        Args:
            retrieved_ids: Ordered list of retrieved document/chunk IDs.
            relevant_ids: Ground-truth relevant document/chunk IDs.
            predicted_answer: Model-generated answer string.
            reference_answer: Ground-truth reference answer string.
            k_values: List of K cutoff values for @K metrics.
                Default from ``config.EVAL_K_VALUES``.

        Returns:
            Dictionary with all metrics::

                {
                    "retrieval": {
                        "precision@1": float, "precision@3": float, ...,
                        "recall@1": float, ...,
                        "ndcg@1": float, ...,
                        "hit_rate@1": float, ...,
                        "mrr": float,
                    },
                    "generation": {
                        "exact_match": float,
                        "f1_score": float,
                    }
                }
        """
        if k_values is None:
            k_values = config.EVAL_K_VALUES

        # ------------------------------------------------------------------
        # Retrieval metrics
        # ------------------------------------------------------------------
        retrieval_metrics = {}

        for k in k_values:
            retrieval_metrics[f"precision@{k}"] = EvaluationMetrics.precision_at_k(
                retrieved_ids, relevant_ids, k
            )
            retrieval_metrics[f"recall@{k}"] = EvaluationMetrics.recall_at_k(
                retrieved_ids, relevant_ids, k
            )
            retrieval_metrics[f"ndcg@{k}"] = EvaluationMetrics.ndcg_at_k(
                retrieved_ids, relevant_ids, k
            )
            retrieval_metrics[f"hit_rate@{k}"] = EvaluationMetrics.hit_rate_at_k(
                retrieved_ids, relevant_ids, k
            )

        retrieval_metrics["mrr"] = EvaluationMetrics.mrr(
            retrieved_ids, relevant_ids
        )

        # ------------------------------------------------------------------
        # Generation metrics
        # ------------------------------------------------------------------
        generation_metrics = {
            "exact_match": EvaluationMetrics.exact_match(
                predicted_answer, reference_answer
            ),
            "f1_score": EvaluationMetrics.f1_score(
                predicted_answer, reference_answer
            ),
        }

        all_metrics = {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
        }

        logger.debug("Computed all metrics: %s", all_metrics)

        return all_metrics
