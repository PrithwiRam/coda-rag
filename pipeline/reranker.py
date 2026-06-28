"""
Cross-Encoder Reranker Module
===============================
Re-ranks retrieval results using a cross-encoder model to improve
relevance ordering. Cross-encoders jointly encode (query, document)
pairs, providing more accurate relevance scores than bi-encoder
similarity at the cost of higher latency.

Default Model: cross-encoder/ms-marco-MiniLM-L-6-v2
(A lightweight, fast cross-encoder fine-tuned on MS MARCO).
"""

import logging
from typing import List, Dict, Optional

from sentence_transformers import CrossEncoder

import config

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieval results using a cross-encoder model.

    The cross-encoder scores (query, chunk_text) pairs jointly,
    providing more nuanced relevance judgments than initial retrieval
    scores. Results are re-sorted by cross-encoder score.

    Attributes:
        model: A sentence_transformers CrossEncoder model instance.
        top_k: Number of top results to return after reranking.
    """

    def __init__(
        self,
        model_name: str = config.RERANKER_MODEL,
        top_k: int = config.RERANK_TOP_K,
    ):
        """
        Initialize the CrossEncoderReranker.

        Args:
            model_name: Name or path of the cross-encoder model to load.
                        Default from config.RERANKER_MODEL.
            top_k: Number of top results to return after reranking.
                   Default from config.RERANK_TOP_K.
        """
        self.top_k = top_k

        logger.info("Loading cross-encoder model: %s", model_name)
        try:
            self.model = CrossEncoder(model_name)
            logger.info("Cross-encoder model loaded successfully: %s", model_name)
        except Exception as e:
            logger.error("Failed to load cross-encoder model '%s': %s", model_name, e)
            raise

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Rerank retrieval results using the cross-encoder model.

        Process:
        1. Create (query, chunk_text) pairs from the results.
        2. Score each pair using the cross-encoder.
        3. Add 'rerank_score' field to each result.
        4. Sort results by rerank_score in descending order.
        5. Return the top_k highest-scoring results with updated ranks.

        Args:
            query: The search query string.
            results: A list of result dicts, each expected to have a 'chunk'
                     key containing a dict with 'text' or 'content' field.
            top_k: Override for the number of results to return.
                   If None, uses self.top_k.

        Returns:
            A list of reranked result dicts (top_k), each with:
                - All original fields preserved.
                - 'rerank_score' (float): The cross-encoder relevance score.
                - 'rank' (int): Updated 1-based rank after reranking.
                - 'original_rank' (int): The rank before reranking.
        """
        effective_top_k = top_k if top_k is not None else self.top_k

        # Handle edge cases
        if not query or not query.strip():
            logger.warning("Empty query received for reranking, returning empty results.")
            return []

        if not results:
            logger.warning("No results provided for reranking.")
            return []

        logger.debug(
            "Reranking %d results for query: '%s'",
            len(results),
            query[:80],
        )

        # --- Step 1: Create (query, chunk_text) pairs ---
        pairs = []
        valid_indices = []  # Track which results have valid text

        for idx, result in enumerate(results):
            chunk_text = self._extract_text(result)
            if chunk_text:
                pairs.append((query, chunk_text))
                valid_indices.append(idx)
            else:
                logger.warning(
                    "Result at index %d has no extractable text, skipping.", idx
                )

        if not pairs:
            logger.warning("No valid (query, text) pairs for reranking.")
            return results[:effective_top_k]

        # --- Step 2: Score with cross-encoder ---
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error("Cross-encoder prediction failed: %s", e)
            # Return original results on failure
            return results[:effective_top_k]

        # --- Step 3: Attach rerank scores to results ---
        scored_results = []
        for score_idx, result_idx in enumerate(valid_indices):
            result_copy = dict(results[result_idx])  # Shallow copy to avoid mutation
            result_copy["rerank_score"] = float(scores[score_idx])
            result_copy["original_rank"] = result_copy.get("rank", result_idx + 1)
            scored_results.append(result_copy)

        # --- Step 4: Sort by rerank_score (descending) ---
        scored_results.sort(key=lambda r: r["rerank_score"], reverse=True)

        # --- Step 5: Assign new ranks and return top_k ---
        reranked = []
        for new_rank, result in enumerate(scored_results[:effective_top_k], start=1):
            result["rank"] = new_rank
            reranked.append(result)

        logger.info(
            "Reranking complete: %d -> %d results for query: '%s'",
            len(results),
            len(reranked),
            query[:80],
        )

        return reranked

    @staticmethod
    def _extract_text(result: Dict) -> Optional[str]:
        """
        Extract the text content from a result dictionary.

        Tries multiple common key paths to locate the chunk text:
        1. result['chunk']['text']
        2. result['chunk']['content']
        3. result['chunk']['page_content']
        4. result['text']
        5. result['content']
        6. str(result['chunk']) if chunk is a string

        Args:
            result: A result dictionary from retrieval.

        Returns:
            The extracted text string, or None if no text is found.
        """
        chunk = result.get("chunk")

        if isinstance(chunk, dict):
            # Try common text field names
            for key in ("text", "content", "page_content", "passage"):
                if key in chunk and chunk[key]:
                    return str(chunk[key])

        elif isinstance(chunk, str):
            return chunk

        # Fallback: try top-level text fields
        for key in ("text", "content", "page_content"):
            if key in result and result[key]:
                return str(result[key])

        return None
