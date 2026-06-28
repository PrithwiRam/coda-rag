"""
Hybrid Retriever Module
========================
Combines BM25 (lexical) and dense (semantic) retrieval results using
Reciprocal Rank Fusion (RRF) to produce a single ranked list.

RRF Score:
    RRF_score(d) = Σ 1 / (k + rank_i(d))
    where k is a constant (default 60) and rank_i(d) is the rank of
    document d in the i-th retrieval system.
"""

import logging
from typing import List, Dict, Optional
from collections import defaultdict

import config

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever that fuses results from BM25 and dense retrievers
    using Reciprocal Rank Fusion (RRF).

    Attributes:
        bm25_retriever: Lexical retriever instance (must have .retrieve(query, top_k)).
        dense_retriever: Dense/semantic retriever instance (must have .retrieve(query, top_k)).
        rrf_k: RRF smoothing constant (prevents high scores for top-ranked docs).
        top_k: Number of results to return after fusion.
    """

    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        rrf_k: int = config.RRF_K,
        top_k: int = config.HYBRID_TOP_K,
    ):
        """
        Initialize the HybridRetriever.

        Args:
            bm25_retriever: A BM25-based lexical retriever with a .retrieve() method.
            dense_retriever: A dense/semantic retriever with a .retrieve() method.
            rrf_k: RRF constant k. Higher values reduce the influence of rank position.
                   Default from config.RRF_K (60).
            top_k: Maximum number of fused results to return.
                   Default from config.HYBRID_TOP_K (20).
        """
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.top_k = top_k

        logger.info(
            "HybridRetriever initialized with rrf_k=%d, top_k=%d",
            self.rrf_k,
            self.top_k,
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve and fuse results from both BM25 and dense retrievers using RRF.

        The method:
        1. Queries both BM25 and dense retrievers independently.
        2. Computes RRF scores for each document across both result sets.
        3. Deduplicates by chunk_id (summing RRF contributions).
        4. Sorts by fused RRF score in descending order.
        5. Returns the top_k results.

        Args:
            query: The search query string.
            top_k: Override for the number of results to return.
                   If None, uses self.top_k.

        Returns:
            A list of dicts, each containing:
                - 'chunk': The original chunk dictionary.
                - 'score': The fused RRF score.
                - 'rank': 1-based rank in the fused result list.
                - 'retriever': Always 'hybrid'.
        """
        effective_top_k = top_k if top_k is not None else self.top_k

        # Handle empty query
        if not query or not query.strip():
            logger.warning("Empty query received, returning empty results.")
            return []

        # --- Step 1: Retrieve from both systems ---
        try:
            bm25_results = self.bm25_retriever.retrieve(query, top_k=effective_top_k)
        except Exception as e:
            logger.error("BM25 retrieval failed: %s", e)
            bm25_results = []

        try:
            dense_results = self.dense_retriever.retrieve(query, top_k=effective_top_k)
        except Exception as e:
            logger.error("Dense retrieval failed: %s", e)
            dense_results = []

        # If both retrievers failed, return empty
        if not bm25_results and not dense_results:
            logger.warning("Both retrievers returned empty results for query: '%s'", query)
            return []

        logger.debug(
            "Retrieved %d BM25 and %d dense results for query: '%s'",
            len(bm25_results),
            len(dense_results),
            query[:80],
        )

        # --- Step 2: Compute RRF scores ---
        # rrf_scores maps chunk_id -> cumulative RRF score
        rrf_scores: Dict[str, float] = defaultdict(float)
        # chunk_store maps chunk_id -> chunk dict (for deduplication)
        chunk_store: Dict[str, dict] = {}

        # Process BM25 results
        for rank_idx, result in enumerate(bm25_results):
            chunk = result.get("chunk", result)
            chunk_id = self._get_chunk_id(chunk, rank_idx, "bm25")

            # RRF contribution: 1 / (k + rank), rank is 1-based
            rrf_score = 1.0 / (self.rrf_k + (rank_idx + 1))
            rrf_scores[chunk_id] += rrf_score

            # Store chunk data (first occurrence wins)
            if chunk_id not in chunk_store:
                chunk_store[chunk_id] = chunk

        # Process dense results
        for rank_idx, result in enumerate(dense_results):
            chunk = result.get("chunk", result)
            chunk_id = self._get_chunk_id(chunk, rank_idx, "dense")

            # RRF contribution: 1 / (k + rank), rank is 1-based
            rrf_score = 1.0 / (self.rrf_k + (rank_idx + 1))
            rrf_scores[chunk_id] += rrf_score

            # Store chunk data (first occurrence wins)
            if chunk_id not in chunk_store:
                chunk_store[chunk_id] = chunk

        # --- Step 3: Sort by fused RRF score (descending) ---
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        # --- Step 4: Build final results with ranks ---
        fused_results = []
        for rank, chunk_id in enumerate(sorted_chunk_ids[:effective_top_k], start=1):
            fused_results.append(
                {
                    "chunk": chunk_store[chunk_id],
                    "score": rrf_scores[chunk_id],
                    "rank": rank,
                    "retriever": "hybrid",
                }
            )

        logger.info(
            "Hybrid retrieval produced %d fused results (from %d unique chunks) for query: '%s'",
            len(fused_results),
            len(rrf_scores),
            query[:80],
        )

        return fused_results

    @staticmethod
    def _get_chunk_id(chunk: dict, fallback_idx: int, source: str) -> str:
        """
        Extract a unique chunk identifier for deduplication.

        Tries 'chunk_id' key first, then 'id', then falls back to a
        composite key based on source and index.

        Args:
            chunk: The chunk dictionary.
            fallback_idx: Index to use if no ID field is found.
            source: Source retriever name (used in fallback ID).

        Returns:
            A string identifier for the chunk.
        """
        if isinstance(chunk, dict):
            if "chunk_id" in chunk:
                return str(chunk["chunk_id"])
            if "id" in chunk:
                return str(chunk["id"])
        # Fallback: use source + index (won't deduplicate across retrievers)
        return f"{source}_{fallback_idx}"
