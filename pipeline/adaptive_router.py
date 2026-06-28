"""
Adaptive Router Module
========================
Routes incoming queries to the most appropriate retrieval strategy
based on query classification results.

Routing Logic:
- LEXICAL  -> BM25 retriever  (keyword-based matching)
- SEMANTIC -> Dense retriever  (embedding-based similarity)
- AMBIGUOUS -> Hybrid retriever (RRF fusion of both)

This module acts as the central dispatcher in the ACORA adaptive
retrieval pipeline, ensuring each query type is handled by the
retrieval method best suited to its characteristics.
"""

import logging
from typing import List, Dict, Tuple, Optional

import config

logger = logging.getLogger(__name__)


class AdaptiveRouter:
    """
    Routes queries to the optimal retriever based on query classification.

    The router uses a QueryClassifier to determine the query type, then
    dispatches to the corresponding retriever:
    - LEXICAL queries  -> BM25 retriever
    - SEMANTIC queries -> Dense retriever
    - AMBIGUOUS queries -> Hybrid retriever (RRF fusion)

    Attributes:
        bm25_retriever: BM25/lexical retriever instance.
        dense_retriever: Dense/semantic retriever instance.
        hybrid_retriever: Hybrid retriever (RRF fusion) instance.
        query_classifier: QueryClassifier instance for query type detection.
    """

    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        hybrid_retriever,
        query_classifier,
    ):
        """
        Initialize the AdaptiveRouter.

        Args:
            bm25_retriever: A BM25-based lexical retriever with .retrieve(query, top_k).
            dense_retriever: A dense/semantic retriever with .retrieve(query, top_k).
            hybrid_retriever: A HybridRetriever instance with .retrieve(query, top_k).
            query_classifier: A QueryClassifier instance with .classify(query).
        """
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.hybrid_retriever = hybrid_retriever
        self.query_classifier = query_classifier

        # Map query types to retriever instances
        self._retriever_map = {
            config.QUERY_TYPE_LEXICAL: self.bm25_retriever,
            config.QUERY_TYPE_SEMANTIC: self.dense_retriever,
            config.QUERY_TYPE_AMBIGUOUS: self.hybrid_retriever,
        }

        logger.info("AdaptiveRouter initialized with 3 retriever routes.")

    def route(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Classify a query and route it to the appropriate retriever.

        The routing process:
        1. Classify the query using the QueryClassifier.
        2. Select the retriever based on the classification label.
        3. Retrieve results using the selected retriever.
        4. Return results along with classification metadata.

        Args:
            query: The input query string.
            top_k: Optional override for the number of results to return.
                   If None, the selected retriever uses its default top_k.

        Returns:
            A tuple of (results, classification_info):
            - results: A list of result dicts from the selected retriever.
            - classification_info: A dict containing:
                - 'label' (str): Query type (LEXICAL, SEMANTIC, AMBIGUOUS).
                - 'confidence' (float): Classification confidence.
                - 'method' (str): Classification method used.
                - 'features' (dict): Extracted query features.
                - 'routed_to' (str): Name of the retriever used.
        """
        # Handle empty query
        if not query or not query.strip():
            logger.warning("Empty query received in router, returning empty results.")
            return [], {
                "label": config.QUERY_TYPE_AMBIGUOUS,
                "confidence": 0.0,
                "method": "fallback",
                "features": {},
                "routed_to": "none",
            }

        # Step 1: Classify the query
        classification = self.query_classifier.classify(query)
        label = classification["label"]
        confidence = classification["confidence"]
        method = classification["method"]

        # Step 2: Select the appropriate retriever
        retriever = self._retriever_map.get(label)

        if retriever is None:
            # Unexpected label — fall back to hybrid
            logger.warning(
                "Unknown classification label '%s' for query '%s'. "
                "Falling back to hybrid retriever.",
                label,
                query[:60],
            )
            retriever = self.hybrid_retriever
            label = config.QUERY_TYPE_AMBIGUOUS

        # Determine retriever name for logging
        retriever_name = self._get_retriever_name(label)

        logger.info(
            "Routing query '%s' -> %s (label=%s, confidence=%.2f, method=%s)",
            query[:60],
            retriever_name,
            label,
            confidence,
            method,
        )

        # Step 3: Retrieve results
        try:
            if top_k is not None:
                results = retriever.retrieve(query, top_k=top_k)
            else:
                results = retriever.retrieve(query)
        except Exception as e:
            logger.error(
                "Retrieval failed for query '%s' routed to %s: %s",
                query[:60],
                retriever_name,
                e,
            )
            results = []

        logger.info(
            "Router returned %d results from %s for query: '%s'",
            len(results),
            retriever_name,
            query[:60],
        )

        # Step 4: Build classification info with routing metadata
        classification_info = {
            "label": label,
            "confidence": confidence,
            "method": method,
            "features": classification.get("features", {}),
            "routed_to": retriever_name,
        }

        return results, classification_info

    @staticmethod
    def _get_retriever_name(label: str) -> str:
        """
        Map a classification label to a human-readable retriever name.

        Args:
            label: The classification label string.

        Returns:
            A descriptive name for the retriever.
        """
        name_map = {
            config.QUERY_TYPE_LEXICAL: "bm25_retriever",
            config.QUERY_TYPE_SEMANTIC: "dense_retriever",
            config.QUERY_TYPE_AMBIGUOUS: "hybrid_retriever",
        }
        return name_map.get(label, "unknown_retriever")
