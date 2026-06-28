"""
BM25 Retriever Module
=====================

This module provides the BM25Retriever class for lexical (sparse) retrieval
in the ACORA pipeline.

It wraps the ``rank_bm25.BM25Okapi`` implementation and adds:
  - Simple whitespace + lowercasing tokenization
  - Configurable BM25 hyper-parameters (k1, b) via ``config.py``
  - Serialisation / deserialisation of the index via pickle
  - Consistent result format shared across all ACORA retrievers

Usage:
    from pipeline.retriever_bm25 import BM25Retriever

    retriever = BM25Retriever()
    retriever.build_index(chunks)
    results = retriever.retrieve("What is ACORA?")
"""

import sys
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Import project config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class BM25Retriever:
    """Lexical retriever based on the BM25 (Okapi) scoring function.

    BM25 is a bag-of-words ranking function that scores documents by term
    frequency, inverse document frequency, and document length normalisation.

    Attributes:
        k1 (float): Term-frequency saturation parameter.
        b (float): Document-length normalisation parameter (0 = no
            normalisation, 1 = full normalisation).
        top_k (int): Default number of results to return.
        index (BM25Okapi | None): The underlying rank_bm25 index.
        chunks (List[Dict]): Chunk dicts corresponding 1-to-1 with index rows.
        tokenized_corpus (List[List[str]]): Tokenized version of the corpus,
            kept for serialisation.
    """

    def __init__(
        self,
        k1: float = config.BM25_K1,
        b: float = config.BM25_B,
        top_k: int = config.BM25_TOP_K,
    ) -> None:
        """Initialise the BM25 retriever.

        Args:
            k1: BM25 k1 parameter (default from config).
            b: BM25 b parameter (default from config).
            top_k: Default number of top results to retrieve.
        """
        self.k1 = k1
        self.b = b
        self.top_k = top_k

        # These are populated by ``build_index`` or ``load_index``
        self.index: Optional[BM25Okapi] = None
        self.chunks: List[Dict] = []
        self.tokenized_corpus: List[List[str]] = []

        logger.info(
            "BM25Retriever initialised  |  k1=%.2f  b=%.2f  top_k=%d",
            self.k1,
            self.b,
            self.top_k,
        )

    # -----------------------------------------------------------------------
    # Tokenization helper
    # -----------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text by lowercasing and splitting on whitespace.

        This intentionally keeps tokenization simple and fast.  More
        sophisticated stemmers / lemmatizers can be swapped in later without
        changing the public API.

        Args:
            text: Input text string.

        Returns:
            List of lowercase tokens.
        """
        return text.lower().split()

    # -----------------------------------------------------------------------
    # Index building
    # -----------------------------------------------------------------------

    def build_index(self, chunks: List[Dict]) -> None:
        """Build a BM25 index from a list of chunk dictionaries.

        Each chunk dict must contain at least a ``text`` key.

        Args:
            chunks: List of chunk dictionaries (as produced by
                ``DocumentProcessor.process_documents``).

        Raises:
            ValueError: If *chunks* is empty.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from an empty chunk list.")

        self.chunks = chunks

        # Tokenize every chunk's text
        self.tokenized_corpus = [
            self._tokenize(chunk["text"]) for chunk in self.chunks
        ]

        # Build BM25Okapi index
        self.index = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

        logger.info(
            "BM25 index built  |  corpus_size=%d  avg_doc_len=%.1f",
            len(self.chunks),
            self.index.avgdl,
        )

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: The search query string.
            top_k: Number of results to return.  Falls back to ``self.top_k``
                if not specified.

        Returns:
            A list of result dictionaries sorted by descending BM25 score::

                {
                    "chunk":     dict,   # the original chunk dictionary
                    "score":     float,  # BM25 score
                    "rank":      int,    # 1-based rank
                    "retriever": "bm25", # identifier for fusion / logging
                }

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self.index is None:
            raise RuntimeError(
                "BM25 index not built. Call build_index() or load_index() first."
            )

        if not query or not query.strip():
            logger.warning("Empty query passed to BM25Retriever.retrieve().")
            return []

        top_k = top_k if top_k is not None else self.top_k

        # Tokenize the query the same way we tokenized the corpus
        query_tokens = self._tokenize(query)

        # Score every document
        scores = self.index.get_scores(query_tokens)

        # Get indices of the top-k highest scores (descending)
        # np.argsort is ascending, so we negate or reverse.
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "chunk": self.chunks[idx],
                    "score": float(scores[idx]),
                    "rank": rank,
                    "retriever": "bm25",
                }
            )

        logger.debug(
            "BM25 retrieval for '%s'  |  top score=%.4f  results=%d",
            query[:80],
            results[0]["score"] if results else 0.0,
            len(results),
        )
        return results

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save_index(self, path: str) -> None:
        """Serialise the BM25 index and associated chunks to disk.

        Both the ``BM25Okapi`` object, the raw chunks list, and the
        tokenized corpus are stored together in a single pickle file.

        Args:
            path: Filesystem path for the output pickle file.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "index": self.index,
            "chunks": self.chunks,
            "tokenized_corpus": self.tokenized_corpus,
            "k1": self.k1,
            "b": self.b,
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("BM25 index saved to %s", save_path)

    def load_index(self, path: str) -> None:
        """Load a previously saved BM25 index from disk.

        Args:
            path: Path to the pickle file written by ``save_index``.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {load_path}")

        with open(load_path, "rb") as f:
            payload = pickle.load(f)

        self.index = payload["index"]
        self.chunks = payload["chunks"]
        self.tokenized_corpus = payload["tokenized_corpus"]
        self.k1 = payload.get("k1", self.k1)
        self.b = payload.get("b", self.b)

        logger.info(
            "BM25 index loaded from %s  |  corpus_size=%d",
            load_path,
            len(self.chunks),
        )
