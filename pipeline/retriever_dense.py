"""
Dense Retriever Module
======================

This module provides the DenseRetriever class for semantic (dense) retrieval
in the ACORA pipeline.

It combines:
  - **SentenceTransformer** embeddings for encoding queries and passages
  - **FAISS IndexFlatIP** (inner-product) for fast nearest-neighbour search

Because all embeddings are L2-normalised before insertion, inner-product
scores are equivalent to cosine similarity, which makes scores directly
comparable and interpretable.

Persistence is split into two files:
  - A FAISS binary index (``*.faiss``)
  - A pickle file for the chunk metadata list (``*.chunks.pkl``)

Usage:
    from pipeline.retriever_dense import DenseRetriever

    retriever = DenseRetriever()
    retriever.build_index(chunks)
    results = retriever.retrieve("What is ACORA?")
"""

import sys
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Import project config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class DenseRetriever:
    """Semantic retriever backed by SentenceTransformer + FAISS.

    Embeddings are L2-normalised so that FAISS inner-product search returns
    cosine-similarity scores in the range [−1, 1].

    Attributes:
        model_name (str): HuggingFace model identifier for
            SentenceTransformer.
        top_k (int): Default number of results to return.
        model (SentenceTransformer): The loaded embedding model.
        index (faiss.IndexFlatIP | None): The FAISS inner-product index.
        chunks (List[Dict]): Chunk dicts aligned 1-to-1 with index rows.
        embedding_dim (int): Dimensionality of the embedding vectors.
    """

    _model_cache = {}

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        top_k: int = config.DENSE_TOP_K,
    ) -> None:
        """Initialise the DenseRetriever.

        The SentenceTransformer model is loaded eagerly so that encoding
        latency is incurred once (at init time) rather than on the first
        query.

        Args:
            model_name: HuggingFace model identifier (default from config).
            top_k: Default number of top results to retrieve.
        """
        self.model_name = model_name
        self.top_k = top_k

        # Load the embedding model
        if self.model_name not in DenseRetriever._model_cache:
            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            DenseRetriever._model_cache[self.model_name] = SentenceTransformer(self.model_name)
        self.model = DenseRetriever._model_cache[self.model_name]
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()

        # These are populated by ``build_index`` or ``load_index``
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict] = []

        logger.info(
            "DenseRetriever initialised  |  model=%s  dim=%d  top_k=%d",
            self.model_name,
            self.embedding_dim,
            self.top_k,
        )

    # -----------------------------------------------------------------------
    # Index building
    # -----------------------------------------------------------------------

    def build_index(self, chunks: List[Dict]) -> None:
        """Build a FAISS inner-product index from chunk embeddings.

        Steps:
            1. Extract text from each chunk dictionary.
            2. Encode all texts into dense vectors via SentenceTransformer.
            3. L2-normalise the vectors (so IP ≡ cosine similarity).
            4. Add the normalised vectors to a ``faiss.IndexFlatIP`` index.

        Args:
            chunks: List of chunk dictionaries (must contain a ``text`` key).

        Raises:
            ValueError: If *chunks* is empty.
        """
        if not chunks:
            raise ValueError(
                "Cannot build dense index from an empty chunk list."
            )

        self.chunks = chunks
        texts = [chunk["text"] for chunk in self.chunks]

        # Encode all chunk texts → numpy float32 matrix
        logger.info("Encoding %d chunks with %s …", len(texts), self.model_name)
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we normalise explicitly below
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # L2-normalise so inner product == cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS IndexFlatIP (exact search via inner product)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        logger.info(
            "FAISS index built  |  vectors=%d  dim=%d",
            self.index.ntotal,
            self.embedding_dim,
        )

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve the most semantically similar chunks for a query.

        Args:
            query: The search query string.
            top_k: Number of results to return.  Falls back to ``self.top_k``
                if not specified.

        Returns:
            A list of result dictionaries sorted by descending similarity::

                {
                    "chunk":     dict,    # the original chunk dictionary
                    "score":     float,   # cosine similarity (IP after L2 norm)
                    "rank":      int,     # 1-based rank
                    "retriever": "dense", # identifier for fusion / logging
                }

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self.index is None:
            raise RuntimeError(
                "Dense index not built. Call build_index() or load_index() "
                "first."
            )

        if not query or not query.strip():
            logger.warning("Empty query passed to DenseRetriever.retrieve().")
            return []

        top_k = top_k if top_k is not None else self.top_k

        # Clamp top_k to the number of indexed vectors
        top_k = min(top_k, self.index.ntotal)

        # Encode and normalise the query vector
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # Search the FAISS index
        scores, indices = self.index.search(query_embedding, top_k)

        # ``scores`` and ``indices`` have shape (1, top_k) — unwrap.
        scores = scores[0]
        indices = indices[0]

        results: List[Dict] = []
        for rank, (idx, score) in enumerate(
            zip(indices, scores), start=1
        ):
            # FAISS may return -1 index when fewer results are available
            if idx < 0:
                continue
            results.append(
                {
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "rank": rank,
                    "retriever": "dense",
                }
            )

        logger.debug(
            "Dense retrieval for '%s'  |  top score=%.4f  results=%d",
            query[:80],
            results[0]["score"] if results else 0.0,
            len(results),
        )
        return results

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save_index(self, path: str) -> None:
        """Save the FAISS index and chunk metadata to disk.

        Two files are written:
          - ``<path>.faiss``  – the FAISS binary index
          - ``<path>.chunks.pkl`` – pickled chunk list

        Args:
            path: Base path (without extension).  Extensions are appended
                automatically.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")

        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss_path = str(base_path) + ".faiss"
        faiss.write_index(self.index, faiss_path)

        # Save chunks metadata
        chunks_path = str(base_path) + ".chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "Dense index saved  |  faiss=%s  chunks=%s",
            faiss_path,
            chunks_path,
        )

    def load_index(self, path: str) -> None:
        """Load a previously saved FAISS index and chunk metadata.

        Expects two files at the given base path:
          - ``<path>.faiss``
          - ``<path>.chunks.pkl``

        Args:
            path: Base path (without extension) used in ``save_index``.

        Raises:
            FileNotFoundError: If either the FAISS file or the chunks file
                is missing.
        """
        faiss_path = str(Path(path)) + ".faiss"
        chunks_path = str(Path(path)) + ".chunks.pkl"

        if not Path(faiss_path).exists():
            raise FileNotFoundError(
                f"FAISS index file not found: {faiss_path}"
            )
        if not Path(chunks_path).exists():
            raise FileNotFoundError(
                f"Chunks metadata file not found: {chunks_path}"
            )

        # Load FAISS index
        self.index = faiss.read_index(faiss_path)

        # Load chunks
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(
            "Dense index loaded  |  vectors=%d  chunks=%d  source=%s",
            self.index.ntotal,
            len(self.chunks),
            path,
        )
