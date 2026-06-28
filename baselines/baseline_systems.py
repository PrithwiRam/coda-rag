"""
Baseline Systems Module
=======================

Provides four baseline RAG systems for comparison with ACORA:

  1. **BM25OnlyBaseline**: Lexical retrieval only (BM25), no reranking, no ACOA.
  2. **DenseOnlyBaseline**: Dense retrieval only, no reranking, no ACOA.
  3. **StaticHybridBaseline**: BM25 + Dense with RRF fusion and reranking,
     but no adaptive routing and no ACOA context optimization.
  4. **StandardRAGBaseline**: LangChain-style default RAG with FAISS vector
     store and default retrieval.

All baselines share a common ``BaselineSystem`` interface so they can be
evaluated with the same harness.

Usage:
    from baselines.baseline_systems import BM25OnlyBaseline, DenseOnlyBaseline

    baseline = BM25OnlyBaseline(
        bm25_retriever=bm25,
        generator=gen,
    )
    result = baseline.process_query("What is the GDP of India?")
"""

import sys
import time
import logging
from abc import ABC, abstractmethod
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


# ===========================================================================
# Base class
# ===========================================================================

class BaselineSystem(ABC):
    """Abstract base class for all baseline RAG systems.

    Every baseline must implement ``process_query()`` and return a result
    dictionary with a consistent schema for fair comparison.

    Attributes:
        name (str): Human-readable name of the baseline system.
    """

    def __init__(self, name: str) -> None:
        """Initialise the baseline system.

        Args:
            name: Human-readable name for this baseline.
        """
        self.name = name
        logger.info("Baseline '%s' initialised.", self.name)

    @abstractmethod
    def process_query(self, query: str) -> Dict:
        """Process a single query through this baseline system.

        Args:
            query: The user's question.

        Returns:
            Dictionary with the following keys::

                {
                    "answer":           str,         # generated answer
                    "retrieved_chunks": List[Dict],  # chunks used for context
                    "latency_ms":       float,       # end-to-end latency
                    "retrieval_calls":  int,          # number of retrieval calls made
                    "system_name":      str,          # name of this baseline
                }
        """
        ...

    def _build_context(self, chunks: List[Dict]) -> str:
        """Build a context string from a list of chunk dictionaries.

        Args:
            chunks: List of chunk dictionaries with a ``text`` key.

        Returns:
            Concatenated context string.
        """
        separator = "\n\n---\n\n"
        texts = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if text:
                texts.append(text)
        return separator.join(texts)


# ===========================================================================
# 1. BM25-Only Baseline
# ===========================================================================

class BM25OnlyBaseline(BaselineSystem):
    """Baseline using BM25 lexical retrieval only.

    No reranking, no ACOA context optimization. Retrieved chunks are sent
    directly to the generator.

    Args:
        bm25_retriever: An instance of ``BM25Retriever``.
        generator: An instance of ``LLMGenerator``.
        top_k: Number of chunks to retrieve. Defaults to 10.
    """

    def __init__(
        self,
        bm25_retriever,
        generator,
        top_k: int = 10,
    ) -> None:
        super().__init__(name="BM25-Only")
        self.bm25_retriever = bm25_retriever
        self.generator = generator
        self.top_k = top_k

    def process_query(self, query: str) -> Dict:
        """Process a query using BM25 retrieval only.

        Args:
            query: The user's question.

        Returns:
            Result dictionary (see ``BaselineSystem.process_query``).
        """
        start = time.perf_counter()

        # Retrieve with BM25
        try:
            results = self.bm25_retriever.retrieve(query, top_k=self.top_k)
        except Exception as e:
            logger.error("BM25 retrieval failed: %s", e)
            results = []

        # Extract chunk data
        chunks = [r.get("chunk", r) for r in results]

        # Build context and generate
        context = self._build_context(chunks)
        gen_result = self.generator.generate(query=query, context=context)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": gen_result.get("answer", ""),
            "retrieved_chunks": chunks,
            "latency_ms": round(elapsed_ms, 2),
            "retrieval_calls": 1,
            "system_name": self.name,
        }


# ===========================================================================
# 2. Dense-Only Baseline
# ===========================================================================

class DenseOnlyBaseline(BaselineSystem):
    """Baseline using dense (embedding) retrieval only.

    No reranking, no ACOA context optimization.

    Args:
        dense_retriever: An instance of ``DenseRetriever``.
        generator: An instance of ``LLMGenerator``.
        top_k: Number of chunks to retrieve. Defaults to 10.
    """

    def __init__(
        self,
        dense_retriever,
        generator,
        top_k: int = 10,
    ) -> None:
        super().__init__(name="Dense-Only")
        self.dense_retriever = dense_retriever
        self.generator = generator
        self.top_k = top_k

    def process_query(self, query: str) -> Dict:
        """Process a query using dense retrieval only.

        Args:
            query: The user's question.

        Returns:
            Result dictionary (see ``BaselineSystem.process_query``).
        """
        start = time.perf_counter()

        # Retrieve with dense embeddings
        try:
            results = self.dense_retriever.retrieve(query, top_k=self.top_k)
        except Exception as e:
            logger.error("Dense retrieval failed: %s", e)
            results = []

        # Extract chunk data
        chunks = [r.get("chunk", r) for r in results]

        # Build context and generate
        context = self._build_context(chunks)
        gen_result = self.generator.generate(query=query, context=context)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": gen_result.get("answer", ""),
            "retrieved_chunks": chunks,
            "latency_ms": round(elapsed_ms, 2),
            "retrieval_calls": 1,
            "system_name": self.name,
        }


# ===========================================================================
# 3. Static Hybrid Baseline
# ===========================================================================

class StaticHybridBaseline(BaselineSystem):
    """Baseline using static hybrid retrieval (BM25 + Dense with RRF).

    Always runs both retrievers and fuses results. Includes reranking but
    does NOT use adaptive routing or ACOA context optimization.

    Args:
        bm25_retriever: An instance of ``BM25Retriever``.
        dense_retriever: An instance of ``DenseRetriever``.
        reranker: An instance of ``CrossEncoderReranker``.
        generator: An instance of ``LLMGenerator``.
        top_k: Number of chunks to retrieve per retriever. Defaults to 10.
    """

    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        reranker,
        generator,
        top_k: int = 10,
    ) -> None:
        super().__init__(name="Static-Hybrid")
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k

    def process_query(self, query: str) -> Dict:
        """Process a query using static hybrid retrieval with reranking.

        Args:
            query: The user's question.

        Returns:
            Result dictionary (see ``BaselineSystem.process_query``).
        """
        start = time.perf_counter()
        retrieval_calls = 0

        # Retrieve with BM25
        try:
            bm25_results = self.bm25_retriever.retrieve(query, top_k=self.top_k)
            retrieval_calls += 1
        except Exception as e:
            logger.error("BM25 retrieval failed in hybrid baseline: %s", e)
            bm25_results = []

        # Retrieve with dense
        try:
            dense_results = self.dense_retriever.retrieve(query, top_k=self.top_k)
            retrieval_calls += 1
        except Exception as e:
            logger.error("Dense retrieval failed in hybrid baseline: %s", e)
            dense_results = []

        # Fuse with Reciprocal Rank Fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(
            bm25_results, dense_results
        )

        # Rerank
        try:
            reranked = self.reranker.rerank(query, fused_results[:self.top_k])
        except Exception as e:
            logger.error("Reranking failed in hybrid baseline: %s", e)
            reranked = fused_results[:self.top_k]

        # Extract chunks
        chunks = [r.get("chunk", r) for r in reranked]

        # Build context and generate
        context = self._build_context(chunks)
        gen_result = self.generator.generate(query=query, context=context)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": gen_result.get("answer", ""),
            "retrieved_chunks": chunks,
            "latency_ms": round(elapsed_ms, 2),
            "retrieval_calls": retrieval_calls,
            "system_name": self.name,
        }

    def _reciprocal_rank_fusion(
        self,
        results_a: List[Dict],
        results_b: List[Dict],
        k: int = config.RRF_K,
    ) -> List[Dict]:
        """Fuse two ranked result lists using Reciprocal Rank Fusion.

        RRF_score(d) = Σ 1 / (k + rank(d))

        Args:
            results_a: First ranked list.
            results_b: Second ranked list.
            k: RRF constant (default from config).

        Returns:
            Fused and re-sorted result list.
        """
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, Dict] = {}

        for rank, result in enumerate(results_a, start=1):
            chunk = result.get("chunk", result)
            chunk_id = chunk.get("chunk_id", str(rank))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            result_map[chunk_id] = result

        for rank, result in enumerate(results_b, start=1):
            chunk = result.get("chunk", result)
            chunk_id = chunk.get("chunk_id", str(rank + len(results_a)))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in result_map:
                result_map[chunk_id] = result

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        fused = []
        for chunk_id in sorted_ids:
            result = result_map[chunk_id]
            result["rrf_score"] = rrf_scores[chunk_id]
            fused.append(result)

        return fused


# ===========================================================================
# 4. Standard RAG Baseline
# ===========================================================================

class StandardRAGBaseline(BaselineSystem):
    """Baseline using a standard LangChain-style RAG pipeline.

    Uses FAISS vector store with default retrieval (no adaptive routing,
    no reranking, no ACOA).

    Args:
        documents: List of document chunk dictionaries to index.
        generator: An instance of ``LLMGenerator``.
        embedding_model: Name of the HuggingFace embedding model.
        top_k: Number of chunks to retrieve. Defaults to 10.
    """

    def __init__(
        self,
        documents: List[Dict],
        generator,
        embedding_model: str = config.EMBEDDING_MODEL,
        top_k: int = 10,
    ) -> None:
        super().__init__(name="Standard-RAG")
        self.generator = generator
        self.top_k = top_k
        self.vectorstore = None
        self.embedding_model_name = embedding_model

        # Build the FAISS index
        self._build_index(documents)

    def _build_index(self, documents: List[Dict]) -> None:
        """Build a FAISS vector store from document chunks.

        Args:
            documents: List of chunk dictionaries with 'text' key.
        """
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name
            )

            texts = [doc.get("text", "") for doc in documents if doc.get("text")]
            metadatas = [
                {"chunk_id": doc.get("chunk_id", str(i)), "doc_id": doc.get("doc_id", "")}
                for i, doc in enumerate(documents) if doc.get("text")
            ]

            if texts:
                self.vectorstore = FAISS.from_texts(
                    texts=texts,
                    embedding=embeddings,
                    metadatas=metadatas,
                )
                logger.info(
                    "Standard-RAG FAISS index built with %d documents.", len(texts)
                )
            else:
                logger.warning("No texts provided for Standard-RAG index.")

        except ImportError as e:
            logger.error(
                "Failed to import FAISS/HuggingFace for Standard-RAG: %s. "
                "Install: pip install langchain-community faiss-cpu", e
            )
        except Exception as e:
            logger.error("Failed to build Standard-RAG index: %s", e)

    def process_query(self, query: str) -> Dict:
        """Process a query using standard FAISS-based RAG.

        Args:
            query: The user's question.

        Returns:
            Result dictionary (see ``BaselineSystem.process_query``).
        """
        start = time.perf_counter()

        chunks = []
        retrieval_calls = 0

        if self.vectorstore is not None:
            try:
                docs = self.vectorstore.similarity_search(
                    query, k=self.top_k
                )
                retrieval_calls = 1
                chunks = [
                    {
                        "text": doc.page_content,
                        "chunk_id": doc.metadata.get("chunk_id", ""),
                        "doc_id": doc.metadata.get("doc_id", ""),
                    }
                    for doc in docs
                ]
            except Exception as e:
                logger.error("Standard-RAG retrieval failed: %s", e)
        else:
            logger.warning(
                "Standard-RAG vectorstore not initialised. Returning empty."
            )

        # Build context and generate
        context = self._build_context(chunks)
        gen_result = self.generator.generate(query=query, context=context)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": gen_result.get("answer", ""),
            "retrieved_chunks": chunks,
            "latency_ms": round(elapsed_ms, 2),
            "retrieval_calls": retrieval_calls,
            "system_name": self.name,
        }
