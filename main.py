"""
ACORA Main Pipeline
===================

End-to-end pipeline for the Adaptive Context-Optimized Retrieval Architecture.

Orchestrates all components:
  1. **Query Classification** – Determines query type (LEXICAL / SEMANTIC / AMBIGUOUS).
  2. **Adaptive Routing** – Selects the optimal retrieval strategy.
  3. **Retrieval** – Fetches candidate chunks via BM25, Dense, or Hybrid.
  4. **Reranking** – Refines result ordering with a cross-encoder.
  5. **Confidence Gate** – Validates retrieval quality; triggers fallback if needed.
  6. **ACOA** – Selects a non-redundant, dense subset within token budget.
  7. **LLM Generation** – Produces the final answer.

Usage:
    from main import ACORAPipeline

    pipeline = ACORAPipeline()
    pipeline.initialize(documents)
    result = pipeline.query("What is the GDP of India in 2024?")
    print(result['answer'])
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Pipeline component imports
# ---------------------------------------------------------------------------
from pipeline.document_processor import DocumentProcessor
from pipeline.confidence_gate import ConfidenceGate
from pipeline.acoa import ACOA
from pipeline.generator import LLMGenerator

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ACORAPipeline:
    """End-to-end ACORA pipeline.

    Integrates all components from document processing through answer
    generation.  Must be initialised with documents before querying.

    Attributes:
        is_initialized (bool): Whether the pipeline has been initialised
            with documents and indexes are built.
    """

    def __init__(self) -> None:
        """Initialise all pipeline components.

        Components that require data (retrievers) will be fully set up
        during ``initialize()``.
        """
        logger.info("=" * 60)
        logger.info("Initialising ACORA Pipeline components...")
        logger.info("=" * 60)

        # Document processing
        self.document_processor = DocumentProcessor()

        # Retrievers – will be initialised with data in initialize()
        self.bm25_retriever = None
        self.dense_retriever = None
        self.hybrid_retriever = None

        # Query analysis
        self.query_feature_extractor = None
        self.query_classifier = None
        self.adaptive_router = None

        # Reranker
        self.reranker = None

        # Confidence gate
        self.confidence_gate = ConfidenceGate()

        # ACOA context optimizer
        self.acoa = ACOA()

        # LLM generator
        self.generator = None

        # Pipeline state
        self.is_initialized = False
        self.chunks = []

        # Lazy-load components that may have heavy dependencies
        self._init_components()

        logger.info("ACORA Pipeline components created (not yet initialised with data).")

    def _init_components(self) -> None:
        """Lazy-initialise components with potentially heavy imports.

        Components that fail to import are set to None and the pipeline
        will use fallback behaviour.
        """
        # --- Query Feature Extractor ---
        try:
            from pipeline.query_feature_extractor import QueryFeatureExtractor
            self.query_feature_extractor = QueryFeatureExtractor()
        except (ImportError, Exception) as e:
            logger.warning("QueryFeatureExtractor unavailable: %s", e)
            self.query_feature_extractor = None

        # --- Query Classifier ---
        try:
            from pipeline.query_classifier import QueryClassifier
            self.query_classifier = QueryClassifier()
        except (ImportError, Exception) as e:
            logger.warning("QueryClassifier unavailable: %s", e)
            self.query_classifier = None

        # --- Adaptive Router ---
        # Note: AdaptiveRouter is initialised in initialize() after retrievers are built
        self.adaptive_router = None

        # --- Cross-Encoder Reranker ---
        try:
            from pipeline.reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker()
        except (ImportError, Exception) as e:
            logger.warning("CrossEncoderReranker unavailable: %s", e)
            self.reranker = None

        # --- LLM Generator ---
        try:
            self.generator = LLMGenerator()
        except (ImportError, Exception) as e:
            logger.warning("LLMGenerator unavailable: %s", e)
            self.generator = None

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def initialize(self, documents: List[Dict]) -> None:
        """Initialise the pipeline with documents.

        Processes documents into chunks and builds retrieval indexes.

        Args:
            documents: List of document dictionaries, each with at least
                ``text`` (str) and ``doc_id`` (str) keys.
        """
        logger.info("=" * 60)
        logger.info("Initialising ACORA Pipeline with %d document(s)...", len(documents))
        logger.info("=" * 60)

        # --- Step 1: Process documents into chunks ---
        self.chunks = self.document_processor.process_documents(documents)
        logger.info("Document processing complete: %d chunks created.", len(self.chunks))

        # --- Step 2: Build BM25 index ---
        try:
            from pipeline.retriever_bm25 import BM25Retriever
            self.bm25_retriever = BM25Retriever()
            self.bm25_retriever.build_index(self.chunks)
            logger.info("BM25 index built successfully.")
        except (ImportError, Exception) as e:
            logger.warning("BM25Retriever unavailable: %s", e)
            self.bm25_retriever = None

        # --- Step 3: Build Dense index ---
        try:
            from pipeline.retriever_dense import DenseRetriever
            self.dense_retriever = DenseRetriever()
            self.dense_retriever.build_index(self.chunks)
            logger.info("Dense index built successfully.")
        except (ImportError, Exception) as e:
            logger.warning("DenseRetriever unavailable: %s", e)
            self.dense_retriever = None

        # --- Step 4: Set up Hybrid retriever ---
        try:
            from pipeline.retriever_hybrid import HybridRetriever
            if self.bm25_retriever and self.dense_retriever:
                self.hybrid_retriever = HybridRetriever(
                    bm25_retriever=self.bm25_retriever,
                    dense_retriever=self.dense_retriever,
                )
                logger.info("Hybrid retriever initialised.")
        except (ImportError, Exception) as e:
            logger.warning("HybridRetriever unavailable: %s", e)
            self.hybrid_retriever = None

        # --- Step 5: Set up Adaptive Router ---
        try:
            from pipeline.adaptive_router import AdaptiveRouter
            if (self.bm25_retriever and self.dense_retriever and 
                self.hybrid_retriever and self.query_classifier):
                self.adaptive_router = AdaptiveRouter(
                    bm25_retriever=self.bm25_retriever,
                    dense_retriever=self.dense_retriever,
                    hybrid_retriever=self.hybrid_retriever,
                    query_classifier=self.query_classifier,
                )
                logger.info("Adaptive router initialised successfully.")
            else:
                logger.warning("Could not initialize AdaptiveRouter: missing retrievers or classifier.")
                self.adaptive_router = None
        except (ImportError, Exception) as e:
            logger.warning("AdaptiveRouter unavailable: %s", e)
            self.adaptive_router = None

        self.is_initialized = True
        logger.info("ACORA Pipeline initialised and ready for queries.")

    def query(self, query_text: str) -> Dict:
        """Process a single query through the full ACORA pipeline.

        Pipeline stages:
            1. Classify query → determine query type
            2. Route → select retrieval strategy
            3. Retrieve → fetch candidate chunks
            4. Rerank → refine ordering with cross-encoder
            5. Confidence gate → validate quality (fallback if needed)
            6. ACOA → optimize context (select, deduplicate, budget)
            7. Generate → produce answer with LLM

        Args:
            query_text: The user's question.

        Returns:
            Comprehensive result dictionary::

                {
                    "answer":             str,
                    "query":              str,
                    "classification":     dict,
                    "retrieval_results":  list,
                    "reranked_results":   list,
                    "confidence":         dict,
                    "acoa_stats":         dict,
                    "context":            str,
                    "latency_ms":         float,
                    "retrieval_calls":    int,
                }
        """
        if not self.is_initialized:
            return {
                "answer": "[Error] Pipeline not initialised. Call initialize() first.",
                "query": query_text,
                "classification": {},
                "retrieval_results": [],
                "reranked_results": [],
                "confidence": {},
                "acoa_stats": {},
                "context": "",
                "latency_ms": 0.0,
                "retrieval_calls": 0,
            }

        start_time = time.perf_counter()
        retrieval_calls = 0

        # ==================================================================
        # Stages 1-3: Classification, Routing, and Retrieval
        # ==================================================================
        retrieval_results = []
        classification = {
            "label": config.QUERY_TYPE_AMBIGUOUS,
            "query_type": config.QUERY_TYPE_AMBIGUOUS,
            "confidence": 0.0,
            "method": "fallback",
            "features": {},
        }
        retrieval_calls = 0
        top_k = config.HYBRID_TOP_K

        if self.adaptive_router is not None:
            try:
                retrieval_results, route_info = self.adaptive_router.route(query_text, top_k=top_k)
                classification = dict(route_info)
                classification["query_type"] = classification.get("label", config.QUERY_TYPE_AMBIGUOUS)
                
                routed_to = classification.get("routed_to", "hybrid_retriever")
                if routed_to == "hybrid_retriever":
                    retrieval_calls = 2
                else:
                    retrieval_calls = 1
            except Exception as e:
                logger.warning("Adaptive routing and retrieval failed: %s. Using manual fallback.", e)

        # Fallback if router was unavailable or routing failed to retrieve results
        if not retrieval_results:
            if self.query_classifier is not None:
                try:
                    classification = self.query_classifier.classify(query_text)
                    classification["query_type"] = classification.get("label", config.QUERY_TYPE_AMBIGUOUS)
                except Exception as e:
                    logger.warning("Query classification failed: %s. Using AMBIGUOUS fallback.", e)
            
            label = classification.get("label", config.QUERY_TYPE_AMBIGUOUS)
            logger.info("Using manual retrieval path for label: %s", label)
            try:
                if label == config.QUERY_TYPE_LEXICAL and self.bm25_retriever:
                    retrieval_results = self.bm25_retriever.retrieve(query_text, top_k=top_k)
                    retrieval_calls = 1
                elif label == config.QUERY_TYPE_SEMANTIC and self.dense_retriever:
                    retrieval_results = self.dense_retriever.retrieve(query_text, top_k=top_k)
                    retrieval_calls = 1
                elif self.hybrid_retriever:
                    retrieval_results = self.hybrid_retriever.retrieve(query_text, top_k=top_k)
                    retrieval_calls = 2
                elif self.bm25_retriever:
                    retrieval_results = self.bm25_retriever.retrieve(query_text, top_k=top_k)
                    retrieval_calls = 1
                elif self.dense_retriever:
                    retrieval_results = self.dense_retriever.retrieve(query_text, top_k=top_k)
                    retrieval_calls = 1
                else:
                    logger.error("No retrievers available for manual retrieval fallback!")
            except Exception as e:
                logger.error("Manual retrieval fallback failed: %s", e)

        logger.info("Query classified as: %s (confidence=%.2f)",
                    classification.get("label"), classification.get("confidence", 0.0))
        logger.info("Retrieved %d candidates.", len(retrieval_results))

        # ==================================================================
        # Stage 4: Rerank
        # ==================================================================
        reranked_results = retrieval_results

        if self.reranker is not None and retrieval_results:
            try:
                reranked_results = self.reranker.rerank(
                    query_text,
                    retrieval_results,
                    top_k=config.RERANK_TOP_K,
                )
                logger.info("Reranked to top-%d results.", len(reranked_results))
            except Exception as e:
                logger.warning("Reranking failed: %s. Using original order.", e)

        # ==================================================================
        # Stage 5: Confidence Gate
        # ==================================================================
        confidence = self.confidence_gate.evaluate(query_text, reranked_results)

        if not confidence["passed"]:
            logger.info(
                "Confidence gate FAILED (%.3f < %.3f). Triggering fallback...",
                confidence["confidence"], self.confidence_gate.threshold,
            )

            # Fallback: retrieve additional chunks and re-evaluate
            try:
                extra_k = config.FALLBACK_EXTRA_K
                fallback_strategy = "hybrid" if self.hybrid_retriever else "lexical"

                if fallback_strategy == "hybrid" and self.hybrid_retriever:
                    extra_results = self.hybrid_retriever.retrieve(
                        query_text, top_k=top_k + extra_k
                    )
                    retrieval_calls += 2
                elif self.bm25_retriever:
                    extra_results = self.bm25_retriever.retrieve(
                        query_text, top_k=top_k + extra_k
                    )
                    retrieval_calls += 1
                else:
                    extra_results = []

                if extra_results:
                    # Merge and deduplicate
                    seen_ids = {
                        r.get("chunk", {}).get("chunk_id")
                        for r in reranked_results
                    }
                    for r in extra_results:
                        cid = r.get("chunk", {}).get("chunk_id")
                        if cid not in seen_ids:
                            reranked_results.append(r)
                            seen_ids.add(cid)

                    # Re-rerank if reranker is available
                    if self.reranker is not None:
                        reranked_results = self.reranker.rerank(
                            query_text,
                            reranked_results,
                            top_k=config.RERANK_TOP_K + extra_k,
                        )

                    # Re-evaluate confidence
                    confidence = self.confidence_gate.evaluate(
                        query_text, reranked_results
                    )
                    logger.info(
                        "Post-fallback confidence: %.3f (passed=%s)",
                        confidence["confidence"], confidence["passed"],
                    )

            except Exception as e:
                logger.warning("Fallback retrieval failed: %s", e)

        # ==================================================================
        # Stage 6: ACOA – Adaptive Context Optimization
        # ==================================================================
        acoa_result = self.acoa.optimize(reranked_results)
        selected_chunks = acoa_result.get("selected_chunks", [])
        context = self.acoa.build_context(selected_chunks)

        logger.info(
            "ACOA selected %d chunks (%d tokens).",
            acoa_result.get("num_selected", 0),
            acoa_result.get("total_tokens", 0),
        )

        # ==================================================================
        # Stage 7: Generate answer
        # ==================================================================
        answer = "[No generator available]"
        gen_result = {}

        if self.generator is not None:
            try:
                gen_result = self.generator.generate(query=query_text, context=context)
                answer = gen_result.get("answer", "")
            except Exception as e:
                logger.error("LLM generation failed: %s", e)
                answer = f"[Generation Error] {e}"

        # ==================================================================
        # Compile final result
        # ==================================================================
        total_latency = (time.perf_counter() - start_time) * 1000

        result = {
            "answer": answer,
            "query": query_text,
            "classification": classification,
            "retrieval_results": retrieval_results,
            "reranked_results": reranked_results,
            "confidence": confidence,
            "acoa_stats": acoa_result.get("optimization_stats", {}),
            "context": context,
            "latency_ms": round(total_latency, 2),
            "retrieval_calls": retrieval_calls,
        }

        logger.info(
            "Query complete  |  latency=%.0fms  retrieval_calls=%d",
            total_latency, retrieval_calls,
        )

        return result

    def query_batch(self, queries: List[str]) -> List[Dict]:
        """Process multiple queries with a progress bar.

        Args:
            queries: List of query strings.

        Returns:
            List of result dictionaries (one per query).
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(queries, desc="Processing queries", unit="query")
        except ImportError:
            logger.warning("tqdm not installed. Running without progress bar.")
            iterator = queries

        results = []
        for query_text in iterator:
            result = self.query(query_text)
            results.append(result)

        return results


# ===========================================================================
# Demo / main entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  ACORA – Adaptive Context-Optimized Retrieval Architecture")
    print("  Demo Pipeline Run")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 1. Create sample documents (Wikipedia-style paragraphs)
    # ------------------------------------------------------------------
    sample_documents = [
        {
            "doc_id": "doc_india_economy",
            "title": "Indian Economy",
            "text": (
                "India is the world's fifth-largest economy by nominal GDP and "
                "the third-largest by purchasing power parity (PPP). As of 2024, "
                "India's nominal GDP is estimated at approximately $3.94 trillion. "
                "The Indian economy is driven by a large services sector, which "
                "contributes about 54% of GDP, followed by industry at 26% and "
                "agriculture at 20%. Key economic indicators include a GDP growth "
                "rate of 7.2% in FY2024, making India one of the fastest-growing "
                "major economies. The country has significant economic reforms "
                "including GST implementation, digital payments infrastructure "
                "like UPI, and the Make in India manufacturing initiative. "
                "Major cities like Mumbai, Delhi, and Bangalore serve as economic "
                "hubs. India's foreign exchange reserves stood at approximately "
                "$620 billion in 2024. The country is a member of BRICS, G20, "
                "and the World Trade Organization."
            ),
        },
        {
            "doc_id": "doc_neural_networks",
            "title": "Neural Networks and Deep Learning",
            "text": (
                "Neural networks are computational models inspired by the biological "
                "neural networks in the human brain. They consist of layers of "
                "interconnected nodes (neurons) that process information using "
                "connectionist approaches. Deep learning, a subset of machine "
                "learning, uses neural networks with multiple hidden layers to "
                "learn hierarchical representations of data. The learning process "
                "involves forward propagation, where input data flows through "
                "the network to produce predictions, and backpropagation, where "
                "gradients of the loss function are computed and used to update "
                "weights via gradient descent optimization. Key architectures "
                "include Convolutional Neural Networks (CNNs) for image processing, "
                "Recurrent Neural Networks (RNNs) for sequential data, and "
                "Transformers for natural language processing. The attention "
                "mechanism, introduced in the 'Attention Is All You Need' paper "
                "by Vaswani et al. (2017), revolutionized NLP by allowing models "
                "to weigh the importance of different parts of the input. Modern "
                "large language models like GPT-4 and LLaMA use transformer "
                "architectures with billions of parameters trained on massive "
                "text corpora."
            ),
        },
        {
            "doc_id": "doc_climate_agriculture",
            "title": "Climate Change and Agriculture",
            "text": (
                "Climate change poses significant threats to global agriculture "
                "and food security. Rising temperatures affect crop yields, with "
                "studies showing that for every 1°C increase in global mean "
                "temperature, wheat yields decline by approximately 6%, rice "
                "yields by 3.2%, and maize yields by 7.4%. Changes in precipitation "
                "patterns lead to both droughts and floods, disrupting growing "
                "seasons and damaging crops. The increased frequency of extreme "
                "weather events, including heat waves, hurricanes, and unseasonal "
                "frost, causes direct crop losses and infrastructure damage. "
                "Climate change also affects agriculture indirectly through "
                "changes in pest and disease patterns, soil degradation, and "
                "water availability. Adaptation strategies include developing "
                "drought-resistant crop varieties, implementing precision "
                "agriculture techniques, improving irrigation efficiency, and "
                "diversifying crop portfolios. Carbon sequestration through "
                "regenerative farming practices can help mitigate climate change "
                "while improving soil health. The IPCC estimates that without "
                "adaptation, global crop production could decline by 2-6% per "
                "decade due to climate change."
            ),
        },
        {
            "doc_id": "doc_quantum_computing",
            "title": "Quantum Computing",
            "text": (
                "Quantum computing leverages the principles of quantum mechanics, "
                "including superposition and entanglement, to process information "
                "in fundamentally different ways from classical computers. A quantum "
                "bit (qubit) can exist in a superposition of states, representing "
                "both 0 and 1 simultaneously, unlike classical bits. This allows "
                "quantum computers to explore multiple solutions in parallel. "
                "Key algorithms include Shor's algorithm for integer factorization "
                "(threatening RSA encryption), Grover's algorithm for database "
                "search (quadratic speedup), and the Variational Quantum "
                "Eigensolver (VQE) for chemistry simulations. Major players "
                "include IBM (with its 1,121-qubit Condor processor), Google "
                "(which demonstrated quantum supremacy with Sycamore), and "
                "various startups like IonQ and Rigetti. Challenges include "
                "quantum decoherence, error correction overhead, and the need "
                "for extreme cooling (millikelvin temperatures). Applications "
                "span drug discovery, financial optimization, cryptography, "
                "and materials science."
            ),
        },
    ]

    # ------------------------------------------------------------------
    # 2. Initialise the pipeline
    # ------------------------------------------------------------------
    print("Step 1: Initialising ACORA pipeline...")
    print("-" * 50)

    pipeline = ACORAPipeline()

    print()
    print("Step 2: Processing documents and building indexes...")
    print("-" * 50)

    pipeline.initialize(sample_documents)

    # ------------------------------------------------------------------
    # 3. Run demo queries
    # ------------------------------------------------------------------
    demo_queries = [
        {
            "query": "What is the GDP of India in 2024?",
            "type": "Lexical (entity-heavy, specific facts)",
        },
        {
            "query": "Explain how neural networks learn representations",
            "type": "Semantic (conceptual, how/why question)",
        },
        {
            "query": "What are the effects of climate change on agriculture?",
            "type": "Ambiguous (broad topic, multi-faceted)",
        },
    ]

    print()
    print("Step 3: Running demo queries...")
    print("=" * 70)

    for i, demo in enumerate(demo_queries, 1):
        query_text = demo["query"]
        expected_type = demo["type"]

        print(f"\n{'─' * 70}")
        print(f"  Query {i}: {query_text}")
        print(f"  Expected type: {expected_type}")
        print(f"{'─' * 70}")

        result = pipeline.query(query_text)

        # Print results
        print(f"\n  📋 Classification: {result['classification'].get('query_type', 'N/A')}")
        print(f"     Confidence:     {result['classification'].get('confidence', 0.0):.3f}")

        print(f"\n  🔍 Retrieval:")
        print(f"     Candidates:     {len(result['retrieval_results'])}")
        print(f"     After rerank:   {len(result['reranked_results'])}")
        print(f"     Calls:          {result['retrieval_calls']}")

        print(f"\n  🚪 Confidence Gate:")
        conf = result['confidence']
        print(f"     Passed:         {conf.get('passed', 'N/A')}")
        print(f"     Score:          {conf.get('confidence', 0.0):.3f}")

        print(f"\n  ⚡ ACOA Stats:")
        acoa = result['acoa_stats']
        print(f"     Token budget:   {acoa.get('token_budget', 'N/A')}")
        print(f"     Candidates:     {acoa.get('candidates_considered', 'N/A')}")
        print(f"     Avg density:    {acoa.get('avg_density', 'N/A')}")

        print(f"\n  ⏱️  Latency:        {result['latency_ms']:.0f}ms")

        print(f"\n  💬 Answer:")
        answer = result['answer']
        # Truncate long answers for demo display
        if len(answer) > 300:
            print(f"     {answer[:300]}...")
        else:
            print(f"     {answer}")

    print(f"\n{'=' * 70}")
    print("  Demo complete.")
    print(f"{'=' * 70}")
