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
    {
        "doc_id": "doc_user_profile",
        "title": "User Profile - Harinarayanan",
        "text": "Name: Harinarayanan. From Vengara, Malappuram, Kerala. Studying IMSC in Computer Science AI and ML. Loves programming. Currently interning at IIIT Kottayam. Age: 22. Harinarayanan is 22 years old. He is 22 years old."

    },
]


# ---------------------------------------------------------------------------
# Pipeline component imports
# ---------------------------------------------------------------------------
from pipeline.document_processor import DocumentProcessor
from pipeline.confidence_gate import ConfidenceGate
from pipeline.acoa import ACOA
from pipeline.generator import LLMGenerator
from pipeline.query_encoder import QueryEncoder
from pipeline.retrieval_computation_planner import RetrievalComputationPlanner
from pipeline.confidence_predictor import ConfidencePredictor
from pipeline.mosaic import MOSAIC
from pipeline.reward_computer import RewardComputer

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
    """End-to-end ACORA v2 pipeline.

    Integrates all components from document processing through answer
    generation. Must be initialised with documents before querying.

    Attributes:
        is_initialized (bool): Whether the pipeline has been initialised
            with documents and indexes are built.
    """

    def __init__(self) -> None:
        """Initialise all pipeline components."""
        logger.info("=" * 60)
        logger.info("Initialising ACORA Pipeline components...")
        logger.info("=" * 60)

        # Document processing
        self.document_processor = DocumentProcessor()

        # Retrievers – will be initialised with data in initialize()
        self.bm25_retriever = None
        self.dense_retriever = None
        self.hybrid_retriever = None

        # Query analysis (v2)
        self.query_encoder = None
        self.rcp = None # RetrievalComputationPlanner

        # Reranker
        self.reranker = None

        # Confidence Predictor (v2)
        self.confidence_predictor = None

        # MOSAIC context optimizer (v2)
        self.mosaic = None

        # LLM generator
        self.generator = None

        # Reward Computer (v2)
        self.reward_computer = None

        # Pipeline state
        self.is_initialized = False
        self.chunks = []

        # Lazy-load components that may have heavy dependencies
        self._init_components()

        logger.info("ACORA Pipeline components created (not yet initialised with data).")

    def _init_components(self) -> None:
        """Lazy-initialise components with potentially heavy imports."""
        # --- Query Encoder (v2) ---
        try:
            self.query_encoder = QueryEncoder()
        except Exception as e:
            logger.warning("QueryEncoder unavailable: %s", e)
            self.query_encoder = None

        # --- Retrieval Computation Planner (v2) ---
        try:
            self.rcp = RetrievalComputationPlanner()
        except Exception as e:
            logger.warning("RetrievalComputationPlanner unavailable: %s", e)
            self.rcp = None

        # --- Cross-Encoder Reranker ---
        try:
            from pipeline.reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker()
        except (ImportError, Exception) as e:
            logger.warning("CrossEncoderReranker unavailable: %s", e)
            self.reranker = None

        # --- Confidence Predictor (v2) ---
        try:
            self.confidence_predictor = ConfidencePredictor()
        except Exception as e:
            logger.warning("ConfidencePredictor unavailable: %s", e)
            self.confidence_predictor = None

        # --- MOSAIC (v2) ---
        try:
            self.mosaic = MOSAIC()
        except Exception as e:
            logger.warning("MOSAIC unavailable: %s", e)
            self.mosaic = None
            
        # --- Reward Computer (v2) ---
        try:
            self.reward_computer = RewardComputer()
        except Exception as e:
            logger.warning("RewardComputer unavailable: %s", e)
            self.reward_computer = None

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
        """Process a single query through the full ACORA v2 pipeline.

        Pipeline stages:
            1. Encode Query → extract features (QueryEncoder)
            2. Compute Plan → select optimal execution plan (RCP)
            3. Retrieve → fetch candidate chunks based on plan
            4. Rerank → refine ordering (if plan says so)
            5. Predict Confidence → validate quality
            6. MOSAIC → submodular context assembly based on plan budget
            7. Generate → produce answer with LLM
            8. Reward → compute multi-objective reward and update RCP

        Args:
            query_text: The user's question.

        Returns:
            Comprehensive result dictionary.
        """
        if not self.is_initialized:
            return {
                "answer": "[Error] Pipeline not initialised. Call initialize() first.",
                "query": query_text,
                "latency_ms": 0.0,
            }

        start_time = time.perf_counter()
        retrieval_calls = 0

        # ==================================================================
        # Stage 1 & 2: Encode Query and Generate Execution Plan
        # ==================================================================
        q_features = {}
        plan = None
        
        if self.query_encoder and self.rcp:
            try:
                q_features = self.query_encoder.encode(query_text)
                ctx_vec = self.query_encoder.to_context_vector(q_features)
                plan = self.rcp.select_plan(ctx_vec)
                logger.info("RCP Plan: %s (depth=%d, budget=%d)", 
                            plan.strategy, plan.retrieval_depth, plan.token_budget)
            except Exception as e:
                logger.warning("RCP planning failed: %s. Using default plan.", e)
                
        if not plan:
            # Fallback default plan
            from pipeline.execution_plan import ExecutionPlan
            plan = ExecutionPlan.from_defaults(0)

        # ==================================================================
        # Stage 3: Retrieval Execution
        # ==================================================================
        retrieval_results = []
        top_k = plan.retrieval_depth
        
        try:
            if plan.strategy == "bm25" and self.bm25_retriever:
                retrieval_results = self.bm25_retriever.retrieve(query_text, top_k=top_k)
                retrieval_calls = 1
            elif plan.strategy == "dense" and self.dense_retriever:
                retrieval_results = self.dense_retriever.retrieve(query_text, top_k=top_k)
                retrieval_calls = 1
            elif self.hybrid_retriever:
                retrieval_results = self.hybrid_retriever.retrieve(query_text, top_k=top_k)
                retrieval_calls = 2
        except Exception as e:
            logger.error("Retrieval execution failed: %s", e)

        logger.info("Retrieved %d candidates.", len(retrieval_results))

        # ==================================================================
        # Stage 4: Conditional Reranking
        # ==================================================================
        reranked_results = retrieval_results

        if plan.rerank and self.reranker is not None and retrieval_results:
            try:
                rerank_depth = plan.rerank_depth if plan.rerank_depth > 0 else config.RERANK_TOP_K
                reranked_results = self.reranker.rerank(
                    query_text,
                    retrieval_results,
                    top_k=rerank_depth,
                )
                logger.info("Reranked to top-%d results.", len(reranked_results))
            except Exception as e:
                logger.warning("Reranking failed: %s. Using original order.", e)

        # ==================================================================
        # Stage 5: Confidence Prediction
        # ==================================================================
        confidence = {'confident': True, 'score': 1.0}
        if self.confidence_predictor:
            try:
                confidence = self.confidence_predictor.predict_confidence(reranked_results, query_text)
                logger.info("Confidence: %.3f (passed=%s)", confidence.get('score', 0.0), confidence.get('confident', False))
            except Exception as e:
                logger.warning("Confidence prediction failed: %s", e)

        # ==================================================================
        # Stage 6: MOSAIC Context Assembly
        # ==================================================================
        mosaic_result = {}
        context = ""
        if self.mosaic:
            try:
                mosaic_result = self.mosaic.select_context(
                    query=query_text,
                    candidates=reranked_results,
                    token_budget=plan.token_budget,
                    weights=plan.mosaic_weights
                )
                
                # Build context string
                selected_chunks = mosaic_result.get("selected_chunks", [])
                context = "\n\n".join([f"[{i+1}] {c.get('chunk', {}).get('text', '')}" 
                                      for i, c in enumerate(selected_chunks)])
                                      
                logger.info("MOSAIC selected %d chunks (%d tokens).",
                            mosaic_result.get("num_selected", 0),
                            mosaic_result.get("total_tokens", 0))
            except Exception as e:
                logger.warning("MOSAIC assembly failed: %s", e)
                # Simple fallback
                context = "\n\n".join([c.get('chunk', {}).get('text', '') for c in reranked_results[:3]])

        # ==================================================================
        # Stage 7: Generate answer
        # ==================================================================
        answer = "[No generator available]"
        
        if self.generator is not None:
            try:
                gen_result = self.generator.generate(query=query_text, context=context)
                answer = gen_result.get("answer", "")
            except Exception as e:
                logger.error("LLM generation failed: %s", e)
                answer = f"[Generation Error] {e}"

        total_latency = (time.perf_counter() - start_time) * 1000

        # ==================================================================
        # Stage 8: Reward Computation & RCP Update
        # ==================================================================
        reward_info = {}
        if self.reward_computer and self.rcp and hasattr(plan, '_arm_index'):
            try:
                temp_result = {
                    "latency_ms": total_latency,
                    "acoa_stats": mosaic_result, # map to old key for compat
                    "confidence": confidence
                }
                reward_info = self.reward_computer.compute_reward(temp_result)
                
                # Update Bandit
                ctx_vec = self.query_encoder.to_context_vector(q_features)
                self.rcp.update(ctx_vec, plan._arm_index, reward_info.get('composite_reward', 0.0))
                logger.info("RCP updated. Reward: %.3f", reward_info.get('composite_reward', 0.0))
            except Exception as e:
                logger.warning("Reward computation/update failed: %s", e)

        result = {
            "answer": answer,
            "query": query_text,
            "plan": plan.to_dict(),
            "retrieval_results": retrieval_results,
            "reranked_results": reranked_results,
            "confidence": confidence,
            "mosaic_stats": mosaic_result,
            "reward_info": reward_info,
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
        plan = result.get('plan', {})
        print(f"\n  📋 Plan Strategy:  {plan.get('strategy', 'N/A')}")
        print(f"     Depth/Budget:   {plan.get('retrieval_depth', 'N/A')} / {plan.get('token_budget', 'N/A')}")

        print(f"\n  🔍 Retrieval:")
        print(f"     Candidates:     {len(result['retrieval_results'])}")
        print(f"     After rerank:   {len(result['reranked_results'])}")
        print(f"     Calls:          {result['retrieval_calls']}")

        print(f"\n  🚪 Confidence Gate:")
        conf = result['confidence']
        print(f"     Passed:         {conf.get('confident', conf.get('passed', 'N/A'))}")
        print(f"     Score:          {conf.get('score', conf.get('confidence', 0.0)):.3f}")

        print(f"\n  ⚡ MOSAIC Stats:")
        mosaic = result.get('mosaic_stats', {})
        print(f"     Token budget:   {mosaic.get('token_budget', 'N/A')}")
        print(f"     Selected chunks:{mosaic.get('num_selected', 'N/A')}")
        print(f"     Tokens used:    {mosaic.get('total_tokens', 'N/A')}")
        
        print(f"\n  🏆 RCP Reward:")
        reward = result.get('reward_info', {})
        print(f"     Composite:      {reward.get('composite_reward', 0.0):.3f}")

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
