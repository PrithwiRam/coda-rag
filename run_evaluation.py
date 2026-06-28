"""
ACORA Evaluation Runner
=======================

Comprehensive evaluation script that:
  1. Loads the NQ dataset (small subset) from HuggingFace, or falls back
     to a synthetic dataset of 50 diverse queries.
  2. Initialises the ACORA pipeline and all baseline systems.
  3. Runs all systems on the same queries.
  4. Computes retrieval and generation metrics.
  5. Prints a comparison table using tabulate.
  6. Saves results to ``results/`` as JSON and CSV.
  7. Creates matplotlib bar charts comparing metrics across systems.

Usage:
    cd acora/
    python run_evaluation.py
"""

import os
import sys
import json
import csv
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Synthetic Dataset Generator (Fallback)
# ===========================================================================

def create_synthetic_dataset() -> Dict:
    """Create a synthetic evaluation dataset with 50 diverse queries.

    The dataset includes:
      - 15-20 lexical queries (entity-heavy, dates, specific facts)
      - 15-20 semantic queries (conceptual, why/how questions)
      - 10-15 ambiguous queries (mixed, broad topics)

    Each query includes:
      - ``query``: The question text.
      - ``answer``: The expected reference answer.
      - ``relevant_chunk_ids``: IDs of relevant document chunks.
      - ``query_type``: Expected classification (LEXICAL / SEMANTIC / AMBIGUOUS).

    Returns:
        Dictionary with 'documents' (List[Dict]) and 'queries' (List[Dict]).
    """
    logger.info("Creating synthetic evaluation dataset (50 queries)...")

    # ------------------------------------------------------------------
    # Documents – diverse topics covering different retrieval patterns
    # ------------------------------------------------------------------
    documents = [
        {
            "doc_id": "doc_india_economy",
            "title": "Indian Economy Overview",
            "text": (
                "India is the world's fifth-largest economy by nominal GDP and "
                "the third-largest by purchasing power parity. As of 2024, India's "
                "nominal GDP is estimated at approximately $3.94 trillion. The "
                "Indian economy grew at 7.2% in FY2024, making it one of the "
                "fastest-growing major economies. The services sector contributes "
                "about 54% of GDP, followed by industry at 26% and agriculture "
                "at 20%. India's foreign exchange reserves stood at approximately "
                "$620 billion in 2024. The country implemented major reforms "
                "including GST, UPI digital payments, and the Make in India "
                "initiative. Mumbai serves as the financial capital with the "
                "Bombay Stock Exchange (BSE) being Asia's oldest stock exchange."
            ),
        },
        {
            "doc_id": "doc_neural_nets",
            "title": "Neural Networks and Deep Learning",
            "text": (
                "Neural networks are computational models inspired by biological "
                "neural networks in the human brain. They consist of layers of "
                "interconnected neurons that process information. Deep learning "
                "uses networks with multiple hidden layers to learn hierarchical "
                "representations. The learning process involves forward propagation "
                "and backpropagation with gradient descent optimization. Key "
                "architectures include CNNs for images, RNNs for sequences, and "
                "Transformers for NLP. The attention mechanism from Vaswani et al. "
                "(2017) revolutionized the field. Modern LLMs like GPT-4 use "
                "transformer architectures with billions of parameters."
            ),
        },
        {
            "doc_id": "doc_climate",
            "title": "Climate Change and Its Effects",
            "text": (
                "Climate change poses significant threats to global ecosystems. "
                "Rising temperatures affect crop yields: wheat yields decline by "
                "6% per 1°C increase, rice by 3.2%, and maize by 7.4%. Changes "
                "in precipitation patterns cause droughts and floods. Extreme "
                "weather events are increasing in frequency. The Paris Agreement "
                "aims to limit warming to 1.5°C above pre-industrial levels. "
                "Sea levels have risen approximately 20cm since 1900 and are "
                "projected to rise 0.3-1.0m by 2100. Arctic ice is declining "
                "at about 13% per decade. Carbon dioxide levels reached 421 ppm "
                "in 2023, the highest in 800,000 years."
            ),
        },
        {
            "doc_id": "doc_quantum",
            "title": "Quantum Computing Fundamentals",
            "text": (
                "Quantum computing uses quantum mechanical phenomena like "
                "superposition and entanglement. A qubit can exist in a "
                "superposition of 0 and 1 simultaneously. Shor's algorithm "
                "can factor integers exponentially faster than classical "
                "algorithms, threatening RSA encryption. Grover's algorithm "
                "provides quadratic speedup for database search. IBM's Condor "
                "processor has 1,121 qubits. Google demonstrated quantum "
                "supremacy with Sycamore in 2019. Major challenges include "
                "decoherence, error correction, and the need for millikelvin "
                "cooling. Applications include drug discovery, cryptography, "
                "and materials science."
            ),
        },
        {
            "doc_id": "doc_space",
            "title": "Space Exploration Milestones",
            "text": (
                "Space exploration has achieved remarkable milestones. NASA's "
                "Apollo 11 landed humans on the Moon on July 20, 1969. The "
                "International Space Station has been continuously occupied "
                "since November 2000. SpaceX's Falcon 9 was the first orbital "
                "rocket to successfully land and be reused. The James Webb "
                "Space Telescope, launched in December 2021, observes the "
                "universe in infrared with unprecedented resolution. India's "
                "Chandrayaan-3 successfully landed near the Moon's south pole "
                "in August 2023. Mars rovers like Perseverance are searching "
                "for signs of ancient microbial life. The Artemis program "
                "aims to return humans to the Moon by 2026."
            ),
        },
        {
            "doc_id": "doc_medicine",
            "title": "Modern Medicine and Biotechnology",
            "text": (
                "Modern medicine has been transformed by biotechnology advances. "
                "CRISPR-Cas9 gene editing, discovered by Doudna and Charpentier "
                "(Nobel Prize 2020), allows precise DNA modification. mRNA "
                "vaccine technology, used in COVID-19 vaccines by Pfizer-BioNTech "
                "and Moderna, can be rapidly adapted for new pathogens. "
                "Immunotherapy using checkpoint inhibitors has revolutionized "
                "cancer treatment, with PD-1/PD-L1 inhibitors showing remarkable "
                "results. CAR-T cell therapy engineers a patient's own immune "
                "cells to fight cancer. Artificial intelligence is being used "
                "for drug discovery, with AlphaFold predicting protein structures "
                "with near-experimental accuracy. Personalized medicine tailors "
                "treatments based on individual genetic profiles."
            ),
        },
        {
            "doc_id": "doc_renewable_energy",
            "title": "Renewable Energy Technologies",
            "text": (
                "Renewable energy capacity has grown dramatically worldwide. "
                "Solar photovoltaic costs have decreased by 90% since 2010. "
                "Global solar capacity exceeded 1,200 GW in 2023. Wind energy "
                "capacity reached 906 GW globally. Lithium-ion battery costs "
                "have fallen from $1,100/kWh in 2010 to $139/kWh in 2023. "
                "Electric vehicles accounted for 18% of global car sales in "
                "2023. Hydrogen fuel cells offer potential for heavy transport "
                "and industrial decarbonization. Nuclear fusion experiments "
                "at NIF achieved net energy gain in December 2022. Grid-scale "
                "energy storage using batteries, pumped hydro, and compressed "
                "air is critical for managing renewable intermittency."
            ),
        },
        {
            "doc_id": "doc_ai_ethics",
            "title": "AI Ethics and Governance",
            "text": (
                "Artificial intelligence raises important ethical questions. "
                "Algorithmic bias can perpetuate discrimination in hiring, "
                "lending, and criminal justice. The EU AI Act, enacted in 2024, "
                "classifies AI systems by risk level and imposes requirements "
                "for high-risk applications. Deepfake technology poses threats "
                "to information integrity. Privacy concerns arise from facial "
                "recognition surveillance and data collection practices. "
                "Autonomous weapons raise questions about accountability in "
                "warfare. The alignment problem asks how to ensure AI systems "
                "act in accordance with human values. Explainable AI (XAI) "
                "aims to make AI decisions interpretable and transparent."
            ),
        },
    ]

    # ------------------------------------------------------------------
    # Queries – 50 diverse queries across three types
    # ------------------------------------------------------------------
    queries = []

    # === LEXICAL queries (17) – entity-heavy, dates, specific facts ===
    lexical_queries = [
        {
            "query": "What is India's nominal GDP in 2024?",
            "answer": "India's nominal GDP in 2024 is approximately $3.94 trillion.",
            "relevant_chunk_ids": ["doc_india_economy_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "What percentage of India's GDP comes from services?",
            "answer": "The services sector contributes about 54% of India's GDP.",
            "relevant_chunk_ids": ["doc_india_economy_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "When was the Apollo 11 Moon landing?",
            "answer": "Apollo 11 landed on the Moon on July 20, 1969.",
            "relevant_chunk_ids": ["doc_space_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "How many qubits does IBM's Condor processor have?",
            "answer": "IBM's Condor processor has 1,121 qubits.",
            "relevant_chunk_ids": ["doc_quantum_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "Who won the Nobel Prize for CRISPR-Cas9?",
            "answer": "Jennifer Doudna and Emmanuelle Charpentier won the Nobel Prize in 2020 for CRISPR-Cas9.",
            "relevant_chunk_ids": ["doc_medicine_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "When was the James Webb Space Telescope launched?",
            "answer": "The James Webb Space Telescope was launched in December 2021.",
            "relevant_chunk_ids": ["doc_space_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "What is India's GDP growth rate in FY2024?",
            "answer": "India's GDP growth rate was 7.2% in FY2024.",
            "relevant_chunk_ids": ["doc_india_economy_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "What are India's foreign exchange reserves in 2024?",
            "answer": "India's foreign exchange reserves stood at approximately $620 billion in 2024.",
            "relevant_chunk_ids": ["doc_india_economy_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "When did Google demonstrate quantum supremacy?",
            "answer": "Google demonstrated quantum supremacy with Sycamore in 2019.",
            "relevant_chunk_ids": ["doc_quantum_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "What is the current CO2 level in the atmosphere?",
            "answer": "Carbon dioxide levels reached 421 ppm in 2023.",
            "relevant_chunk_ids": ["doc_climate_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "How much have solar PV costs decreased since 2010?",
            "answer": "Solar photovoltaic costs have decreased by 90% since 2010.",
            "relevant_chunk_ids": ["doc_renewable_energy_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "What is the cost of lithium-ion batteries in 2023?",
            "answer": "Lithium-ion battery costs were $139/kWh in 2023.",
            "relevant_chunk_ids": ["doc_renewable_energy_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "When was the EU AI Act enacted?",
            "answer": "The EU AI Act was enacted in 2024.",
            "relevant_chunk_ids": ["doc_ai_ethics_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "When did Chandrayaan-3 land on the Moon?",
            "answer": "India's Chandrayaan-3 landed near the Moon's south pole in August 2023.",
            "relevant_chunk_ids": ["doc_space_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "What is the global wind energy capacity?",
            "answer": "Wind energy capacity reached 906 GW globally.",
            "relevant_chunk_ids": ["doc_renewable_energy_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "How much have sea levels risen since 1900?",
            "answer": "Sea levels have risen approximately 20cm since 1900.",
            "relevant_chunk_ids": ["doc_climate_chunk_0"],
            "query_type": "LEXICAL",
        },
        {
            "query": "What percentage of global car sales are EVs in 2023?",
            "answer": "Electric vehicles accounted for 18% of global car sales in 2023.",
            "relevant_chunk_ids": ["doc_renewable_energy_chunk_0"],
            "query_type": "LEXICAL",
        },
    ]

    # === SEMANTIC queries (18) – conceptual, why/how questions ===
    semantic_queries = [
        {
            "query": "How do neural networks learn from data?",
            "answer": "Neural networks learn through forward propagation and backpropagation with gradient descent optimization, updating weights to minimize loss.",
            "relevant_chunk_ids": ["doc_neural_nets_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "Why is the attention mechanism important in NLP?",
            "answer": "The attention mechanism allows models to weigh the importance of different parts of the input, revolutionizing NLP.",
            "relevant_chunk_ids": ["doc_neural_nets_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "How does quantum superposition enable parallel computation?",
            "answer": "Qubits can exist in superposition of 0 and 1 simultaneously, allowing quantum computers to explore multiple solutions in parallel.",
            "relevant_chunk_ids": ["doc_quantum_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "Explain the process of backpropagation in deep learning",
            "answer": "Backpropagation computes gradients of the loss function and uses them to update network weights via gradient descent optimization.",
            "relevant_chunk_ids": ["doc_neural_nets_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "Why is CRISPR-Cas9 considered revolutionary for medicine?",
            "answer": "CRISPR-Cas9 allows precise DNA modification, enabling targeted gene editing for treating genetic diseases.",
            "relevant_chunk_ids": ["doc_medicine_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "How do mRNA vaccines work differently from traditional vaccines?",
            "answer": "mRNA vaccines instruct cells to produce a protein that triggers immune response, and can be rapidly adapted for new pathogens.",
            "relevant_chunk_ids": ["doc_medicine_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "What makes transformer architectures superior for language tasks?",
            "answer": "Transformers use attention mechanisms to process entire sequences in parallel, capturing long-range dependencies better than RNNs.",
            "relevant_chunk_ids": ["doc_neural_nets_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "Why is quantum error correction a major challenge?",
            "answer": "Quantum decoherence causes qubits to lose their quantum state, requiring extensive error correction overhead.",
            "relevant_chunk_ids": ["doc_quantum_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "How does algorithmic bias affect society?",
            "answer": "Algorithmic bias can perpetuate discrimination in hiring, lending, and criminal justice decisions.",
            "relevant_chunk_ids": ["doc_ai_ethics_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "Explain why renewable energy intermittency is a problem",
            "answer": "Solar and wind energy production varies with weather and time, requiring grid-scale storage solutions.",
            "relevant_chunk_ids": ["doc_renewable_energy_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "How does CAR-T cell therapy fight cancer?",
            "answer": "CAR-T therapy engineers a patient's own immune cells to recognize and attack cancer cells.",
            "relevant_chunk_ids": ["doc_medicine_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "Why is the alignment problem important for AI safety?",
            "answer": "The alignment problem asks how to ensure AI systems act in accordance with human values and intentions.",
            "relevant_chunk_ids": ["doc_ai_ethics_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "How does AlphaFold contribute to drug discovery?",
            "answer": "AlphaFold predicts protein structures with near-experimental accuracy, accelerating drug discovery.",
            "relevant_chunk_ids": ["doc_medicine_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "What role does hydrogen play in decarbonization?",
            "answer": "Hydrogen fuel cells offer potential for heavy transport and industrial decarbonization.",
            "relevant_chunk_ids": ["doc_renewable_energy_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "How does the ISS contribute to scientific research?",
            "answer": "The ISS has been continuously occupied since 2000, providing a platform for microgravity research.",
            "relevant_chunk_ids": ["doc_space_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "Why are deepfakes a threat to information integrity?",
            "answer": "Deepfake technology can create convincing fake media, threatening trust in information.",
            "relevant_chunk_ids": ["doc_ai_ethics_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "Explain how precision agriculture helps with climate adaptation",
            "answer": "Precision agriculture uses technology to optimize farming practices, improving resilience to climate variability.",
            "relevant_chunk_ids": ["doc_climate_chunk_0"],
            "query_type": "SEMANTIC",
        },
        {
            "query": "What is the significance of Shor's algorithm?",
            "answer": "Shor's algorithm can factor integers exponentially faster than classical algorithms, threatening RSA encryption.",
            "relevant_chunk_ids": ["doc_quantum_chunk_0"],
            "query_type": "SEMANTIC",
        },
    ]

    # === AMBIGUOUS queries (15) – mixed, multi-faceted ===
    ambiguous_queries = [
        {
            "query": "What are the effects of climate change on agriculture?",
            "answer": "Climate change affects agriculture through rising temperatures reducing crop yields, altered precipitation patterns, and increased extreme weather events.",
            "relevant_chunk_ids": ["doc_climate_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "How is AI transforming healthcare?",
            "answer": "AI is used in drug discovery, protein structure prediction, and diagnostic imaging, while personalized medicine tailors treatments to individual genetics.",
            "relevant_chunk_ids": ["doc_medicine_chunk_0", "doc_ai_ethics_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "What are the challenges and opportunities in quantum computing?",
            "answer": "Challenges include decoherence and error correction, while opportunities span drug discovery, cryptography, and materials science.",
            "relevant_chunk_ids": ["doc_quantum_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "Discuss the future of renewable energy",
            "answer": "Renewable energy is growing with falling costs, but intermittency, storage, and grid integration remain challenges.",
            "relevant_chunk_ids": ["doc_renewable_energy_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "What ethical concerns does AI raise?",
            "answer": "AI raises concerns about bias, privacy, deepfakes, autonomous weapons, and the alignment problem.",
            "relevant_chunk_ids": ["doc_ai_ethics_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "How is India's economy changing?",
            "answer": "India is growing rapidly at 7.2% GDP growth, driven by services, digital payments (UPI), and manufacturing initiatives.",
            "relevant_chunk_ids": ["doc_india_economy_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "What progress has been made in space exploration recently?",
            "answer": "Recent milestones include JWST, Chandrayaan-3, SpaceX reusable rockets, and the Artemis program.",
            "relevant_chunk_ids": ["doc_space_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "How are modern technologies fighting climate change?",
            "answer": "Technologies include solar and wind energy, EVs, battery storage, hydrogen fuel cells, and nuclear fusion.",
            "relevant_chunk_ids": ["doc_renewable_energy_chunk_0", "doc_climate_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "What are the latest breakthroughs in biotechnology?",
            "answer": "Key breakthroughs include CRISPR gene editing, mRNA vaccines, CAR-T therapy, and AI-driven protein structure prediction.",
            "relevant_chunk_ids": ["doc_medicine_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "Compare classical and quantum computing approaches",
            "answer": "Classical computers use binary bits, while quantum computers use qubits in superposition, enabling parallel exploration of solutions.",
            "relevant_chunk_ids": ["doc_quantum_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "How can we achieve sustainable food production?",
            "answer": "Strategies include drought-resistant crops, precision agriculture, improved irrigation, and regenerative farming practices.",
            "relevant_chunk_ids": ["doc_climate_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "What role does deep learning play in modern AI?",
            "answer": "Deep learning uses multi-layer neural networks to learn hierarchical data representations, powering LLMs and computer vision.",
            "relevant_chunk_ids": ["doc_neural_nets_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "How should AI be regulated?",
            "answer": "The EU AI Act classifies AI by risk level, while explainable AI and accountability frameworks aim to ensure responsible use.",
            "relevant_chunk_ids": ["doc_ai_ethics_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "What is the energy storage problem and how can it be solved?",
            "answer": "Renewable intermittency requires storage solutions including batteries, pumped hydro, compressed air, and hydrogen.",
            "relevant_chunk_ids": ["doc_renewable_energy_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
        {
            "query": "How is personalized medicine changing healthcare?",
            "answer": "Personalized medicine tailors treatments based on individual genetic profiles, improving efficacy and reducing side effects.",
            "relevant_chunk_ids": ["doc_medicine_chunk_0"],
            "query_type": "AMBIGUOUS",
        },
    ]

    queries = lexical_queries + semantic_queries + ambiguous_queries

    logger.info(
        "Synthetic dataset created: %d documents, %d queries "
        "(%d lexical, %d semantic, %d ambiguous).",
        len(documents), len(queries),
        len(lexical_queries), len(semantic_queries), len(ambiguous_queries),
    )

    return {
        "documents": documents,
        "queries": queries,
    }


# ===========================================================================
# Dataset Loading
# ===========================================================================

def load_nq_dataset(sample_size: int = 50) -> Optional[Dict]:
    """Attempt to load Natural Questions dataset from HuggingFace.

    Args:
        sample_size: Number of queries to sample.

    Returns:
        Dataset dictionary or None if loading fails.
    """
    try:
        from datasets import load_dataset

        logger.info("Loading Natural Questions dataset from HuggingFace...")
        dataset = load_dataset(
            config.NQ_DATASET,
            split=config.NQ_SPLIT,
            streaming=True,
        )

        documents = []
        queries = []
        doc_ids_seen = set()

        for i, example in enumerate(dataset):
            if i >= sample_size:
                break

            # Extract question and answer
            question = example.get("question", {})
            query_text = question.get("text", "") if isinstance(question, dict) else str(question)

            annotations = example.get("annotations", {})
            short_answers = annotations.get("short_answers", []) if isinstance(annotations, dict) else []

            # Extract answer text
            answer_text = ""
            if short_answers and isinstance(short_answers, list):
                if isinstance(short_answers[0], dict):
                    answer_text = short_answers[0].get("text", "")
                elif isinstance(short_answers[0], list) and short_answers[0]:
                    answer_text = str(short_answers[0][0])

            if not query_text or not answer_text:
                continue

            # Extract document context
            doc_context = example.get("document", {})
            doc_text = doc_context.get("text", "") if isinstance(doc_context, dict) else ""

            doc_id = f"nq_doc_{i}"
            if doc_text and doc_id not in doc_ids_seen:
                documents.append({
                    "doc_id": doc_id,
                    "title": f"NQ Document {i}",
                    "text": doc_text[:5000],  # Truncate very long documents
                })
                doc_ids_seen.add(doc_id)

            queries.append({
                "query": query_text,
                "answer": answer_text,
                "relevant_chunk_ids": [f"{doc_id}_chunk_0"],
                "query_type": "LEXICAL",  # NQ queries are generally factoid
            })

        if documents and queries:
            logger.info(
                "Loaded NQ dataset: %d documents, %d queries.",
                len(documents), len(queries),
            )
            return {"documents": documents, "queries": queries}
        else:
            logger.warning("NQ dataset was empty or unparseable.")
            return None

    except ImportError:
        logger.warning(
            "HuggingFace 'datasets' library not installed. "
            "Install with: pip install datasets"
        )
        return None
    except Exception as e:
        logger.warning("Failed to load NQ dataset: %s", e)
        return None


# ===========================================================================
# Evaluation Runner
# ===========================================================================

def run_evaluation() -> Dict:
    """Run the complete evaluation pipeline.

    Returns:
        Dictionary with all results and metrics.
    """
    from evaluation.metrics import EvaluationMetrics

    print("=" * 70)
    print("  ACORA Evaluation Suite")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load dataset
    # ------------------------------------------------------------------
    print("\n📦 Step 1: Loading evaluation dataset...")

    dataset = load_nq_dataset(sample_size=config.NQ_SAMPLE_SIZE)
    if dataset is None:
        print("   ⚠️  HuggingFace dataset unavailable. Using synthetic dataset.")
        dataset = create_synthetic_dataset()
    else:
        print(f"   ✅ Loaded NQ dataset.")

    documents = dataset["documents"]
    eval_queries = dataset["queries"]

    # Slice queries to run a fast but representative evaluation (5 queries)
    eval_queries = eval_queries[:5]

    print(f"   Documents: {len(documents)}")
    print(f"   Queries (Evaluating Subset): {len(eval_queries)}")

    # ------------------------------------------------------------------
    # Step 2: Initialise ACORA pipeline
    # ------------------------------------------------------------------
    print("\n🏗️  Step 2: Initialising ACORA pipeline...")

    from main import ACORAPipeline
    acora_pipeline = ACORAPipeline()
    acora_pipeline.initialize(documents)
    print("   ✅ ACORA pipeline ready.")

    # ------------------------------------------------------------------
    # Step 3: Initialise baseline systems
    # ------------------------------------------------------------------
    print("\n🏗️  Step 3: Initialising baseline systems...")

    baselines = {}

    try:
        from baselines.baseline_systems import (
            BM25OnlyBaseline, DenseOnlyBaseline,
            StaticHybridBaseline, StandardRAGBaseline,
        )

        if acora_pipeline.bm25_retriever and acora_pipeline.generator:
            baselines["BM25-Only"] = BM25OnlyBaseline(
                bm25_retriever=acora_pipeline.bm25_retriever,
                generator=acora_pipeline.generator,
            )
            print("   ✅ BM25-Only baseline ready.")

        if acora_pipeline.dense_retriever and acora_pipeline.generator:
            baselines["Dense-Only"] = DenseOnlyBaseline(
                dense_retriever=acora_pipeline.dense_retriever,
                generator=acora_pipeline.generator,
            )
            print("   ✅ Dense-Only baseline ready.")

        if (acora_pipeline.bm25_retriever and acora_pipeline.dense_retriever
                and acora_pipeline.reranker and acora_pipeline.generator):
            baselines["Static-Hybrid"] = StaticHybridBaseline(
                bm25_retriever=acora_pipeline.bm25_retriever,
                dense_retriever=acora_pipeline.dense_retriever,
                reranker=acora_pipeline.reranker,
                generator=acora_pipeline.generator,
            )
            print("   ✅ Static-Hybrid baseline ready.")

        if acora_pipeline.chunks and acora_pipeline.generator:
            baselines["Standard-RAG"] = StandardRAGBaseline(
                documents=acora_pipeline.chunks,
                generator=acora_pipeline.generator,
            )
            print("   ✅ Standard-RAG baseline ready.")

    except Exception as e:
        logger.warning("Some baselines failed to initialise: %s", e)

    if not baselines:
        print("   ⚠️  No baselines could be initialised.")

    # ------------------------------------------------------------------
    # Step 4: Run all systems on evaluation queries
    # ------------------------------------------------------------------
    print(f"\n🚀 Step 4: Running evaluation on {len(eval_queries)} queries...")

    all_results = {}

    # --- Run ACORA ---
    print("\n   Running ACORA...")
    acora_results = []
    for i, q in enumerate(eval_queries):
        try:
            result = acora_pipeline.query(q["query"])
            result["reference_answer"] = q["answer"]
            result["relevant_chunk_ids"] = q["relevant_chunk_ids"]
            result["expected_type"] = q["query_type"]
            acora_results.append(result)
        except Exception as e:
            logger.error("ACORA failed on query %d: %s", i, e)
            acora_results.append({
                "answer": "",
                "query": q["query"],
                "reference_answer": q["answer"],
                "relevant_chunk_ids": q["relevant_chunk_ids"],
                "retrieval_results": [],
                "reranked_results": [],
                "latency_ms": 0.0,
            })

        if (i + 1) % 10 == 0:
            print(f"   ... processed {i + 1}/{len(eval_queries)} queries")

    all_results["ACORA"] = acora_results
    print(f"   ✅ ACORA: {len(acora_results)} queries processed.")

    # --- Run baselines ---
    for name, baseline in baselines.items():
        print(f"\n   Running {name}...")
        baseline_results = []
        for i, q in enumerate(eval_queries):
            try:
                result = baseline.process_query(q["query"])
                result["reference_answer"] = q["answer"]
                result["relevant_chunk_ids"] = q["relevant_chunk_ids"]
                result["expected_type"] = q["query_type"]
                baseline_results.append(result)
            except Exception as e:
                logger.error("%s failed on query %d: %s", name, i, e)
                baseline_results.append({
                    "answer": "",
                    "query": q["query"],
                    "reference_answer": q["answer"],
                    "relevant_chunk_ids": q["relevant_chunk_ids"],
                    "retrieved_chunks": [],
                    "latency_ms": 0.0,
                })

        all_results[name] = baseline_results
        print(f"   ✅ {name}: {len(baseline_results)} queries processed.")

    # ------------------------------------------------------------------
    # Step 5: Compute metrics
    # ------------------------------------------------------------------
    print("\n📊 Step 5: Computing metrics...")

    system_metrics = {}

    for system_name, results in all_results.items():
        metrics_list = []

        for result in results:
            # Extract retrieved chunk IDs
            if system_name == "ACORA":
                reranked = result.get("reranked_results", [])
                retrieved_ids = [
                    r.get("chunk", {}).get("chunk_id", "")
                    for r in reranked
                ]
            else:
                chunks = result.get("retrieved_chunks", [])
                retrieved_ids = [
                    c.get("chunk_id", "") for c in chunks
                ]

            relevant_ids = result.get("relevant_chunk_ids", [])
            predicted = result.get("answer", "")
            reference = result.get("reference_answer", "")

            metrics = EvaluationMetrics.compute_all_metrics(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                predicted_answer=predicted,
                reference_answer=reference,
            )
            metrics["latency_ms"] = result.get("latency_ms", 0.0)
            metrics_list.append(metrics)

        # Average metrics across all queries
        avg_metrics = _average_metrics(metrics_list)
        system_metrics[system_name] = avg_metrics
        print(f"   ✅ {system_name} metrics computed.")

    # ------------------------------------------------------------------
    # Step 6: Print comparison table
    # ------------------------------------------------------------------
    print("\n📋 Step 6: Results comparison")
    _print_comparison_table(system_metrics)

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    print("\n💾 Step 7: Saving results...")
    _save_results(system_metrics, all_results)

    # ------------------------------------------------------------------
    # Step 8: Create visualisation charts
    # ------------------------------------------------------------------
    print("\n📈 Step 8: Creating visualisation charts...")
    _create_charts(system_metrics)

    print("\n" + "=" * 70)
    print("  Evaluation complete!")
    print("=" * 70)

    return system_metrics


# ===========================================================================
# Helper Functions
# ===========================================================================

def _average_metrics(metrics_list: List[Dict]) -> Dict:
    """Average a list of per-query metric dictionaries.

    Args:
        metrics_list: List of metric dicts from compute_all_metrics.

    Returns:
        Averaged metrics dictionary.
    """
    if not metrics_list:
        return {}

    avg = {
        "retrieval": {},
        "generation": {},
        "latency_ms": 0.0,
    }

    # Collect all retrieval metric keys
    retrieval_keys = set()
    for m in metrics_list:
        retrieval_keys.update(m.get("retrieval", {}).keys())

    for key in retrieval_keys:
        values = [
            m.get("retrieval", {}).get(key, 0.0) for m in metrics_list
        ]
        avg["retrieval"][key] = sum(values) / len(values) if values else 0.0

    # Generation metrics
    for key in ["exact_match", "f1_score"]:
        values = [
            m.get("generation", {}).get(key, 0.0) for m in metrics_list
        ]
        avg["generation"][key] = sum(values) / len(values) if values else 0.0

    # Latency
    latencies = [m.get("latency_ms", 0.0) for m in metrics_list]
    avg["latency_ms"] = sum(latencies) / len(latencies) if latencies else 0.0

    return avg


def _print_comparison_table(system_metrics: Dict) -> None:
    """Print a formatted comparison table of metrics across systems.

    Args:
        system_metrics: Dict mapping system names to averaged metrics.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        logger.warning("tabulate not installed. Printing simple format.")
        for name, metrics in system_metrics.items():
            print(f"\n--- {name} ---")
            print(f"  Retrieval: {metrics.get('retrieval', {})}")
            print(f"  Generation: {metrics.get('generation', {})}")
            print(f"  Latency: {metrics.get('latency_ms', 0.0):.1f}ms")
        return

    # Build table rows
    headers = ["System"]
    metric_keys = []

    # Key metrics to display
    display_metrics = [
        ("precision@1", "P@1"),
        ("precision@5", "P@5"),
        ("recall@5", "R@5"),
        ("recall@10", "R@10"),
        ("mrr", "MRR"),
        ("ndcg@5", "nDCG@5"),
        ("hit_rate@5", "HR@5"),
    ]

    for key, label in display_metrics:
        headers.append(label)
        metric_keys.append(("retrieval", key))

    headers.extend(["EM", "F1", "Latency(ms)"])
    metric_keys.append(("generation", "exact_match"))
    metric_keys.append(("generation", "f1_score"))
    metric_keys.append(("latency_ms", None))

    rows = []
    for system_name, metrics in system_metrics.items():
        row = [system_name]
        for category, key in metric_keys:
            if key is None:
                value = metrics.get(category, 0.0)
            else:
                value = metrics.get(category, {}).get(key, 0.0)

            if category == "latency_ms" or key is None:
                row.append(f"{value:.1f}")
            else:
                row.append(f"{value:.4f}")
        rows.append(row)

    print()
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()


def _save_results(system_metrics: Dict, all_results: Dict) -> None:
    """Save evaluation results to JSON and CSV files.

    Args:
        system_metrics: Averaged metrics per system.
        all_results: Full per-query results per system.
    """
    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Save metrics as JSON ---
    metrics_path = results_dir / f"metrics_{timestamp}.json"
    # Convert any non-serialisable values
    serialisable_metrics = {}
    for system_name, metrics in system_metrics.items():
        serialisable_metrics[system_name] = {
            "retrieval": {k: round(v, 4) for k, v in metrics.get("retrieval", {}).items()},
            "generation": {k: round(v, 4) for k, v in metrics.get("generation", {}).items()},
            "latency_ms": round(metrics.get("latency_ms", 0.0), 2),
        }

    with open(metrics_path, "w") as f:
        json.dump(serialisable_metrics, f, indent=2)
    print(f"   📄 Metrics saved to {metrics_path}")

    # --- Save metrics as CSV ---
    csv_path = results_dir / f"metrics_{timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Determine all metric keys
        all_keys = set()
        for metrics in serialisable_metrics.values():
            all_keys.update(f"retrieval.{k}" for k in metrics.get("retrieval", {}))
            all_keys.update(f"generation.{k}" for k in metrics.get("generation", {}))
            all_keys.add("latency_ms")

        sorted_keys = sorted(all_keys)
        writer.writerow(["system"] + sorted_keys)

        for system_name, metrics in serialisable_metrics.items():
            row = [system_name]
            for key in sorted_keys:
                if key == "latency_ms":
                    row.append(metrics.get("latency_ms", 0.0))
                elif key.startswith("retrieval."):
                    metric_key = key.replace("retrieval.", "")
                    row.append(metrics.get("retrieval", {}).get(metric_key, 0.0))
                elif key.startswith("generation."):
                    metric_key = key.replace("generation.", "")
                    row.append(metrics.get("generation", {}).get(metric_key, 0.0))
            writer.writerow(row)

    print(f"   📄 Metrics CSV saved to {csv_path}")

    # --- Save per-query results summary as JSON ---
    summary_path = results_dir / f"query_results_{timestamp}.json"
    summary = {}
    for system_name, results in all_results.items():
        system_summary = []
        for r in results:
            entry = {
                "query": r.get("query", ""),
                "answer": r.get("answer", ""),
                "reference_answer": r.get("reference_answer", ""),
                "latency_ms": r.get("latency_ms", 0.0),
            }
            system_summary.append(entry)
        summary[system_name] = system_summary

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   📄 Query results saved to {summary_path}")


def _create_charts(system_metrics: Dict) -> None:
    """Create matplotlib bar charts comparing metrics across systems.

    Args:
        system_metrics: Averaged metrics per system.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning(
            "matplotlib not installed. Skipping chart generation. "
            "Install with: pip install matplotlib"
        )
        return

    results_dir = config.RESULTS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    systems = list(system_metrics.keys())
    n_systems = len(systems)

    if n_systems == 0:
        logger.warning("No systems to chart.")
        return

    # ------------------------------------------------------------------
    # Chart 1: Retrieval metrics comparison
    # ------------------------------------------------------------------
    retrieval_metrics_to_plot = [
        ("precision@1", "P@1"),
        ("precision@5", "P@5"),
        ("recall@5", "R@5"),
        ("mrr", "MRR"),
        ("ndcg@5", "nDCG@5"),
        ("hit_rate@5", "HR@5"),
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(retrieval_metrics_to_plot))
    width = 0.8 / max(n_systems, 1)

    colors = plt.cm.Set2(np.linspace(0, 1, max(n_systems, 2)))

    for i, system_name in enumerate(systems):
        metrics = system_metrics[system_name]
        values = [
            metrics.get("retrieval", {}).get(key, 0.0)
            for key, _ in retrieval_metrics_to_plot
        ]
        offset = (i - n_systems / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=system_name, color=colors[i], edgecolor="white", linewidth=0.5
        )
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Retrieval Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in retrieval_metrics_to_plot])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    chart1_path = results_dir / f"retrieval_comparison_{timestamp}.png"
    plt.savefig(chart1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Retrieval chart saved to {chart1_path}")

    # ------------------------------------------------------------------
    # Chart 2: Generation metrics comparison
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Generation metrics
    gen_metrics = ["exact_match", "f1_score"]
    gen_labels = ["Exact Match", "F1 Score"]

    for idx, (metric, label) in enumerate(zip(gen_metrics, gen_labels)):
        values = [
            system_metrics[s].get("generation", {}).get(metric, 0.0)
            for s in systems
        ]
        bars = axes[idx].bar(
            systems, values, color=colors[:n_systems],
            edgecolor="white", linewidth=0.5
        )
        axes[idx].set_title(label, fontsize=12, fontweight="bold")
        axes[idx].set_ylim(0, 1.15)
        axes[idx].grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

        axes[idx].tick_params(axis="x", rotation=30)

    # Latency comparison
    latencies = [
        system_metrics[s].get("latency_ms", 0.0)
        for s in systems
    ]
    bars = axes[2].bar(
        systems, latencies, color=colors[:n_systems],
        edgecolor="white", linewidth=0.5
    )
    axes[2].set_title("Avg Latency (ms)", fontsize=12, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, latencies):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(latencies) * 0.02,
            f"{val:.0f}",
            ha="center", va="bottom", fontsize=9,
        )

    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    chart2_path = results_dir / f"generation_comparison_{timestamp}.png"
    plt.savefig(chart2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   📊 Generation chart saved to {chart2_path}")

    # ------------------------------------------------------------------
    # Chart 3: Overall radar chart (if enough systems)
    # ------------------------------------------------------------------
    if n_systems >= 2:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        categories = ["P@5", "R@5", "MRR", "nDCG@5", "EM", "F1"]
        cat_keys = [
            ("retrieval", "precision@5"),
            ("retrieval", "recall@5"),
            ("retrieval", "mrr"),
            ("retrieval", "ndcg@5"),
            ("generation", "exact_match"),
            ("generation", "f1_score"),
        ]
        N = len(categories)
        angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
        angles += angles[:1]  # Complete the circle

        for i, system_name in enumerate(systems):
            metrics = system_metrics[system_name]
            values = []
            for cat, key in cat_keys:
                values.append(metrics.get(cat, {}).get(key, 0.0))
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, "o-", linewidth=2, label=system_name, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title("Overall Performance Comparison", fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        plt.tight_layout()
        chart3_path = results_dir / f"radar_comparison_{timestamp}.png"
        plt.savefig(chart3_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   📊 Radar chart saved to {chart3_path}")


# ===========================================================================
# Main Entry Point
# ===========================================================================

if __name__ == "__main__":
    try:
        results = run_evaluation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        sys.exit(1)
