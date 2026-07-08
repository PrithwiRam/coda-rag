"""
CODA-RAG Configuration
======================
Central configuration for all CODA-RAG components.
"""

import os
from pathlib import Path

# ==============================================================================
# Paths
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "indexes"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
for d in [DATA_DIR, INDEX_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Document Processing
# ==============================================================================

CHUNK_SIZE = 256          # tokens per chunk
CHUNK_OVERLAP = 50        # token overlap between chunks

# ==============================================================================
# Retrieval
# ==============================================================================

# BM25 Parameters
BM25_K1 = 1.5
BM25_B = 0.75
BM25_TOP_K = 20           # initial retrieval count (before reranking)

# Dense Retrieval Parameters
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_TOP_K = 20

# Hybrid Retrieval (RRF)
RRF_K = 60                # RRF constant
HYBRID_TOP_K = 20

# Final top-k after reranking
RERANK_TOP_K = 10

# ==============================================================================
# Reranker
# ==============================================================================

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ==============================================================================
# Confidence Gate
# ==============================================================================

CONFIDENCE_THRESHOLD = 0.0
FALLBACK_EXTRA_K = 10     # additional chunks to retrieve on fallback

# ==============================================================================
# ACOA (Adaptive Context Optimization Algorithm)
# ==============================================================================

ACOA_TOKEN_BUDGET = 2048        # max tokens for context
ACOA_REDUNDANCY_THRESHOLD = 0.3 # Jaccard similarity threshold

# ==============================================================================
# Query Classifier
# ==============================================================================

CLASSIFIER_CONFIDENCE_THRESHOLD = 0.8  # above this, use rule-based result
CLASSIFIER_MODEL_PATH = MODEL_DIR / "query_classifier.joblib"

# Query types
QUERY_TYPE_LEXICAL = "LEXICAL"
QUERY_TYPE_SEMANTIC = "SEMANTIC"
QUERY_TYPE_AMBIGUOUS = "AMBIGUOUS"

# ==============================================================================
# LLM Generation
# ==============================================================================

# Option 1: OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"

# Option 2: Local model via Ollama
OLLAMA_MODEL = "gemma:2b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Which LLM backend to use: "openai" or "ollama"
LLM_BACKEND = "ollama"

# Generation parameters
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 512

# ==============================================================================
# Evaluation
# ==============================================================================

# Dataset configs
NQ_DATASET = "natural_questions"
NQ_SPLIT = "validation"
NQ_SAMPLE_SIZE = 500       # number of queries for evaluation

HOTPOTQA_DATASET = "hotpot_qa"
HOTPOTQA_SPLIT = "validation"
HOTPOTQA_SAMPLE_SIZE = 500

# Metrics
EVAL_K_VALUES = [1, 3, 5, 10]

# ==============================================================================
# Prompt Template
# ==============================================================================

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the following question using ONLY the provided context. If the context does not contain sufficient information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
