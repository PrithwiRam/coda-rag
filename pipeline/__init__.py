"""
ACORA Pipeline Package
======================
Core pipeline components for the Adaptive Context-Optimized Retrieval Architecture.
"""

from pipeline.document_processor import DocumentProcessor
from pipeline.query_feature_extractor import QueryFeatureExtractor
from pipeline.query_classifier import QueryClassifier
from pipeline.retriever_bm25 import BM25Retriever
from pipeline.retriever_dense import DenseRetriever
from pipeline.retriever_hybrid import HybridRetriever
from pipeline.adaptive_router import AdaptiveRouter
from pipeline.reranker import CrossEncoderReranker
from pipeline.confidence_gate import ConfidenceGate
from pipeline.acoa import ACOA
from pipeline.generator import LLMGenerator

__all__ = [
    "DocumentProcessor",
    "QueryFeatureExtractor",
    "QueryClassifier",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "AdaptiveRouter",
    "CrossEncoderReranker",
    "ConfidenceGate",
    "ACOA",
    "LLMGenerator",
]
