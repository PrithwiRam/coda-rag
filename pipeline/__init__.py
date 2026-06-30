"""
ACORA Pipeline Package
======================
Core pipeline components for the Adaptive Context-Optimized Retrieval Architecture.
"""

from pipeline.document_processor import DocumentProcessor
from pipeline.query_feature_extractor import QueryFeatureExtractor
from pipeline.query_classifier import QueryClassifier
from pipeline.query_encoder import QueryEncoder
from pipeline.retriever_bm25 import BM25Retriever
from pipeline.retriever_dense import DenseRetriever
from pipeline.retriever_hybrid import HybridRetriever
from pipeline.adaptive_router import AdaptiveRouter
from pipeline.retrieval_computation_planner import RetrievalComputationPlanner
from pipeline.execution_plan import ExecutionPlan
from pipeline.reranker import CrossEncoderReranker
from pipeline.confidence_gate import ConfidenceGate
from pipeline.confidence_predictor import ConfidencePredictor
from pipeline.acoa import ACOA
from pipeline.mosaic import MOSAIC
from pipeline.reward_computer import RewardComputer
from pipeline.generator import LLMGenerator

__all__ = [
    "DocumentProcessor",
    "QueryFeatureExtractor",
    "QueryClassifier",
    "QueryEncoder",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "AdaptiveRouter",
    "RetrievalComputationPlanner",
    "ExecutionPlan",
    "CrossEncoderReranker",
    "ConfidenceGate",
    "ConfidencePredictor",
    "ACOA",
    "MOSAIC",
    "RewardComputer",
    "LLMGenerator",
]
