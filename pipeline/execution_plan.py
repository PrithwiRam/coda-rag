"""
Execution Plan Module
=====================
Defines the ExecutionPlan dataclass for the RAG pipeline.
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class ExecutionPlan:
    strategy: str
    retrieval_depth: int
    rerank: bool
    rerank_depth: int
    token_budget: int
    compression: str
    fallback: bool
    mosaic_weights: List[float] = field(default_factory=lambda: [1/6]*6)
    
    PLAN_SPACE = [
        # (strategy, depth, rerank, rerank_depth, budget, compression, fallback)
        ("bm25", 10, False, 0, 1024, "none", False),
        ("bm25", 20, False, 0, 2048, "light", True),
        ("dense", 10, False, 0, 1024, "none", False),
        ("dense", 20, False, 0, 2048, "light", True),
        ("hybrid", 10, False, 0, 1024, "none", False),
        ("hybrid", 20, False, 0, 2048, "light", True),
        ("hybrid", 20, True, 10, 2048, "medium", True),
        ("hybrid", 30, True, 15, 4096, "aggressive", True),
    ]

    def to_dict(self):
        return {
            "strategy": self.strategy,
            "retrieval_depth": self.retrieval_depth,
            "rerank": self.rerank,
            "rerank_depth": self.rerank_depth,
            "token_budget": self.token_budget,
            "compression": self.compression,
            "fallback": self.fallback,
            "mosaic_weights": self.mosaic_weights
        }

    @classmethod
    def from_defaults(cls, index: int = 0):
        if index < 0 or index >= len(cls.PLAN_SPACE):
            index = 0
        p = cls.PLAN_SPACE[index]
        return cls(p[0], p[1], p[2], p[3], p[4], p[5], p[6])
        
    def estimated_cost(self) -> float:
        cost = 0.0
        if self.strategy == "hybrid":
            cost += self.retrieval_depth * 2
        else:
            cost += self.retrieval_depth
        if self.rerank:
            cost += self.rerank_depth * 10 # reranking is expensive
        return cost
