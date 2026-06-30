"""
Retrieval Computation Planner
=============================
Core innovation — Retrieval Computation Planner using LinUCB contextual bandits.
"""
import sys
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from pipeline.execution_plan import ExecutionPlan

logger = logging.getLogger(__name__)

class RetrievalComputationPlanner:
    def __init__(self, n_features: int = 15, alpha: float = 1.0):
        self.n_features = n_features
        self.alpha = alpha
        self.n_arms = len(ExecutionPlan.PLAN_SPACE)
        self.A = [np.eye(n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros(n_features) for _ in range(self.n_arms)]
        self.counts = [0] * self.n_arms
        
    def select_plan(self, context_vector: np.ndarray) -> ExecutionPlan:
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            p[a] = theta_a @ context_vector + self.alpha * np.sqrt(context_vector.T @ A_inv @ context_vector)
            
        chosen_arm = int(np.argmax(p))
        self.counts[chosen_arm] += 1
        plan = ExecutionPlan.from_defaults(chosen_arm)
        plan._arm_index = chosen_arm 
        return plan
        
    def update(self, context_vector: np.ndarray, plan_index: int, reward: float):
        x = context_vector
        self.A[plan_index] += np.outer(x, x)
        self.b[plan_index] += reward * x
        
    def compute_reward(self, quality_score: float, latency_ms: float, tokens_used: int, max_latency: float = 5000.0, max_tokens: int = 4096) -> float:
        alpha = 0.6
        beta = 0.2
        gamma = 0.2
        latency_penalty = min(latency_ms / max_latency, 1.0)
        token_penalty = min(tokens_used / max_tokens, 1.0)
        return alpha * quality_score - beta * latency_penalty - gamma * token_penalty
        
    def save(self, path):
        np.savez(path, A=self.A, b=self.b, counts=self.counts)
        
    def load(self, path):
        try:
            data = np.load(path)
            self.A = list(data['A'])
            self.b = list(data['b'])
            self.counts = list(data['counts'])
        except Exception as e:
            logger.warning(f"Could not load RCP models: {e}")
            
    def get_plan_stats(self) -> dict:
        return {f"arm_{i}": count for i, count in enumerate(self.counts)}
