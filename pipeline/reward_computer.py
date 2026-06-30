"""
Reward Computer
===============
Multi-objective reward computation for RCP.
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

class RewardComputer:
    def __init__(self, alpha=0.6, beta=0.2, gamma=0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def compute_reward(self, result_dict: dict, max_latency=5000.0, max_tokens=4096) -> dict:
        latency_ms = result_dict.get('latency_ms', 0.0)
        tokens_used = result_dict.get('acoa_stats', {}).get('total_tokens', 1000)
        if not tokens_used:
            tokens_used = 1000
            
        conf = result_dict.get('confidence', {}).get('score', 0.5)
        quality = conf
        
        lat_score = max(0.0, 1.0 - (latency_ms / max_latency))
        tok_score = max(0.0, 1.0 - (tokens_used / max_tokens))
        
        reward = self.alpha * quality + self.beta * lat_score + self.gamma * tok_score
        
        return {
            'composite_reward': reward,
            'quality_score': quality,
            'latency_score': lat_score,
            'token_score': tok_score
        }
        
    def compute_energy_estimate(self, result_dict: dict) -> float:
        """Estimate energy consumption in Joules for the query execution.
        Based on typical GPU/CPU energy profiles:
        - CPU retrieval/processing: ~15W active power.
        - GPU inference (if local or estimated via tokens):
          - Embedding: ~0.05 Joules/token
          - Reranking: ~0.15 Joules/token
          - Generation: ~0.50 Joules/token
        - LLM generation: 0.3 Joules per generated token.
        """
        latency_sec = result_dict.get('latency_ms', 0.0) / 1000.0
        
        # Base CPU overhead (idle/active overhead during request lifetime)
        energy_retrieval_processing = latency_sec * 15.0  # 15 Watts
        
        # Token-based compute estimation
        generation_tokens = result_dict.get('generation_stats', {}).get('generated_tokens', 100)
        context_tokens = result_dict.get('acoa_stats', {}).get('total_tokens', 256)
        
        # GPU active execution estimations
        energy_embedding = context_tokens * 0.01
        energy_generation = generation_tokens * 0.25
        
        total_joules = energy_retrieval_processing + energy_embedding + energy_generation
        return max(0.01, round(total_joules, 4))
        
    def get_pareto_metrics(self, results_list: list) -> dict:
        """Find the Pareto-optimal plans/runs from a list of query execution results.
        Optimizes for:
        - Maximize quality (or confidence)
        - Minimize latency
        - Minimize energy (or token usage)
        
        A point A dominates B if A is no worse than B in all metrics and strictly better in at least one.
        """
        if not results_list:
            return {"pareto_optimal_count": 0, "pareto_indices": []}
            
        pareto_indices = []
        for i, res_a in enumerate(results_list):
            qa = res_a.get('confidence', {}).get('score', 0.5)
            la = res_a.get('latency_ms', 100.0)
            ea = self.compute_energy_estimate(res_a)
            
            dominated = False
            for j, res_b in enumerate(results_list):
                if i == j:
                    continue
                qb = res_b.get('confidence', {}).get('score', 0.5)
                lb = res_b.get('latency_ms', 100.0)
                eb = self.compute_energy_estimate(res_b)
                
                # Check dominance (B dominates A)
                # B is better than or equal in quality, latency, energy
                if qb >= qa and lb <= la and eb <= ea:
                    # and strictly better in at least one
                    if qb > qa or lb < la or eb < ea:
                        dominated = True
                        break
                        
            if not dominated:
                pareto_indices.append(i)
                
        return {
            "pareto_optimal_count": len(pareto_indices),
            "pareto_indices": pareto_indices
        }
