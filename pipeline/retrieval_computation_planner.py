"""
Retrieval Computation Planner & ReAct Coordinator
=================================================
Implements:
1. RetrievalComputationPlanner: Contextual Bandit (LinUCB) for single-step RAG v2 planning.
2. ReActCoordinator: LLM-based multi-step decision controller for RAG v3 constraint-compliant planning.
"""
import sys
import logging
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from pipeline.execution_plan import ExecutionPlan
from pipeline.state_tracker import StateTracker

logger = logging.getLogger(__name__)

COORDINATOR_PROMPT_TEMPLATE = """You are the LLM Reasoning Coordinator for a Multi-Objective Resource-Constrained RAG system.
Your job is to solve a user's query by generating sub-queries and selecting the most appropriate retrieval method (Lexical BM25 or Dense Semantic) while staying under a strict token budget.

User Query: {query}
Remaining Token Budget: {remaining_budget}
Maximum Reasoning Steps: {max_steps}
Current Step: {current_step}

Prior Reasoning Steps and Observations:
{history}

Decide the next action. You can select one of the following:
1. "RETRIEVE_LEXICAL": Select this if you need to query exact keywords, names, entities, identifiers, or dates. Best for factual searches (Low token cost).
2. "RETRIEVE_DENSE": Select this if you need to query concepts, definitions, broad themes, or synthesize explanations (Medium token cost).
3. "TERMINATE_AND_GENERATE": Select this if the retrieved information is complete and you have enough context to answer the user query, or if you are running out of budget.

Output your decision strictly as a JSON object with keys: "reasoning" (string explaining your thoughts), "action" (one of the three action strings above), and "query" (the sub-query string to search, or empty if terminating).
JSON Response:"""

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


class ReActCoordinator:
    """Orchestrates RAG v3 multi-step planning using LLM-in-the-loop and fallback heuristics."""
    
    def __init__(self, generator=None):
        self.generator = generator

    def decide_action(self, state: StateTracker) -> dict:
        """Decide the next action in the reasoning loop.
        
        Tries to call the LLM generator if available, otherwise falls back to a deterministic heuristic.
        """
        if self.generator and self.generator.llm and not getattr(self.generator, 'backend', '') == 'ollama_offline_stub':
            try:
                prompt = COORDINATOR_PROMPT_TEMPLATE.format(
                    query=state.query,
                    remaining_budget=state.remaining_token_budget,
                    max_steps=state.max_steps,
                    current_step=state.step_count + 1,
                    history=state.to_text_history()
                )
                response_text = self.generator._call_llm(prompt)
                
                # Try to parse JSON from LLM output
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    data = json.loads(response_text[json_start:json_end])
                    # Validate keys
                    if "action" in data and "query" in data and "reasoning" in data:
                        logger.info("ReAct LLM Coordinator decided: %s ('%s')", data["action"], data["query"])
                        return data
                
                logger.warning("LLM response did not contain valid RAG coordinator JSON: %s", response_text)
            except Exception as e:
                logger.warning("Error invoking LLM RAG coordinator: %s. Falling back to heuristics.", e)
                
        # Heuristic fallback
        return self._heuristic_decide(state)

    def _heuristic_decide(self, state: StateTracker) -> dict:
        """Determinstic rule-based decision loop for offline or backup execution."""
        query_lower = state.query.lower()
        step = state.step_count + 1
        
        if step == 1:
            # Fact lookup indicators
            fact_keywords = ["gdp", "nominal", "who", "when", "year", "date", "president", "india", "reform", "reformative", "age"]
            has_entities = any(char.isupper() for char in state.query if char.isalpha()) or any(char.isdigit() for char in state.query)
            
            if has_entities or any(kw in query_lower for kw in fact_keywords):
                return {
                    "reasoning": "Query contains factual or entity keywords. Activating Exploitation Agent (Lexical search) for exact lookup.",
                    "action": "RETRIEVE_LEXICAL",
                    "query": state.query
                }
            else:
                return {
                    "reasoning": "Query appears conceptual or open-ended. Activating Exploration Agent (Dense search) for semantic context.",
                    "action": "RETRIEVE_DENSE",
                    "query": state.query
                }
        elif step == 2:
            if len(state.retrieved_chunks) > 0:
                return {
                    "reasoning": "Step 1 retrieved relevant document chunks. Sufficient context accumulated. Terminating reasoning loop.",
                    "action": "TERMINATE_AND_GENERATE",
                    "query": ""
                }
            else:
                # Try the alternate retriever
                prev_action = state.scratchpad[0]["action_type"]
                next_action = "RETRIEVE_DENSE" if prev_action == "RETRIEVE_LEXICAL" else "RETRIEVE_LEXICAL"
                reasoning = "Initial retrieval returned no chunks. Activating alternate agent to cover the search space."
                
                return {
                    "reasoning": reasoning,
                    "action": next_action,
                    "query": state.query
                }
        else:
            return {
                "reasoning": "Loop threshold reached or sufficient context loaded. Terminating to compile final generation context.",
                "action": "TERMINATE_AND_GENERATE",
                "query": ""
            }
