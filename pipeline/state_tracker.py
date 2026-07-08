"""
State Tracker Module
====================
Manages execution state, token budgets, latency, and scratchpad memory for the multi-step RAG loop.
"""
import time
import logging
from typing import List, Dict, Set

logger = logging.getLogger(__name__)

class StateTracker:
    def __init__(self, query: str, token_budget: int = 2048, max_steps: int = 4):
        self.query = query
        self.max_steps = max_steps
        self.start_time = time.perf_counter()
        
        # Resource tracking
        self.initial_budget = token_budget
        self.remaining_token_budget = token_budget
        self.elapsed_latency_ms = 0.0
        self.step_count = 0
        
        # Scratchpad and Memory tracking
        self.scratchpad: List[Dict[str, str]] = []
        self.retrieved_chunks: List[Dict] = []
        self.retrieved_hashes: Set[str] = set()

    def add_step(self, reasoning: str, action_type: str, action_query: str, observation: str):
        """Record a single step of the reasoning loop."""
        self.step_count += 1
        self.elapsed_latency_ms = (time.perf_counter() - self.start_time) * 1000
        
        self.scratchpad.append({
            "step": self.step_count,
            "reasoning": reasoning,
            "action_type": action_type,
            "action_query": action_query,
            "observation": observation,
            "tokens_left": self.remaining_token_budget,
            "latency_ms": round(self.elapsed_latency_ms, 2)
        })

    def deduct_tokens(self, tokens: int):
        """Deduct tokens consumed by generation or retrieval from the budget."""
        self.remaining_token_budget = max(0, self.remaining_token_budget - tokens)

    def add_chunks(self, chunks: List[Dict], token_cost: int = 0) -> int:
        """Add chunks to retrieved memory, filtering out duplicates."""
        new_added = 0
        for item in chunks:
            chunk = item.get("chunk", item)
            text = chunk.get("text", "")
            chunk_hash = str(hash(text))
            
            if chunk_hash not in self.retrieved_hashes:
                self.retrieved_hashes.add(chunk_hash)
                # Keep score from original retrieval result if present
                self.retrieved_chunks.append(item)
                new_added += 1
                
        self.deduct_tokens(token_cost)
        return new_added

    def to_text_history(self) -> str:
        """Format the execution trace as text for the LLM coordinator."""
        if not self.scratchpad:
            return "No previous steps taken."
            
        lines = []
        for step in self.scratchpad:
            lines.append(f"Step {step['step']}:")
            lines.append(f"  Reasoning: {step['reasoning']}")
            lines.append(f"  Action: {step['action_type']} | Query: '{step['action_query']}'")
            lines.append(f"  Observation: {step['observation']}")
            lines.append(f"  Resource Status: {step['tokens_left']} tokens left, {step['latency_ms']:.1f}ms elapsed")
            lines.append("")
        return "\n".join(lines)

    def should_terminate(self) -> bool:
        """Check if termination conditions (step limit or budget exhaustion) are met."""
        if self.step_count >= self.max_steps:
            logger.info("Maximum reasoning steps reached (%d). Terminating loop.", self.max_steps)
            return True
        if self.remaining_token_budget <= 150:  # threshold to guarantee generation budget
            logger.info("Token budget depleted (%d). Terminating loop.", self.remaining_token_budget)
            return True
        return False
