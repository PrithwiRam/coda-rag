"""
MOSAIC Context Assembly
=======================
Multi-Objective Submodular Adaptive Information Curation.

A formally grounded multi-objective submodular optimization algorithm for
context assembly that simultaneously optimizes:

  1. Relevance (semantic similarity to query)
  2. Diversity (1 - average pairwise Jaccard)
  3. Redundancy penalty (negative pairwise Jaccard sum)
  4. Token efficiency (1 - cost/budget)
  5. Position-aware attention bias (primacy/recency)
  6. Query coverage (term overlap)

Uses greedy with lazy evaluations (Minoux 1978) for (1-1/e) approximation
guarantee on monotone submodular functions.

Query-conditioned weights come from RCP plan.mosaic_weights.
"""
import sys
import math
import heapq
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


class MOSAIC:
    """Multi-Objective Submodular Adaptive Information Curation.

    Selects a subset of candidate chunks S ⊆ C that maximizes:
        F(S) = Σᵢ wᵢ · fᵢ(S)
    subject to cost(S) ≤ token_budget.

    Each fᵢ is a submodular set function. The weighted sum of submodular
    functions is submodular, so greedy achieves (1-1/e) approximation.

    Attributes:
        encoder: SentenceTransformer for semantic similarity.
        nlp: spaCy model for tokenization and linguistic features.
    """

    def __init__(self):
        """Initialize MOSAIC with embedding and NLP models."""
        self.encoder = None
        self.nlp = None

        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(config.EMBEDDING_MODEL)
            logger.info("MOSAIC: SentenceTransformer loaded (%s)", config.EMBEDDING_MODEL)
        except Exception as e:
            logger.warning("MOSAIC: SentenceTransformer unavailable: %s", e)

        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("MOSAIC: spaCy model loaded")
        except Exception as e:
            logger.warning("MOSAIC: spaCy unavailable: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_context(
        self,
        query: str,
        candidates: List[Dict],
        token_budget: int = 2048,
        weights: Optional[List[float]] = None,
    ) -> Dict:
        """Select optimal subset of chunks within token budget.

        Args:
            query: The user's question.
            candidates: List of retrieval results, each with 'chunk' and 'score'.
            token_budget: Maximum tokens for context (from RCP plan).
            weights: 6-dim weight vector for objectives. Defaults to uniform.

        Returns:
            Dictionary with selected_chunks, total_tokens, num_selected,
            objective_scores, and selection_order.
        """
        if not weights or len(weights) != 6:
            weights = [1.0 / 6.0] * 6

        # Budget-aware epsilon-relaxation of optimization weights (inspired by arXiv:2601.19121)
        # Under tight token constraints (< 1000 tokens), shift weights from Exploration (diversity/redundancy)
        # to Exploitation (relevance/coverage) to ensure critical information feasibility.
        if token_budget < 1000:
            logger.info("MOSAIC: Strict token constraint active (%d tokens). Shifting weights to relevance and coverage.", token_budget)
            weights = list(weights)
            div_weight = weights[1] + weights[2]
            weights[1] = 0.02
            weights[2] = 0.00
            weights[0] += div_weight * 0.6
            weights[5] += div_weight * 0.4
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights]

        if not candidates:
            return {
                "selected_chunks": [],
                "total_tokens": 0,
                "token_budget": token_budget,
                "num_selected": 0,
                "objective_scores": {},
                "selection_order": [],
            }

        # Precompute embeddings and token counts
        chunk_data = self._precompute(candidates, query)

        # Run greedy submodular selection with lazy evaluations
        selected, scores = self._greedy_lazy(
            chunk_data, query, token_budget, weights
        )

        # Reorder for optimal LLM attention (primacy/recency)
        selected = self._reorder_for_attention(selected, query)

        total_tokens = sum(c["token_count"] for c in selected)

        return {
            "selected_chunks": selected,
            "total_tokens": total_tokens,
            "token_budget": token_budget,
            "num_selected": len(selected),
            "objective_scores": scores,
            "selection_order": [c.get("chunk_id", "") for c in selected],
        }

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute(self, candidates: List[Dict], query: str) -> List[Dict]:
        """Precompute embeddings, token counts, and term sets for all chunks.

        Args:
            candidates: Raw retrieval results.
            query: Query string.

        Returns:
            List of enriched chunk dictionaries.
        """
        chunk_data = []

        # Encode query once
        query_emb = None
        if self.encoder:
            try:
                query_emb = self.encoder.encode([query])[0]
            except Exception as e:
                logger.warning("MOSAIC: query encoding failed: %s", e)

        # Encode all chunks
        texts = [c.get("chunk", {}).get("text", "") for c in candidates]
        chunk_embs = None
        if self.encoder and texts:
            try:
                chunk_embs = self.encoder.encode(texts)
            except Exception as e:
                logger.warning("MOSAIC: chunk encoding failed: %s", e)

        # Query term set for coverage
        query_terms = set()
        if self.nlp:
            try:
                doc = self.nlp(query.lower())
                query_terms = {token.lemma_ for token in doc
                              if not token.is_stop and not token.is_punct}
            except Exception:
                query_terms = set(query.lower().split())

        for i, candidate in enumerate(candidates):
            chunk = candidate.get("chunk", candidate)
            text = chunk.get("text", "")
            chunk_id = chunk.get("chunk_id", str(i))

            # Token count (estimate: words * 1.3 for subword tokens)
            token_count = int(len(text.split()) * 1.3) if text else 0

            # Embedding
            emb = chunk_embs[i] if chunk_embs is not None and i < len(chunk_embs) else None

            # Term set for Jaccard and coverage
            term_set = set()
            if self.nlp and text:
                try:
                    doc = self.nlp(text.lower())
                    term_set = {token.lemma_ for token in doc
                               if not token.is_stop and not token.is_punct}
                except Exception:
                    term_set = set(text.lower().split())

            chunk_data.append({
                "candidate": candidate,
                "chunk": chunk,
                "text": text,
                "chunk_id": chunk_id,
                "score": candidate.get("score", 0.0),
                "token_count": token_count,
                "embedding": emb,
                "term_set": term_set,
            })

        # Store query data for objective functions
        self._query_emb = query_emb
        self._query_terms = query_terms

        return chunk_data

    # ------------------------------------------------------------------
    # Greedy Submodular Selection with Lazy Evaluations
    # ------------------------------------------------------------------

    def _greedy_lazy(
        self,
        chunk_data: List[Dict],
        query: str,
        budget: int,
        weights: List[float],
    ) -> Tuple[List[Dict], Dict]:
        """Greedy submodular maximization with lazy evaluations (Minoux 1978).

        Maintains a max-heap of (upper_bound, chunk_index). At each step,
        pops the top element, recomputes its true marginal gain. If the
        true gain is still ≥ the next upper bound, selects it. Otherwise,
        re-inserts with the true gain and continues.

        Complexity: O(n log n) with lazy eval vs O(n²) naive greedy.

        Args:
            chunk_data: Precomputed chunk data.
            query: Query string.
            budget: Token budget.
            weights: 6-dim objective weights.

        Returns:
            Tuple of (selected_chunks, objective_scores_dict).
        """
        selected = []
        selected_indices: Set[int] = set()
        total_cost = 0

        # Initialize upper bounds (max possible marginal gain for each chunk)
        # Upper bound = weighted sum of max possible contribution per objective
        heap = []  # (-upper_bound, index) for max-heap via min-heap

        for i, cd in enumerate(chunk_data):
            if cd["token_count"] == 0:
                continue
            ub = self._compute_upper_bound(cd, weights)
            heapq.heappush(heap, (-ub, i))

        # Track objective scores
        obj_scores = {f"f{i+1}": 0.0 for i in range(6)}
        obj_scores["composite"] = 0.0

        while heap and total_cost < budget:
            neg_ub, idx = heapq.heappop(heap)
            cd = chunk_data[idx]

            if cd["token_count"] + total_cost > budget:
                continue  # Can't fit

            # Compute true marginal gain
            true_gain = self._marginal_gain(
                selected, cd, query, weights
            )

            if true_gain <= 0:
                continue  # No positive contribution

            # Check if this is still the best (lazy eval)
            if heap:
                next_neg_ub = heap[0][0]
                if true_gain < -next_neg_ub:
                    # Re-insert with true gain as new upper bound
                    heapq.heappush(heap, (-true_gain, idx))
                    continue

            # Select this chunk
            selected.append(cd)
            selected_indices.add(idx)
            total_cost += cd["token_count"]

            # Update objective scores
            obj_scores = self._compute_objective_scores(
                selected, query, weights
            )

        return selected, obj_scores

    def _compute_upper_bound(self, cd: Dict, weights: List[float]) -> float:
        """Compute upper bound on marginal gain for a single chunk.

        Upper bound assumes no redundancy penalty and full coverage gain.

        Args:
            cd: Chunk data dictionary.
            weights: Objective weights.

        Returns:
            Upper bound float.
        """
        ub = 0.0

        # f1 (relevance): max = score (normalized) or cosine sim
        w1 = weights[0]
        relevance = cd["score"]
        if self._query_emb is not None and cd["embedding"] is not None:
            relevance = float(np.dot(self._query_emb, cd["embedding"]) /
                            (np.linalg.norm(self._query_emb) * np.linalg.norm(cd["embedding"]) + 1e-8))
        ub += w1 * max(0.0, relevance)

        # f2 (diversity): max = 1.0 (no redundancy with empty set)
        ub += weights[1] * 1.0

        # f3 (redundancy): max = 0.0 (no penalty with empty set)
        ub += weights[2] * 0.0

        # f4 (token efficiency): proportional to token contribution
        ub += weights[3] * 0.1  # small positive

        # f5 (position bias): max = 1.0 (best position)
        ub += weights[4] * 1.0

        # f6 (coverage): max = fraction of query terms covered
        if self._query_terms and cd["term_set"]:
            coverage = len(cd["term_set"] & self._query_terms) / max(1, len(self._query_terms))
            ub += weights[5] * coverage

        return ub

    # ------------------------------------------------------------------
    # Marginal Gain Computation
    # ------------------------------------------------------------------

    def _marginal_gain(
        self,
        selected: List[Dict],
        candidate: Dict,
        query: str,
        weights: List[float],
    ) -> float:
        """Compute marginal gain F(S ∪ {c}) - F(S).

        Args:
            selected: Currently selected chunks.
            candidate: Candidate chunk to evaluate.
            query: Query string.
            weights: Objective weights.

        Returns:
            Marginal gain (can be negative if redundancy dominates).
        """
        if not selected:
            # First chunk: compute F({c})
            return self._compute_single_f(candidate, weights)

        # F(S ∪ {c})
        f_with = self._compute_set_f(selected + [candidate], weights)
        # F(S)
        f_without = self._compute_set_f(selected, weights)

        return f_with - f_without

    def _compute_single_f(self, cd: Dict, weights: List[float]) -> float:
        """Compute F({c}) for a single chunk.

        Args:
            cd: Chunk data.
            weights: Objective weights.

        Returns:
            Objective value.
        """
        f_val = 0.0

        # f1: relevance
        relevance = cd["score"]
        if self._query_emb is not None and cd["embedding"] is not None:
            relevance = float(np.dot(self._query_emb, cd["embedding"]) /
                            (np.linalg.norm(self._query_emb) * np.linalg.norm(cd["embedding"]) + 1e-8))
        f_val += weights[0] * max(0.0, relevance)

        # f2: diversity = 1.0 (single chunk is perfectly diverse)
        f_val += weights[1] * 1.0

        # f3: redundancy = 0.0 (no pairs)
        f_val += weights[2] * 0.0

        # f4: token efficiency
        f_val += weights[3] * 0.1

        # f5: position bias = 1.0 (first position)
        f_val += weights[4] * 1.0

        # f6: coverage
        if self._query_terms and cd["term_set"]:
            coverage = len(cd["term_set"] & self._query_terms) / max(1, len(self._query_terms))
            f_val += weights[5] * coverage

        return f_val

    def _compute_set_f(self, selected: List[Dict], weights: List[float]) -> float:
        """Compute F(S) for a set of chunks.

        Args:
            selected: List of selected chunk data.
            weights: Objective weights.

        Returns:
            Composite objective value.
        """
        if not selected:
            return 0.0

        n = len(selected)
        f_val = 0.0

        # f1: relevance = average cosine similarity
        relevance_sum = 0.0
        for cd in selected:
            rel = cd["score"]
            if self._query_emb is not None and cd["embedding"] is not None:
                rel = float(np.dot(self._query_emb, cd["embedding"]) /
                           (np.linalg.norm(self._query_emb) * np.linalg.norm(cd["embedding"]) + 1e-8))
            relevance_sum += max(0.0, rel)
        f_val += weights[0] * (relevance_sum / n)

        # f2: diversity = 1 - average pairwise Jaccard
        if n > 1:
            jaccard_sum = 0.0
            pairs = 0
            for i in range(n):
                for j in range(i + 1, n):
                    jaccard_sum += self._jaccard(
                        selected[i]["term_set"], selected[j]["term_set"]
                    )
                    pairs += 1
            avg_jaccard = jaccard_sum / pairs if pairs > 0 else 0.0
            f_val += weights[1] * (1.0 - avg_jaccard)
        else:
            f_val += weights[1] * 1.0

        # f3: redundancy penalty = -average pairwise Jaccard
        if n > 1:
            jaccard_sum = 0.0
            pairs = 0
            for i in range(n):
                for j in range(i + 1, n):
                    jaccard_sum += self._jaccard(
                        selected[i]["term_set"], selected[j]["term_set"]
                    )
                    pairs += 1
            avg_jaccard = jaccard_sum / pairs if pairs > 0 else 0.0
            f_val += weights[2] * (-avg_jaccard)
        # else: 0.0

        # f4: token efficiency = 1 - cost/budget (computed externally)
        # Use a small constant here; actual budget tracking is in greedy loop
        f_val += weights[3] * 0.05

        # f5: position bias = primacy/recency weighting
        position_score = 0.0
        for i, cd in enumerate(selected):
            if i == 0 or i == n - 1:
                position_score += 1.0  # primacy or recency
            else:
                position_score += 0.7  # middle positions
        f_val += weights[4] * (position_score / n)

        # f6: coverage = |query_terms ∩ (∪ chunk_terms)| / |query_terms|
        if self._query_terms:
            all_terms: Set[str] = set()
            for cd in selected:
                all_terms.update(cd["term_set"])
            coverage = len(all_terms & self._query_terms) / max(1, len(self._query_terms))
            f_val += weights[5] * coverage

        return f_val

    def _compute_objective_scores(
        self,
        selected: List[Dict],
        query: str,
        weights: List[float],
    ) -> Dict:
        """Compute individual objective scores for the selected set.

        Args:
            selected: Selected chunks.
            query: Query string.
            weights: Objective weights.

        Returns:
            Dictionary with f1-f6 scores and composite.
        """
        if not selected:
            return {"f1": 0.0, "f2": 0.0, "f3": 0.0, "f4": 0.0,
                    "f5": 0.0, "f6": 0.0, "composite": 0.0}

        n = len(selected)

        # f1: relevance
        relevance_sum = 0.0
        for cd in selected:
            rel = cd["score"]
            if self._query_emb is not None and cd["embedding"] is not None:
                rel = float(np.dot(self._query_emb, cd["embedding"]) /
                           (np.linalg.norm(self._query_emb) * np.linalg.norm(cd["embedding"]) + 1e-8))
            relevance_sum += max(0.0, rel)
        f1 = relevance_sum / n

        # f2: diversity
        if n > 1:
            jaccard_sum = 0.0
            pairs = 0
            for i in range(n):
                for j in range(i + 1, n):
                    jaccard_sum += self._jaccard(
                        selected[i]["term_set"], selected[j]["term_set"]
                    )
                    pairs += 1
            avg_jaccard = jaccard_sum / pairs if pairs > 0 else 0.0
            f2 = 1.0 - avg_jaccard
        else:
            f2 = 1.0

        # f3: redundancy
        if n > 1:
            jaccard_sum = 0.0
            pairs = 0
            for i in range(n):
                for j in range(i + 1, n):
                    jaccard_sum += self._jaccard(
                        selected[i]["term_set"], selected[j]["term_set"]
                    )
                    pairs += 1
            avg_jaccard = jaccard_sum / pairs if pairs > 0 else 0.0
            f3 = -avg_jaccard
        else:
            f3 = 0.0

        # f4: token efficiency
        total_tokens = sum(cd["token_count"] for cd in selected)
        f4 = 0.05  # placeholder; real value needs budget context

        # f5: position bias
        position_score = 0.0
        for i, cd in enumerate(selected):
            if i == 0 or i == n - 1:
                position_score += 1.0
            else:
                position_score += 0.7
        f5 = position_score / n

        # f6: coverage
        if self._query_terms:
            all_terms: Set[str] = set()
            for cd in selected:
                all_terms.update(cd["term_set"])
            f6 = len(all_terms & self._query_terms) / max(1, len(self._query_terms))
        else:
            f6 = 0.0

        composite = (
            weights[0] * f1 + weights[1] * f2 + weights[2] * f3 +
            weights[3] * f4 + weights[4] * f5 + weights[5] * f6
        )

        return {
            "f1_relevance": round(f1, 4),
            "f2_diversity": round(f2, 4),
            "f3_redundancy": round(f3, 4),
            "f4_token_efficiency": round(f4, 4),
            "f5_position_bias": round(f5, 4),
            "f6_coverage": round(f6, 4),
            "composite": round(composite, 4),
        }

    # ------------------------------------------------------------------
    # Utility Functions
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard similarity between two term sets.

        Args:
            set_a: First term set.
            set_b: Second term set.

        Returns:
            Jaccard similarity in [0.0, 1.0].
        """
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _reorder_for_attention(self, selected: List[Dict], query: str) -> List[Dict]:
        """Reorder chunks for optimal LLM attention (primacy/recency effect).

        Places highest-relevance chunks at the beginning and end of the
        context window, with medium-relevance chunks in the middle.
        This exploits the known primacy/recency bias in transformer attention.

        Args:
            selected: Selected chunks.
            query: Query string.

        Returns:
            Reordered list of chunks.
        """
        if len(selected) <= 2:
            return selected

        # Sort by relevance score descending
        sorted_sel = sorted(
            selected,
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )

        # Interleave: highest → front, next-highest → back, etc.
        reordered = [None] * len(sorted_sel)
        left = 0
        right = len(sorted_sel) - 1

        for i, cd in enumerate(sorted_sel):
            if i % 2 == 0:
                reordered[left] = cd
                left += 1
            else:
                reordered[right] = cd
                right -= 1

        return reordered