"""
ACOA Module – Adaptive Context Optimization Algorithm
=====================================================

Implements the core ACOA algorithm from the ACORA proposal.  Given a ranked
list of retrieved (and optionally reranked) chunks, ACOA selects the optimal
subset that:

  1. Fits within a strict **token budget** (default 2048 tokens).
  2. Maximises **information density** (entities + noun-phrases per token).
  3. Minimises **redundancy** via Jaccard similarity filtering.

The selected chunks are then sorted by their original document order
(``chunk_index``) so the final context reads naturally.

Algorithm pseudo-code::

    S ← [], tokens_used ← 0
    Sort candidates by relevance score (descending)
    For each candidate c:
        density   ← info_density(c)
        redundancy ← max_jaccard(c, S)
        acoa_score ← relevance × density × (1 − redundancy)
        if redundancy > threshold:  skip
        if tokens_used + c.token_count ≤ budget:
            S ← S ∪ {c};  tokens_used += c.token_count
        else: break
    Sort S by chunk_index (document order)
    Return S

Usage:
    from pipeline.acoa import ACOA

    acoa = ACOA()
    result = acoa.optimize(reranked_results)
    context = acoa.build_context(result['selected_chunks'])
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Set

# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class ACOA:
    """Adaptive Context Optimization Algorithm.

    Selects a non-redundant, information-dense subset of retrieval results
    that fits within a specified token budget.

    Attributes:
        token_budget (int): Maximum number of tokens allowed in the
            assembled context.
        redundancy_threshold (float): Jaccard similarity above which a
            candidate chunk is considered redundant and skipped.
        nlp: spaCy language model for entity / noun-phrase extraction.
        encoding: tiktoken encoding for accurate token counting.
    """

    def __init__(
        self,
        token_budget: int = config.ACOA_TOKEN_BUDGET,
        redundancy_threshold: float = config.ACOA_REDUNDANCY_THRESHOLD,
    ) -> None:
        """Initialise the ACOA optimizer.

        Args:
            token_budget: Maximum tokens for the assembled context.
            redundancy_threshold: Jaccard similarity threshold above which
                a chunk is considered redundant (default from config).
        """
        self.token_budget = token_budget
        self.redundancy_threshold = redundancy_threshold

        # ------------------------------------------------------------------
        # Load spaCy model for entity / noun-phrase extraction
        # ------------------------------------------------------------------
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Falling back to blank English model. "
                    "Run: python -m spacy download en_core_web_sm"
                )
                self.nlp = spacy.blank("en")
        except ImportError:
            logger.warning(
                "spaCy not installed. Information density computation "
                "will use a fallback heuristic."
            )
            self.nlp = None

        # ------------------------------------------------------------------
        # Load tiktoken for accurate token counting
        # ------------------------------------------------------------------
        import tiktoken
        self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.info(
            "ACOA initialised  |  budget=%d tokens  redundancy_threshold=%.2f",
            self.token_budget,
            self.redundancy_threshold,
        )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def optimize(self, results: List[Dict]) -> Dict:
        """Run the ACOA algorithm on ranked retrieval results.

        Args:
            results: List of retrieval result dictionaries. Each should have:
                - ``chunk`` (dict): with ``text`` (str), ``token_count`` (int),
                  and optionally ``chunk_index`` (int).
                - ``score`` or ``rerank_score`` (float): relevance score.

        Returns:
            Dictionary containing::

                {
                    "selected_chunks":       List[Dict],
                    "total_tokens":          int,
                    "num_selected":          int,
                    "num_skipped_redundant": int,
                    "optimization_stats": {
                        "token_budget":          int,
                        "redundancy_threshold":  float,
                        "candidates_considered": int,
                        "avg_density":           float,
                        "avg_acoa_score":        float,
                    }
                }
        """
        if not results:
            logger.warning("ACOA.optimize called with empty results.")
            return {
                "selected_chunks": [],
                "total_tokens": 0,
                "num_selected": 0,
                "num_skipped_redundant": 0,
                "optimization_stats": {},
            }

        # ------------------------------------------------------------------
        # Step 1: Initialise
        # ------------------------------------------------------------------
        selected: List[Dict] = []           # S
        selected_texts: List[str] = []      # text cache for redundancy checks
        tokens_used = 0
        num_skipped_redundant = 0
        density_values: List[float] = []
        acoa_scores: List[float] = []

        # ------------------------------------------------------------------
        # Step 2: Sort candidates by relevance score (descending)
        # ------------------------------------------------------------------
        sorted_results = sorted(
            results,
            key=lambda r: float(r.get("rerank_score", r.get("score", 0.0))),
            reverse=True,
        )

        # ------------------------------------------------------------------
        # Step 3: Iterate and apply ACOA selection
        # ------------------------------------------------------------------
        for result in sorted_results:
            chunk = result.get("chunk", {})
            chunk_text = chunk.get("text", "")
            token_count = chunk.get("token_count", self._count_tokens(chunk_text))
            relevance = float(result.get("rerank_score", result.get("score", 0.0)))

            # (a) Compute information density
            density = self._compute_density(chunk_text)
            density_values.append(density)

            # (b) Compute redundancy against already-selected chunks
            redundancy = self._compute_redundancy(chunk_text, selected_texts)

            # (c) Compute composite ACOA score
            # Ensure relevance is non-negative for multiplication
            norm_relevance = max(relevance, 0.0)
            acoa_score = norm_relevance * density * (1.0 - redundancy)
            acoa_scores.append(acoa_score)

            # (d) Skip if too redundant
            if redundancy > self.redundancy_threshold:
                num_skipped_redundant += 1
                logger.debug(
                    "ACOA: skipping chunk (redundancy=%.3f > threshold=%.3f)",
                    redundancy, self.redundancy_threshold,
                )
                continue

            # (e) Check token budget
            if tokens_used + token_count <= self.token_budget:
                # Add chunk with its ACOA metadata
                selected_entry = {
                    "chunk": chunk,
                    "score": relevance,
                    "acoa_score": acoa_score,
                    "density": density,
                    "redundancy": redundancy,
                }
                selected.append(selected_entry)
                selected_texts.append(chunk_text)
                tokens_used += token_count
            else:
                # (f) Budget exhausted – stop
                logger.debug(
                    "ACOA: budget exhausted (used=%d, needed=%d, budget=%d)",
                    tokens_used, token_count, self.token_budget,
                )
                break

        # ------------------------------------------------------------------
        # Step 4: Sort selected chunks by original document position
        # ------------------------------------------------------------------
        selected.sort(
            key=lambda s: s["chunk"].get("chunk_index", 0)
        )

        # ------------------------------------------------------------------
        # Step 5: Build optimisation statistics
        # ------------------------------------------------------------------
        avg_density = (
            sum(density_values) / len(density_values) if density_values else 0.0
        )
        avg_acoa_score = (
            sum(acoa_scores) / len(acoa_scores) if acoa_scores else 0.0
        )

        stats = {
            "selected_chunks": selected,
            "total_tokens": tokens_used,
            "num_selected": len(selected),
            "num_skipped_redundant": num_skipped_redundant,
            "optimization_stats": {
                "token_budget": self.token_budget,
                "redundancy_threshold": self.redundancy_threshold,
                "candidates_considered": len(results),
                "avg_density": round(avg_density, 4),
                "avg_acoa_score": round(avg_acoa_score, 4),
            },
        }

        logger.info(
            "ACOA complete  |  selected=%d/%d  tokens=%d/%d  skipped_redundant=%d",
            len(selected), len(results), tokens_used,
            self.token_budget, num_skipped_redundant,
        )

        return stats

    def build_context(self, selected_chunks: List[Dict]) -> str:
        """Join selected chunks into a single context string.

        Chunks are joined with a visual separator to help the LLM
        distinguish between different source passages.

        Args:
            selected_chunks: List of selected chunk dictionaries from
                ``optimize()``. Each should have a ``chunk`` key with ``text``.

        Returns:
            Formatted context string ready for the LLM prompt.
        """
        separator = "\n\n---\n\n"
        texts = []
        for entry in selected_chunks:
            chunk = entry.get("chunk", {})
            text = chunk.get("text", "")
            if text:
                texts.append(text)

        context = separator.join(texts)
        logger.debug(
            "Built context from %d chunks  |  total_chars=%d",
            len(texts), len(context),
        )
        return context

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _compute_density(self, text: str) -> float:
        """Compute information density of a text chunk.

        Information density is defined as:
            density = (num_entities + num_noun_phrases) / max(num_tokens, 1)

        The result is capped at 1.0 to normalise into [0, 1].

        Args:
            text: The chunk text.

        Returns:
            Information density score in [0.0, 1.0].
        """
        if not text:
            return 0.0

        tokens = self._count_tokens(text)

        if self.nlp is not None:
            # Use spaCy for entity and noun-phrase extraction
            doc = self.nlp(text)
            num_entities = len(doc.ents)
            num_noun_phrases = len(list(doc.noun_chunks))
        else:
            # Fallback heuristic: count capitalised word sequences and
            # simple noun-phrase patterns
            words = text.split()
            num_entities = sum(
                1 for w in words
                if w[0].isupper() and len(w) > 1 and w.isalpha()
            )
            num_noun_phrases = max(num_entities // 2, 1)

        density = (num_entities + num_noun_phrases) / max(tokens, 1)

        # Cap at 1.0 for normalisation
        return min(density, 1.0)

    def _compute_redundancy(self, text: str, selected: List[str]) -> float:
        """Compute redundancy of a candidate chunk against selected chunks.

        Redundancy is the maximum Jaccard similarity between the candidate's
        word set and any already-selected chunk's word set.

        Args:
            text: Candidate chunk text.
            selected: List of already-selected chunk texts.

        Returns:
            Maximum Jaccard similarity in [0.0, 1.0].  Returns 0.0 if
            *selected* is empty.
        """
        if not selected:
            return 0.0

        candidate_words = self._get_word_set(text)

        max_similarity = 0.0
        for sel_text in selected:
            sel_words = self._get_word_set(sel_text)
            sim = self._jaccard_similarity(candidate_words, sel_words)
            if sim > max_similarity:
                max_similarity = sim

        return max_similarity

    def _jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard similarity between two sets.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|

        Args:
            set_a: First set of words.
            set_b: Second set of words.

        Returns:
            Jaccard similarity in [0.0, 1.0].
        """
        if not set_a and not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0

    def _get_word_set(self, text: str) -> Set[str]:
        """Extract a normalised word set from text.

        Args:
            text: Input text.

        Returns:
            Set of lowercased words (alphanumeric only).
        """
        return {w.lower() for w in text.split() if w.isalnum()}

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string using tiktoken.

        Args:
            text: Input text.

        Returns:
            Token count.
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))
