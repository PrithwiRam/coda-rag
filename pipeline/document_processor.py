"""
Document Processor Module
=========================

This module provides the DocumentProcessor class for the ACORA pipeline.
It handles:
  - Text cleaning (whitespace normalization, unicode normalization)
  - Token-accurate chunking using tiktoken (cl100k_base encoding)
  - Metadata generation for each chunk (doc_id, chunk_id, token_count, etc.)

The chunking strategy uses fixed-size windows with configurable overlap,
measured in tokens (not characters) for precise token-budget management
downstream in the ACOA context-optimization step.

Usage:
    from pipeline.document_processor import DocumentProcessor

    processor = DocumentProcessor()
    chunks = processor.process_documents([
        {"text": "Some long document...", "doc_id": "doc_001", "title": "My Doc"}
    ])
"""

import re
import sys
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Import project config (assumes cwd is acora/ OR acora is on sys.path)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import tiktoken

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes raw documents into cleaned, token-counted chunks.

    The processor accepts documents as dictionaries and produces a flat list
    of chunk dictionaries ready for downstream indexing (BM25, dense, etc.).

    Attributes:
        chunk_size (int): Maximum number of tokens per chunk.
        chunk_overlap (int): Number of overlapping tokens between consecutive
            chunks to preserve context continuity.
        encoding (tiktoken.Encoding): The tiktoken encoding used for
            tokenization (cl100k_base).
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ) -> None:
        """Initialise the DocumentProcessor.

        Args:
            chunk_size: Maximum tokens per chunk (default from config).
            chunk_overlap: Token overlap between consecutive chunks
                (default from config).

        Raises:
            ValueError: If chunk_overlap >= chunk_size (would cause infinite
                loop during chunking).
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than "
                f"chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # cl100k_base is the encoding used by GPT-4 / GPT-3.5-turbo and
        # provides a good general-purpose tokenization for counting.
        self.encoding = tiktoken.get_encoding("cl100k_base")
        logger.info(
            "DocumentProcessor initialised  |  chunk_size=%d  overlap=%d",
            self.chunk_size,
            self.chunk_overlap,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process a list of raw documents into chunks.

        Each input document must be a dictionary with at least:
            - ``text`` (str): The document body.
            - ``doc_id`` (str): A unique identifier for the document.

        Optional keys:
            - ``title`` (str): Human-readable title; stored in chunk metadata.

        Args:
            documents: List of document dictionaries.

        Returns:
            A flat list of chunk dictionaries, each containing::

                {
                    "text":        str,   # chunk text
                    "doc_id":      str,   # parent document id
                    "chunk_id":    str,   # unique chunk identifier
                    "chunk_index": int,   # 0-based index within the document
                    "token_count": int,   # exact token count (tiktoken)
                    "metadata":    dict,  # extra info (title, etc.)
                }
        """
        if not documents:
            logger.warning("process_documents called with empty document list.")
            return []

        all_chunks: List[Dict] = []

        for doc in documents:
            # ------ validate required keys ----------------------------------
            text = doc.get("text")
            doc_id = doc.get("doc_id")

            if not text or not doc_id:
                logger.warning(
                    "Skipping document with missing 'text' or 'doc_id': %s",
                    {k: v for k, v in doc.items() if k != "text"},
                )
                continue

            # ------ clean & chunk -------------------------------------------
            cleaned_text = self._clean_text(text)

            if not cleaned_text:
                logger.warning(
                    "Document %s is empty after cleaning – skipping.", doc_id
                )
                continue

            title = doc.get("title", "")
            chunks = self._chunk_text(cleaned_text, doc_id)

            # Attach title metadata to every chunk from this document.
            for chunk in chunks:
                chunk["metadata"]["title"] = title

            all_chunks.extend(chunks)

        logger.info(
            "Processed %d document(s) → %d chunk(s).",
            len(documents),
            len(all_chunks),
        )
        return all_chunks

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Clean and normalise raw text.

        Steps performed:
            1. Unicode NFC normalisation (canonical decomposition + composition).
            2. Replace non-breaking spaces, zero-width chars, and other
               invisible Unicode artefacts.
            3. Collapse multiple whitespace characters into a single space.
            4. Strip leading / trailing whitespace.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text string.
        """
        # Step 1 – Unicode normalisation
        text = unicodedata.normalize("NFC", text)

        # Step 2 – Remove zero-width characters and non-breaking spaces
        text = text.replace("\u00a0", " ")   # non-breaking space → space
        text = text.replace("\u200b", "")     # zero-width space
        text = text.replace("\u200c", "")     # zero-width non-joiner
        text = text.replace("\u200d", "")     # zero-width joiner
        text = text.replace("\ufeff", "")     # BOM / zero-width no-break space

        # Step 3 – Collapse runs of whitespace (incl. newlines, tabs)
        text = re.sub(r"\s+", " ", text)

        # Step 4 – Strip
        text = text.strip()

        return text

    def _chunk_text(self, text: str, doc_id: str) -> List[Dict]:
        """Split *text* into fixed-size token chunks with overlap.

        Chunking is performed at the **token** level using tiktoken so that
        downstream token-budget constraints (e.g. ACOA_TOKEN_BUDGET) are
        respected accurately.

        Algorithm:
            1. Encode the full text into a token-id list.
            2. Slide a window of ``chunk_size`` tokens with a step of
               ``chunk_size - chunk_overlap``.
            3. Decode each token window back to a text string.

        Args:
            text: Cleaned text to chunk.
            doc_id: The parent document identifier.

        Returns:
            List of chunk dictionaries (see ``process_documents`` for schema).
        """
        # Encode text → list of token ids
        token_ids = self.encoding.encode(text)
        total_tokens = len(token_ids)

        # Edge case: text fits entirely within one chunk
        if total_tokens == 0:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: List[Dict] = []
        chunk_index = 0

        for start in range(0, total_tokens, step):
            end = min(start + self.chunk_size, total_tokens)
            chunk_token_ids = token_ids[start:end]

            # Decode token ids back to text
            chunk_text = self.encoding.decode(chunk_token_ids)
            token_count = len(chunk_token_ids)

            chunk_id = f"{doc_id}_chunk_{chunk_index}"

            chunks.append(
                {
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "token_count": token_count,
                    "metadata": {},  # caller adds title, etc.
                }
            )

            chunk_index += 1

            # If we've consumed all tokens, stop
            if end >= total_tokens:
                break

        logger.debug(
            "Document %s chunked into %d chunk(s) (total_tokens=%d).",
            doc_id,
            len(chunks),
            total_tokens,
        )
        return chunks
