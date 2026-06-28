"""
LLM Generator Module
====================

Provides the LLMGenerator class that takes a query and assembled context,
formats them into a prompt, and generates an answer via an LLM backend.

Supported backends:
  - **openai**: Uses the OpenAI API via ``langchain_openai.ChatOpenAI``.
  - **ollama**: Uses a local Ollama server via ``langchain_community.llms.Ollama``.

The prompt template is defined in ``config.PROMPT_TEMPLATE`` and follows
a strict instruction-following format that constrains the LLM to answer
solely from the provided context.

Usage:
    from pipeline.generator import LLMGenerator

    generator = LLMGenerator(backend="ollama")
    result = generator.generate(
        query="What is the GDP of India?",
        context="India's GDP in 2024 was ..."
    )
    print(result['answer'])
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class LLMGenerator:
    """Generates answers from a query and context using an LLM.

    Supports both OpenAI API and local Ollama backends.  The backend is
    selected at initialisation time and the generator exposes a unified
    ``generate()`` interface.

    Attributes:
        backend (str): Either ``"openai"`` or ``"ollama"``.
        model_name (str): Name of the underlying LLM model.
        llm: The LangChain LLM instance.
    """

    def __init__(self, backend: str = config.LLM_BACKEND) -> None:
        """Initialise the LLMGenerator.

        Args:
            backend: LLM backend to use. One of ``"openai"`` or ``"ollama"``.
                Default from ``config.LLM_BACKEND``.

        Raises:
            ValueError: If an unsupported backend is specified.
            ImportError: If required LangChain packages are not installed.
        """
        self.backend = backend.lower()
        self.llm = None
        self.model_name = ""

        if self.backend == "openai":
            self._init_openai()
        elif self.backend == "ollama":
            self._init_ollama()
        else:
            raise ValueError(
                f"Unsupported LLM backend: '{backend}'. "
                f"Choose 'openai' or 'ollama'."
            )

        logger.info(
            "LLMGenerator initialised  |  backend=%s  model=%s",
            self.backend, self.model_name,
        )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def generate(self, query: str, context: str) -> Dict:
        """Generate an answer for the given query using the provided context.

        Args:
            query: The user's question.
            context: The assembled context from ACOA-selected chunks.

        Returns:
            Dictionary containing::

                {
                    "answer":      str,   # generated answer text
                    "prompt":      str,   # the formatted prompt sent to LLM
                    "tokens_used": int,   # approximate token count
                    "model":       str,   # model name used
                }
        """
        # Format the prompt using the config template
        prompt = config.PROMPT_TEMPLATE.format(
            context=context,
            question=query,
        )

        # Approximate token count for the prompt
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            prompt_tokens = len(encoding.encode(prompt))
        except Exception:
            # Rough estimate: ~4 chars per token
            prompt_tokens = len(prompt) // 4

        # Generate answer
        answer = self._call_llm(prompt)

        # Approximate total tokens (prompt + response)
        try:
            response_tokens = len(encoding.encode(answer))
        except Exception:
            response_tokens = len(answer) // 4

        total_tokens = prompt_tokens + response_tokens

        return {
            "answer": answer,
            "prompt": prompt,
            "tokens_used": total_tokens,
            "model": self.model_name,
        }

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _init_openai(self) -> None:
        """Initialise the OpenAI backend."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain_openai is required for the OpenAI backend. "
                "Install it with: pip install langchain-openai"
            )

        if not config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set the environment variable "
                "or update config.py."
            )

        self.model_name = config.OPENAI_MODEL
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=config.OPENAI_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )

    def _init_ollama(self) -> None:
        """Initialise the Ollama backend."""
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            raise ImportError(
                "langchain_community is required for the Ollama backend. "
                "Install it with: pip install langchain-community"
            )

        self.model_name = config.OLLAMA_MODEL
        self.llm = Ollama(
            model=self.model_name,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
            num_predict=config.LLM_MAX_TOKENS,
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM with error handling.

        Args:
            prompt: The formatted prompt string.

        Returns:
            Generated answer text.  On failure, returns an error message.
        """
        try:
            start_time = time.perf_counter()

            if self.backend == "openai":
                # ChatOpenAI expects messages; invoke with a simple string
                response = self.llm.invoke(prompt)
                # ChatOpenAI returns an AIMessage object
                answer = response.content if hasattr(response, "content") else str(response)
            else:
                # Ollama LLM returns a string directly
                answer = self.llm.invoke(prompt)
                if not isinstance(answer, str):
                    answer = str(answer)

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(
                "LLM generation complete  |  model=%s  latency=%.0fms",
                self.model_name, elapsed,
            )
            return answer.strip()

        except ConnectionError as e:
            error_msg = (
                f"Connection error with {self.backend} backend: {e}. "
                f"Ensure the service is running."
            )
            logger.error(error_msg)
            return f"[Generation Error] {error_msg}"

        except Exception as e:
            error_msg = f"LLM generation failed ({self.backend}): {e}"
            logger.error(error_msg, exc_info=True)
            return f"[Generation Error] {error_msg}"
