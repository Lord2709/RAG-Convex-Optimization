"""
llm_generator.py
----------------
LLM generation for end-to-end RAG evaluation.
Supports two providers — switch between them in config.yaml (llm.provider).

  provider: "groq"    → uses Groq API  (GROQ_API_KEY in .env)
  provider: "gemini"  → uses Gemini API (GEMINI_API_KEY in .env)

Both implement the same BaseGenerator interface so the rest of the
pipeline is completely unaware of which backend is running.

Retry logic:
  On 429 (rate limit) the API response includes a suggested retryDelay.
  We parse that value and wait exactly that long before retrying.
  Falls back to exponential backoff if the delay can't be parsed.
  Gives up after llm.max_retries attempts and returns "" for that query.

API keys are always read from the .env file — never hardcoded here.
"""

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
Answer the following question using only the context provided.
Be concise — give the answer in one short phrase or sentence, nothing more.

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prompt(question: str, context_docs: List[Dict]) -> str:
    context = "\n\n".join(
        f"[Doc {i + 1}] {doc.get('title', 'Untitled')}\n{doc['text'][:500]}"
        for i, doc in enumerate(context_docs)
    )
    return _PROMPT_TEMPLATE.format(context=context, question=question)


def _parse_retry_delay(exc: Exception, fallback: float) -> float:
    """
    Try to extract the suggested retryDelay (in seconds) from a 429 error.
    The Gemini API embeds it as e.g. 'retryDelay': '40s' inside the error dict.
    Groq includes it in the HTTP headers which the SDK surfaces as a string.
    Falls back to `fallback` seconds if nothing is parseable.
    """
    text = str(exc)
    match = re.search(r"retry[_ ]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", text, re.I)
    if match:
        return float(match.group(1))
    # also catch plain "Please retry in 40s" or "retry in 40.2s"
    match = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", text, re.I)
    if match:
        return float(match.group(1))
    return fallback


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate_limit" in text.lower()


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseGenerator(ABC):

    def __init__(self, max_retries: int, base_backoff: float):
        self._max_retries = max_retries
        self._base_backoff = base_backoff

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Single API call — raises on error, returns text on success."""

    def generate(self, question: str, context_docs: List[Dict]) -> str:
        """Generate with automatic retry on rate limits."""
        prompt = _build_prompt(question, context_docs)
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._call_api(prompt)
            except Exception as exc:
                if _is_rate_limit(exc):
                    wait = _parse_retry_delay(exc, fallback=self._base_backoff * (2 ** (attempt - 1)))
                    if attempt < self._max_retries:
                        logger.warning(
                            "Rate limit hit (attempt %d/%d). Waiting %.1fs …",
                            attempt, self._max_retries, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            "Rate limit persisted after %d attempts. "
                            "Returning empty string for this query. "
                            "Consider switching provider in config.yaml (llm.provider) "
                            "or reducing llm.num_eval_queries.",
                            self._max_retries,
                        )
                        return ""
                else:
                    logger.error("API error: %s", exc)
                    return ""
        return ""

    def generate_batch(
        self,
        questions: List[str],
        context_docs_list: List[List[Dict]],
    ) -> List[str]:
        answers = []
        total = len(questions)
        for i, (q, docs) in enumerate(zip(questions, context_docs_list)):
            if (i + 1) % 5 == 0 or i == 0:
                logger.info("  Generating  %d / %d", i + 1, total)
            answers.append(self.generate(q, docs))
        return answers


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------

class GroqGenerator(BaseGenerator):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            max_retries=config["llm"].get("max_retries", 3),
            base_backoff=config["llm"].get("base_backoff_seconds", 10.0),
        )
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Copy .env.example → .env and fill it in."
            )
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._model = config["groq"]["model"]
        self._max_tokens = config["llm"]["max_tokens"]
        self._temperature = config["llm"]["temperature"]
        logger.info("GroqGenerator ready  model=%s", self._model)

    def _call_api(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class GeminiGenerator(BaseGenerator):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            max_retries=config["llm"].get("max_retries", 3),
            base_backoff=config["llm"].get("base_backoff_seconds", 40.0),
        )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. Copy .env.example → .env and fill it in."
            )
        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=api_key)
        self._model = config["gemini"]["model"]
        self._max_tokens = config["llm"]["max_tokens"]
        self._temperature = config["llm"]["temperature"]
        self._types = types
        logger.info("GeminiGenerator ready  model=%s", self._model)

    def _call_api(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                max_output_tokens=self._max_tokens,
                temperature=self._temperature,
            ),
        )
        return response.text.strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "groq": GroqGenerator,
    "gemini": GeminiGenerator,
}


def create_generator(config: Dict[str, Any]) -> BaseGenerator:
    provider = config["llm"]["provider"].lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Valid options: {list(_PROVIDERS.keys())}"
        )
    logger.info("LLM provider: %s", provider)
    return _PROVIDERS[provider](config)
