"""
Groq LLaMA API client with retry logic, JSON validation, and response caching.
"""

import json
import hashlib
import logging
import time
from typing import Optional

from cachetools import TTLCache
from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS

logger = logging.getLogger(__name__)

# ── Response Cache (1 hour TTL, 200 entries) ───────────────────────────────
_response_cache: TTLCache = TTLCache(maxsize=200, ttl=3600)


def _cache_key(system: str, prompt: str) -> str:
    """Create a deterministic cache key from the prompt."""
    raw = f"{system}::{prompt}"
    return hashlib.md5(raw.encode()).hexdigest()


class LLMService:
    """Wrapper around Groq SDK for structured JSON completions."""

    def __init__(self):
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set — LLM calls will fail")
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.model = GROQ_MODEL
        self.temperature = GROQ_TEMPERATURE
        self.max_tokens = GROQ_MAX_TOKENS
        self.total_tokens_used = 0

    # ── Core completion ─────────────────────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> dict:
        """
        Send a prompt to the LLM and return parsed JSON.

        Args:
            system_prompt: System-level instructions.
            user_prompt: The user/analysis prompt.
            use_cache: Whether to check/store in cache.
            max_retries: Number of retries on failure.

        Returns:
            Parsed JSON dict from the LLM response.

        Raises:
            RuntimeError: If all retries exhausted.
        """
        # ── Cache check ────────────────────────────────────────────────
        key = _cache_key(system_prompt, user_prompt)
        if use_cache and key in _response_cache:
            logger.debug("LLM cache hit")
            return _response_cache[key]

        if not self.client:
            raise RuntimeError(
                "Groq API key not configured. Set GROQ_API_KEY in your .env file."
            )

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )

                # Track token usage
                if response.usage:
                    self.total_tokens_used += response.usage.total_tokens

                raw = response.choices[0].message.content.strip()
                parsed = self._parse_json(raw)

                # Cache the result
                if use_cache:
                    _response_cache[key] = parsed

                return parsed

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    f"JSON parse error on attempt {attempt}/{max_retries}: {e}"
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed on attempt {attempt}/{max_retries}: {e}"
                )
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.info(f"Retrying in {wait}s...")
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {max_retries} retries. Last error: {last_error}"
        )

    # ── JSON parsing with fallback ──────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """
        Parse JSON from LLM output. Handles markdown fencing and partial JSON.
        """
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting the first JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("Could not parse JSON from LLM output", text, 0)

    # ── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "total_tokens_used": self.total_tokens_used,
            "cache_size": len(_response_cache),
            "model": self.model,
        }


# ── Module-level singleton ──────────────────────────────────────────────────
llm_service = LLMService()
