"""
orchestrator/model_manager.py
=============================
Thin wrapper around the google-genai client.

Provides lazy initialization and a single ``generate()`` method
so nodes never import ``google.genai`` directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_client: Any = None


def _get_client() -> Any:
    """Return a cached google-genai Client, creating one lazily."""
    global _client
    if _client is not None:
        return _client

    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'google-genai' package is not installed.\n"
            "Run: pip install google-genai"
        ) from exc

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Export it or add it to your .env file."
        )
    _client = genai.Client(api_key=api_key)
    return _client


def reset_client() -> None:
    """Force a fresh client on the next call — used by tests."""
    global _client
    _client = None


import time

def generate(
    model: str,
    contents: list[dict[str, Any]],
    config: Any = None,
    max_retries: int = 5,
) -> Any:
    """
    Call the Gemini API and return the raw response, with automatic retry
    and exponential backoff on 429 rate limits or transient errors.

    Parameters
    ----------
    model : str
        Model name, e.g. ``"gemini-2.5-flash"``.
    contents : list[dict]
        Gemini-style contents list.
    config : GenerateContentConfig, optional
        Full generation config (system_instruction, tools, etc.).
    max_retries : int, optional
        Maximum number of retries on 429/transient error (default: 5).

    Returns
    -------
    GenerateContentResponse
    """
    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "contents": contents,
    }
    if config is not None:
        kwargs["config"] = config

    logger.debug("LLM call: model=%s, contents_len=%d", model, len(contents))

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            is_rate_limit = (
                "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "ResourceExhausted" in err_str
                or "rate limit" in err_str.lower()
                or "quota" in err_str.lower()
                or "503" in err_str
                or "500" in err_str
            )
            if is_rate_limit and attempt < max_retries:
                sleep_secs = max(4.0, (2 ** attempt) * 2.5)
                logger.warning(
                    "LLM call to %s rate-limited (%s). Retrying in %.1fs (attempt %d/%d)...",
                    model,
                    err_str[:120],
                    sleep_secs,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(sleep_secs)
            else:
                raise last_exc

