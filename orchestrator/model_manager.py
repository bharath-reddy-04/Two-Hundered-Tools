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


def generate(
    model: str,
    contents: list[dict[str, Any]],
    config: Any = None,
) -> Any:
    """
    Call the Gemini API and return the raw response.

    Parameters
    ----------
    model : str
        Model name, e.g. ``"gemini-2.5-flash"``.
    contents : list[dict]
        Gemini-style contents list.
    config : GenerateContentConfig, optional
        Full generation config (system_instruction, tools, etc.).

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
    return client.models.generate_content(**kwargs)
