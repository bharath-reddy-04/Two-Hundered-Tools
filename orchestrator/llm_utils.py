"""
orchestrator/llm_utils.py
=========================
Shared LLM call utilities for the orchestration agent.

Centralises all google-genai request-building, response extraction,
code-fence stripping, and JSON parsing so that mode files (all_loaded,
category_gated, …) contain **zero** provider-specific imports.

Public API
----------
    call_llm_json(role, system_prompt, user_content, model_router) -> dict | list
    LLMEmptyResponseError
    LLMMalformedJSONError
"""

from __future__ import annotations

import json
import logging
from typing import Any

from orchestrator.config import ModelRole

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions — callers handle these distinctly
# ---------------------------------------------------------------------------

class LLMEmptyResponseError(RuntimeError):
    """The model returned no text content (empty candidates or empty parts)."""


class LLMMalformedJSONError(ValueError):
    """The model's response text is not valid JSON."""

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def call_llm_json(
    role: ModelRole,
    system_prompt: str,
    user_content: str,
    model_router: Any,
) -> dict | list:
    """
    Call model_router.generate() for *role* and return the parsed JSON response.

    The google-genai import lives here — callers need no provider-specific code.

    Parameters
    ----------
    role          : ModelRole for model selection.
    system_prompt : Instruction text (passed as system_instruction).
    user_content  : User-turn text.
    model_router  : ModelRouter instance.

    Returns
    -------
    dict | list — the parsed JSON value from the model.

    Raises
    ------
    LLMEmptyResponseError
        The model returned no text (empty candidates or empty parts).
    LLMMalformedJSONError
        The response came back but is not valid JSON.
    Any provider exception
        Network / API failures from model_router.generate() bubble up unchanged
        so callers can distinguish them from JSON-shape failures.
    """
    from google.genai import types as _types  # type: ignore  # one authorised import

    gen_config = _types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
    )
    contents = [{"role": "user", "parts": [{"text": user_content}]}]

    response = model_router.generate(role, contents, config=gen_config)

    # ── Extract text ──────────────────────────────────────────────────────
    response_text = ""
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                response_text += part.text

    if not response_text.strip():
        raise LLMEmptyResponseError(
            f"Model returned empty response for role={role.value!r}"
        )

    # ── Strip code fences ─────────────────────────────────────────────────
    clean = response_text.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    # ── Parse JSON ────────────────────────────────────────────────────────
    try:
        return json.loads(clean)
    except json.JSONDecodeError as exc:
        raise LLMMalformedJSONError(
            f"Model response is not valid JSON: {exc}",
            raw_text=clean,
        ) from exc
