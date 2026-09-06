"""
orchestrator/model_router.py
============================
Routes model requests through ModelRole → model name mapping.

Nodes always call ``model_router.generate(role, ...)``, never a provider
directly, so this can be split into distinct models later purely via config.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from orchestrator import model_manager
from orchestrator.config import ModelRole, OrchestratorConfig

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Maps ModelRole → model name and delegates generation to model_manager.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config

    def select(self, role: ModelRole, context: Optional[dict] = None) -> str:
        """Return the model name for the given role."""
        return self._config.get_model(role)

    def generate(
        self,
        role: ModelRole,
        contents: list[dict[str, Any]],
        config: Any = None,
        context: Optional[dict] = None,
    ) -> Any:
        """
        Select model for role and generate a response.

        Parameters
        ----------
        role : ModelRole
            The role requesting generation.
        contents : list[dict]
            Gemini-style contents list.
        config : GenerateContentConfig, optional
            Generation config.
        context : dict, optional
            Additional context for model selection (unused for now).

        Returns
        -------
        GenerateContentResponse
        """
        model_name = self.select(role, context)
        logger.debug("ModelRouter: role=%s → model=%s", role.value, model_name)
        return model_manager.generate(model_name, contents, config)
