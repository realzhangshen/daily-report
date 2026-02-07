"""Abstract interfaces for LLM-driven entry analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

from ...core.types import Article


class AnalysisProvider(ABC):
    """Provider interface for deep-fetch decision and entry analysis."""

    @abstractmethod
    def decide_deep_fetch(
        self,
        article: Article,
        text: str,
        candidate_links: list[str],
        entry_logger: logging.Logger | None = None,
    ) -> dict[str, Any]:
        """Return deep-fetch decision payload."""
        raise NotImplementedError

    @abstractmethod
    def analyze_entry(
        self,
        article: Article,
        base_text: str,
        deep_texts: list[str],
        decision: dict[str, Any],
        entry_logger: logging.Logger | None = None,
    ) -> str:
        """Return long-form analysis text."""
        raise NotImplementedError
