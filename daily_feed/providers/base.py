"""
Abstract base class for LLM providers.

New providers should inherit from Provider and implement
the summarize_article and group_topics methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.types import Article, ArticleSummary


class Provider(ABC):
    """Abstract base class for LLM providers.

    Defines the interface for summarization and topic grouping.
    Concrete implementations (e.g., GeminiProvider) must implement
    both abstract methods.
    """

    @abstractmethod
    def summarize_article(self, article: Article, text: str) -> ArticleSummary:
        """Generate a summary for a single article.

        Args:
            article: The article to summarize (contains metadata)
            text: The full text content of the article

        Returns:
            ArticleSummary with bullets, takeaway, topic, and status
        """
        raise NotImplementedError

    @abstractmethod
    def group_topics(self, summaries: list[ArticleSummary]) -> dict[str, list[ArticleSummary]]:
        """Group articles by topic.

        Args:
            summaries: List of ArticleSummary objects to group

        Returns:
            Dictionary mapping topic names to lists of summaries
        """
        raise NotImplementedError
