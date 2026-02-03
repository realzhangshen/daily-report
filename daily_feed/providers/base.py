from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Article, ArticleSummary


class Provider(ABC):
    @abstractmethod
    def summarize_article(self, article: Article, text: str) -> ArticleSummary:
        raise NotImplementedError

    @abstractmethod
    def group_topics(self, summaries: list[ArticleSummary]) -> dict[str, list[ArticleSummary]]:
        raise NotImplementedError
