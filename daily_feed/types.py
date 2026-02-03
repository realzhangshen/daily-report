from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Article:
    title: str
    site: str
    url: str
    time: str | None = None
    author: str | None = None
    summary: str | None = None
    category: str | None = None


@dataclass
class ExtractedArticle:
    article: Article
    text: str | None
    error: str | None = None


@dataclass
class ArticleSummary:
    article: Article
    bullets: list[str] = field(default_factory=list)
    takeaway: str = ""
    topic: str | None = None
    status: str = "ok"
    meta: dict[str, Any] = field(default_factory=dict)
