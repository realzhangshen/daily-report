"""
Core data types for the Daily Feed Agent.

This module defines the fundamental data structures used throughout the pipeline:
- Article: Raw article data parsed from markdown input
- ExtractedArticle: Article with fetched and extracted content
- ArticleSummary: Article with AI-generated summary and metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Article:
    """Represents a raw article parsed from RSS feed markdown.

    Attributes:
        title: The article headline
        site: The website/publication name (e.g., "TechCrunch", "New York Times")
        url: The full URL to the original article
        time: Optional timestamp from the RSS feed
        author: Optional author name, extracted from source field if present
        summary: Optional brief summary from the RSS feed
        category: Optional category section this article belongs to
        id: Optional unique identifier from JSON input
        published_at: Optional ISO 8601 published timestamp from JSON input
        inserted_at: Optional ISO 8601 insertion timestamp from JSON input
    """
    title: str
    site: str
    url: str
    time: str | None = None
    author: str | None = None
    summary: str | None = None
    category: str | None = None
    id: str | None = None
    published_at: str | None = None
    inserted_at: str | None = None


@dataclass
class ExtractedArticle:
    """Article with fetched and extracted full text content.

    This represents an article after the fetch/extract stage of the pipeline.
    Either text or error will be populated, but not both.

    Attributes:
        article: The original Article object with metadata
        text: The extracted plain text content, or None if extraction failed
        error: Error message if fetch/extract failed, None otherwise
    """
    article: Article
    text: str | None
    error: str | None = None


@dataclass
class ArticleSummary:
    """Article with AI-generated summary and metadata.

    This represents the final output of the summarization stage, containing
    both the original article and the LLM-generated summary components.

    Attributes:
        article: The original Article object with metadata
        bullets: List of key bullet points extracted by the LLM
        takeaway: A one-sentence takeaway/summary from the LLM
        topic: The topic group this article belongs to (assigned during grouping)
        status: Processing status - "ok", "summary_only", "provider_error", "parse_error"
        meta: Additional metadata (e.g., model used, raw response on parse error)
    """
    article: Article
    bullets: list[str] = field(default_factory=list)
    takeaway: str = ""
    topic: str | None = None
    status: str = "ok"
    meta: dict[str, Any] = field(default_factory=dict)
