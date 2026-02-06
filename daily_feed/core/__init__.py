"""
Core domain models and business logic.

This package contains data types and business logic that is
independent of any specific pipeline stage.
"""

from .types import Article, ExtractedArticle, ArticleSummary
from .entry import EntryManager, slugify, short_hash
from .dedup import dedup_articles

__all__ = [
    "Article",
    "ExtractedArticle",
    "ArticleSummary",
    "EntryManager",
    "slugify",
    "short_hash",
    "dedup_articles",
]
