"""
Article deduplication using URL matching and fuzzy title comparison.

This module removes duplicate articles based on:
1. Exact URL matches (canonical duplicates)
2. Fuzzy title similarity (same content, different URLs)
"""

from __future__ import annotations

from rapidfuzz import fuzz

from .types import Article


def dedup_articles(articles: list[Article], threshold: int = 92) -> list[Article]:
    """Remove duplicate articles from a list.

    Deduplication happens in two passes:
    1. Remove exact URL duplicates
    2. Remove articles with titles similar to already-seen articles

    The fuzzy matching helps catch cases where the same article is
    syndicated to multiple sites with slightly different titles.

    Args:
        articles: List of articles to deduplicate
        threshold: Similarity threshold (0-100) for fuzzy title matching.
                   Default 92 means titles must be 92% similar to be duplicates.

    Returns:
        Deduplicated list of articles, preserving original order
    """
    seen_urls: set[str] = set()
    kept: list[Article] = []
    titles: list[str] = []

    for article in articles:
        # Skip if we've seen this exact URL
        if article.url in seen_urls:
            continue
        # Skip if title is similar to any kept article
        if _is_similar_title(article.title, titles, threshold):
            continue
        seen_urls.add(article.url)
        titles.append(article.title)
        kept.append(article)

    return kept


def _is_similar_title(title: str, titles: list[str], threshold: int) -> bool:
    """Check if a title is similar to any title in the given list.

    Uses rapidfuzz's ratio function which calculates the Levenshtein
    distance as a similarity percentage.

    Args:
        title: The title to check
        titles: List of existing titles to compare against
        threshold: Minimum similarity (0-100) to consider titles duplicate

    Returns:
        True if title is similar to any in the list, False otherwise
    """
    for existing in titles:
        if fuzz.ratio(title, existing) >= threshold:
            return True
    return False
