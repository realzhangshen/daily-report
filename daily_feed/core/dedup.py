"""Article deduplication using exact URL matching only."""

from __future__ import annotations

from .types import Article


def dedup_articles(articles: list[Article], threshold: int = 92) -> list[Article]:
    """Remove duplicate articles from a list.

    Deduplication is based on exact URL matches only.

    Args:
        articles: List of articles to deduplicate
        threshold: Deprecated and ignored. Kept for backward compatibility.

    Returns:
        Deduplicated list of articles, preserving original order
    """
    seen_urls: set[str] = set()
    kept: list[Article] = []

    for article in articles:
        if article.url in seen_urls:
            continue
        seen_urls.add(article.url)
        kept.append(article)

    return kept
