from __future__ import annotations

from rapidfuzz import fuzz

from .types import Article


def dedup_articles(articles: list[Article], threshold: int = 92) -> list[Article]:
    seen_urls: set[str] = set()
    kept: list[Article] = []
    titles: list[str] = []

    for article in articles:
        if article.url in seen_urls:
            continue
        if _is_similar_title(article.title, titles, threshold):
            continue
        seen_urls.add(article.url)
        titles.append(article.title)
        kept.append(article)

    return kept


def _is_similar_title(title: str, titles: list[str], threshold: int) -> bool:
    for existing in titles:
        if fuzz.ratio(title, existing) >= threshold:
            return True
    return False
