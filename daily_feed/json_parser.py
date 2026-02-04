"""JSON parser for Folo RSS export format.

This module parses RSS feed exports in Folo's JSON format into structured
Article objects. The format uses:
- Top-level metadata (exportTime, total)
- Articles array with id, title, url, publishedAt, insertedAt, summary, feedTitle, category
"""

from __future__ import annotations

import logging
from typing import Any

from urllib.parse import urlparse

from .types import Article

logger = logging.getLogger(__name__)


def parse_folo_json(data: dict[str, Any]) -> list[Article]:
    """Parse Folo JSON format into a list of Article objects.

    The Folo JSON format structure:
        {
            "exportTime": "2026-02-03T13:22:16.502Z",
            "total": 41,
            "articles": [
                {
                    "id": "241476308963169281",
                    "title": "Article Title",
                    "url": "https://example.com/article",
                    "publishedAt": "2026-02-03T11:44:10.702Z",
                    "insertedAt": "2026-02-03T11:44:11.423Z",
                    "summary": "Brief summary or empty string",
                    "feedTitle": "Twitter @OpenAI",
                    "category": "Twitter"
                }
            ]
        }

    Args:
        data: The parsed JSON content as a dictionary

    Returns:
        A list of Article objects parsed from the JSON. Articles with
        missing required fields (title, url) are skipped with a warning.

    Raises:
        ValueError: If the JSON is missing the 'articles' key
    """
    if "articles" not in data:
        raise ValueError("Invalid JSON format: missing 'articles' key")

    articles: list[Article] = []

    for item in data["articles"]:
        # Required fields validation
        title = item.get("title")
        url = item.get("url")

        if not title or not url:
            article_id = item.get("id", "unknown")
            logger.warning(f"Skipping article {article_id}: missing required fields (title or url)")
            continue

        # Parse feedTitle into site and author
        feed_title = item.get("feedTitle", "")
        if not feed_title:
            # Fallback: extract domain from URL
            domain = urlparse(url).netloc
            feed_title = domain

        site, author = _parse_feed_title(feed_title)

        # Convert empty summary to None
        summary = item.get("summary")
        if summary == "":
            summary = None

        article = Article(
            id=item.get("id"),
            title=title,
            url=url,
            site=site,
            author=author,
            category=item.get("category"),
            summary=summary,
            published_at=item.get("publishedAt"),
            inserted_at=item.get("insertedAt"),
        )
        articles.append(article)

    return articles


def _parse_feed_title(feed_title: str) -> tuple[str, str | None]:
    """Parse the feedTitle field into site and optional author.

    The feedTitle field format is "Site Name @Author". The @author part
    is optional.

    Args:
        feed_title: The raw feedTitle string from the JSON

    Returns:
        A tuple of (site, author) where author may be None

    Examples:
        >>> _parse_feed_title("Twitter @OpenAI")
        ("Twitter", "OpenAI")
        >>> _parse_feed_title("Product Hunt")
        ("Product Hunt", None)
    """
    if "@" in feed_title:
        parts = feed_title.split("@", 1)
        site = parts[0].strip()
        author = parts[1].strip() if parts[1].strip() else None
        return site, author
    return feed_title.strip(), None
