"""
Markdown parser for Folo RSS export format.

This module parses RSS feed exports in Folo's markdown format into structured
Article objects. The format uses:
- ## headings for categories
- ### headings for article titles
- - prefixed lines for metadata fields (Source, Time, Link, Summary)
"""

from __future__ import annotations

import re
from typing import Iterable

from .types import Article


# Regex patterns for matching Folo markdown structure
CATEGORY_RE = re.compile(r"^##\s+(.+?)\s*$")  # Matches "## Category Name"
TITLE_RE = re.compile(r"^###\s+(.+?)\s*$")     # Matches "### Article Title"
FIELD_RE = re.compile(r"^-\s+(Source|Time|Link|Summary):\s*(.+?)\s*$")  # Matches "- Field: value"


def parse_folo_markdown(text: str) -> list[Article]:
    """Parse Folo markdown format into a list of Article objects.

    The Folo format structure:
        ## Category Name
        ### Article Title
        - Source: Site Name @Author
        - Time: timestamp
        - Link: https://example.com
        - Summary: Brief summary text

    Args:
        text: The full markdown content as a string

    Returns:
        A list of Article objects parsed from the markdown. Articles with
        missing required fields (title, link, source) are skipped.
    """
    lines = text.splitlines()
    current_category: str | None = None  # Active category section
    current: dict[str, str] | None = None  # Accumulator for current article fields
    articles: list[Article] = []

    def flush():
        """Finalize the current article being parsed and add to the list.

        Skips articles missing required fields (title, link, source).
        """
        if not current:
            return
        title = current.get("title")
        link = current.get("link")
        source = current.get("source")
        # Required fields validation
        if not title or not link or not source:
            return
        site, author = _parse_source(source)
        articles.append(
            Article(
                title=title,
                site=site,
                url=link,
                time=current.get("time"),
                author=author,
                summary=current.get("summary"),
                category=current_category,
            )
        )

    # Process each line sequentially, tracking state via current and current_category
    for line in lines:
        category_match = CATEGORY_RE.match(line)
        if category_match:
            flush()  # Finalize any pending article
            current_category = category_match.group(1).strip()
            current = None
            continue

        title_match = TITLE_RE.match(line)
        if title_match:
            flush()  # Finalize any pending article
            current = {"title": title_match.group(1).strip()}
            continue

        field_match = FIELD_RE.match(line)
        if field_match and current is not None:
            field, value = field_match.group(1), field_match.group(2)
            key = field.lower()
            current[key] = value.strip()

    # Don't forget the last article
    flush()
    return articles


def _parse_source(source: str) -> tuple[str, str | None]:
    """Parse the source field into site and optional author.

    The source field format is "Site Name @Author". The @author part
    is optional.

    Args:
        source: The raw source string from the markdown

    Returns:
        A tuple of (site, author) where author may be None

    Examples:
        >>> _parse_source("TechCrunch")
        ("TechCrunch", None)
        >>> _parse_source("New York Times @John Doe")
        ("New York Times", "John Doe")
    """
    if "@" in source:
        parts = source.split("@", 1)
        site = parts[0].strip()
        author = parts[1].strip() if parts[1].strip() else None
        return site, author
    return source.strip(), None
