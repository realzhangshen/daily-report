from __future__ import annotations

import re
from typing import Iterable

from .types import Article


CATEGORY_RE = re.compile(r"^##\s+(.+?)\s*$")
TITLE_RE = re.compile(r"^###\s+(.+?)\s*$")
FIELD_RE = re.compile(r"^-\s+(Source|Time|Link|Summary):\s*(.+?)\s*$")


def parse_folo_markdown(text: str) -> list[Article]:
    lines = text.splitlines()
    current_category: str | None = None
    current: dict[str, str] | None = None
    articles: list[Article] = []

    def flush():
        if not current:
            return
        title = current.get("title")
        link = current.get("link")
        source = current.get("source")
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

    for line in lines:
        category_match = CATEGORY_RE.match(line)
        if category_match:
            flush()
            current_category = category_match.group(1).strip()
            current = None
            continue

        title_match = TITLE_RE.match(line)
        if title_match:
            flush()
            current = {"title": title_match.group(1).strip()}
            continue

        field_match = FIELD_RE.match(line)
        if field_match and current is not None:
            field, value = field_match.group(1), field_match.group(2)
            key = field.lower()
            current[key] = value.strip()

    flush()
    return articles


def _parse_source(source: str) -> tuple[str, str | None]:
    if "@" in source:
        parts = source.split("@", 1)
        site = parts[0].strip()
        author = parts[1].strip() if parts[1].strip() else None
        return site, author
    return source.strip(), None
