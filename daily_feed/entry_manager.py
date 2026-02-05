"""Entry manager for per-article cache storage.

This module provides functionality for managing individual article entries
in a hierarchical cache structure. Each article gets its own folder with
a slug-based name and short hash for uniqueness.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from daily_feed.types import Article
from daily_feed.types import ArticleSummary
from daily_feed.logging_utils import JsonlFormatter


def slugify(text: str) -> str:
    """Convert text to URL-safe slug.

    Args:
        text: The text to slugify

    Returns:
        A lowercase, hyphenated slug limited to 50 characters
    """
    # Convert to lowercase
    slug = text.lower()
    # Replace non-alphanumeric characters with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    # Strip leading/trailing hyphens
    slug = slug.strip('-')
    # Fallback for empty slug (e.g., empty title or only special characters)
    if not slug:
        slug = "untitled"
    # Limit to 50 characters
    slug = slug[:50]
    return slug


def short_hash(url: str) -> str:
    """Return first 5 characters of MD5 hash of URL.

    Args:
        url: The URL to hash

    Returns:
        First 5 characters of the MD5 hash (hexadecimal)
    """
    return hashlib.md5(url.encode()).hexdigest()[:5]


class EntryManager:
    """Manages file paths and folder structure for a single article entry.

    Each article entry gets its own folder named as {slug}-{shortHash}
    containing the cached files for that article.
    """

    def __init__(self, articles_dir: Path, article: Article):
        """Initialize the EntryManager.

        Args:
            articles_dir: The parent directory where all article folders are stored
            article: The Article object to manage
        """
        self._articles_dir = articles_dir
        self._article = article

    def _entry_folder(self) -> Path:
        """Returns the entry folder path.

        The folder name is in the format {slug}-{shortHash} where:
        - slug: URL-safe version of the article title
        - shortHash: First 5 chars of MD5 hash of the article URL

        Returns:
            Path object for the article's entry folder
        """
        name = f"{slugify(self._article.title)}-{short_hash(self._article.url)}"
        return self._articles_dir / name

    @property
    def folder(self) -> Path:
        """Returns the entry folder path.

        Returns:
            Path object for the article's entry folder
        """
        return self._entry_folder()

    @property
    def fetched_html(self) -> Path:
        """Returns path to the fetched HTML file.

        Returns:
            Path object for fetched.html
        """
        return self.folder / "fetched.html"

    @property
    def extracted_txt(self) -> Path:
        """Returns path to the extracted text file.

        Returns:
            Path object for extracted.txt
        """
        return self.folder / "extracted.txt"

    @property
    def llm_summary(self) -> Path:
        """Returns path to the LLM summary JSON file.

        Returns:
            Path object for llm_summary.json
        """
        return self.folder / "llm_summary.json"

    @property
    def llm_debug(self) -> Path:
        """Returns path to the LLM debug JSONL file.

        Returns:
            Path object for llm_debug.jsonl
        """
        return self.folder / "llm_debug.jsonl"

    def ensure_folder(self) -> None:
        """Create the entry folder if it doesn't exist.

        This creates the directory structure for storing cached files
        for this article entry.
        """
        self.folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_entry_valid(folder: Path, ttl_days: int | None) -> bool:
        """Check if entry cache is still valid based on TTL.

        Args:
            folder: Path to the entry folder
            ttl_days: TTL in days, or None for no expiration

        Returns:
            True if folder exists and is within TTL (or TTL is None)
        """
        # If folder doesn't exist, it's invalid
        if not folder.exists():
            return False

        # If TTL is None, no expiration - always valid if it exists
        if ttl_days is None:
            return True

        # Check if folder age is within TTL
        folder_mtime = folder.stat().st_mtime
        folder_age_seconds = time.time() - folder_mtime
        ttl_seconds = ttl_days * 86400

        return folder_age_seconds <= ttl_seconds

    def write_llm_summary(self, summary: ArticleSummary) -> None:
        """Write LLM summary to JSON file.

        The JSON includes:
        - bullets
        - takeaway
        - topic
        - status
        - article (title, url, site)
        - model (from summary.meta)
        - generated_at (from summary.meta)

        Args:
            summary: The ArticleSummary object to write
        """
        data: dict[str, Any] = {
            "bullets": summary.bullets,
            "takeaway": summary.takeaway,
            "topic": summary.topic,
            "status": summary.status,
            "article": {
                "title": summary.article.title,
                "url": summary.article.url,
                "site": summary.article.site,
            },
            "model": summary.meta.get("model"),
            "generated_at": summary.meta.get("generated_at"),
        }

        with open(self.llm_summary, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def read_llm_summary(self) -> dict[str, Any] | None:
        """Read LLM summary from JSON file.

        Returns:
            Dictionary with summary data, or None if file doesn't exist
        """
        if not self.llm_summary.exists():
            return None

        with open(self.llm_summary, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_llm_logger(self) -> logging.Logger | None:
        """Get or create logger for this entry's LLM interactions.

        Returns:
            Logger instance, or None if folder doesn't exist
        """
        if not self.folder.exists():
            return None

        logger = logging.getLogger(f"daily_feed.entry.{self.folder.name}")
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.llm_debug, encoding="utf-8")
        file_handler.setFormatter(JsonlFormatter())
        logger.addHandler(file_handler)

        return logger
