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

from daily_feed.core.types import Article
from daily_feed.core.types import AnalysisResult
from daily_feed.utils.logging import JsonlFormatter


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
    def llm_debug(self) -> Path:
        """Returns path to the LLM debug JSONL file.

        Returns:
            Path object for llm_events.jsonl
        """
        return self.folder / "llm_events.jsonl"

    @property
    def analysis_raw(self) -> Path:
        """Returns path to the analysis raw text file.

        Returns:
            Path object for analysis.txt
        """
        return self.folder / "analysis.txt"

    @property
    def analysis_meta(self) -> Path:
        """Returns path to the analysis metadata JSON file.

        Returns:
            Path object for analysis.json
        """
        return self.folder / "analysis.json"

    @property
    def extraction_raw(self) -> Path:
        """Returns path to the extraction result JSON file.

        Returns:
            Path object for extraction.json
        """
        return self.folder / "extraction.json"

    @property
    def analysis_debug(self) -> Path:
        """Returns path to the analysis debug JSONL file.

        Returns:
            Path object for entry_events.jsonl
        """
        return self.folder / "entry_events.jsonl"

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

    def write_analysis_result(self, result: AnalysisResult) -> None:
        """Write analysis output and metadata to entry files.

        Args:
            result: The AnalysisResult object to write
        """
        if result.analysis:
            self.analysis_raw.write_text(result.analysis, encoding="utf-8")

        meta: dict[str, Any] = {
            "status": result.status,
            "article": {
                "title": result.article.title,
                "url": result.article.url,
                "site": result.article.site,
            },
        }
        meta.update(result.meta or {})
        with open(self.analysis_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def read_analysis_result(self) -> dict[str, Any] | None:
        """Read analysis output and metadata from entry files.

        Returns:
            Dictionary with analysis data, or None if files are missing
        """
        if not self.analysis_meta.exists():
            return None
        analysis_text = ""
        if self.analysis_raw.exists():
            analysis_text = self.analysis_raw.read_text(encoding="utf-8")
        with open(self.analysis_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["analysis"] = analysis_text
        return meta

    def write_extraction_result(self, result) -> None:
        """Write extraction result to cache.

        Args:
            result: The ExtractionResult object to write
        """
        data = {
            "one_line_summary": result.one_line_summary,
            "category": result.category,
            "tags": result.tags,
            "importance": result.importance,
            "content_type": result.content_type,
            "key_takeaway": result.key_takeaway,
            "status": result.status,
            "article": {
                "title": result.article.title,
                "url": result.article.url,
                "site": result.article.site,
            },
        }
        data.update(result.meta or {})
        with open(self.extraction_raw, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def read_extraction_result(self) -> dict[str, Any] | None:
        """Read extraction result from cache.

        Returns:
            Dictionary with extraction data, or None if file is missing
        """
        if not self.extraction_raw.exists():
            return None
        with open(self.extraction_raw, "r", encoding="utf-8") as f:
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

    def get_analysis_logger(self) -> logging.Logger | None:
        """Get or create logger for this entry's analysis events.

        Returns:
            Logger instance, or None if folder doesn't exist
        """
        if not self.folder.exists():
            return None

        logger = logging.getLogger(f"daily_feed.entry.analysis.{self.folder.name}")
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(self.analysis_debug, encoding="utf-8")
        file_handler.setFormatter(JsonlFormatter())
        logger.addHandler(file_handler)

        return logger
