"""Tests for EntryManager class."""

import pytest
import tempfile
from pathlib import Path

from daily_feed.types import Article
from daily_feed.entry_manager import EntryManager


def test_entry_folder_naming():
    """Entry folder should be slug-shortHash format"""
    articles_dir = Path("/tmp/test_articles")
    article = Article(
        title="OpenAI Codex App Launch",
        site="Tech News",
        url="https://example.com/openai-codex"
    )
    manager = EntryManager(articles_dir, article)

    # Folder name should be slug with short hash
    assert manager.folder.name == "openai-codex-app-launch-d3f76"


def test_entry_folder_paths():
    """Entry manager should provide correct file paths"""
    articles_dir = Path("/tmp/test_articles")
    article = Article(
        title="Test Article",
        site="Test Site",
        url="https://example.com/test"
    )
    manager = EntryManager(articles_dir, article)

    assert manager.fetched_html.name == "fetched.html"
    assert manager.extracted_txt.name == "extracted.txt"
    assert manager.llm_summary.name == "llm_summary.json"
    assert manager.llm_debug.name == "llm_debug.jsonl"


def test_ensure_folder():
    """ensure_folder should create directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(
            title="Test",
            site="Test",
            url="https://example.com/test"
        )
        manager = EntryManager(articles_dir, article)

        manager.ensure_folder()
        assert manager.folder.exists()
        assert manager.folder.is_dir()


def test_slugify_basic():
    """slugify should convert text to URL-safe format"""
    from daily_feed.entry_manager import slugify

    assert slugify("Hello World") == "hello-world"
    assert slugify("Test Article!") == "test-article"
    assert slugify("Multiple   Spaces") == "multiple-spaces"


def test_slugify_special_chars():
    """slugify should handle special characters"""
    from daily_feed.entry_manager import slugify

    assert slugify("Hello@World!") == "hello-world"
    assert slugify("Test$%^&*Article") == "test-article"


def test_slugify_limit():
    """slugify should limit to 50 characters"""
    from daily_feed.entry_manager import slugify

    long_text = "a" * 100
    result = slugify(long_text)
    assert len(result) == 50


def test_slugify_strip_hyphens():
    """slugify should strip leading/trailing hyphens"""
    from daily_feed.entry_manager import slugify

    assert slugify("---Hello World---") == "hello-world"
    assert slugify("$$$Test$$$") == "test"


def test_short_hash():
    """short_hash should return first 5 chars of MD5 hash"""
    from daily_feed.entry_manager import short_hash

    # Known hash for this URL
    result = short_hash("https://example.com/openai-codex")
    assert result == "d3f76"
    assert len(result) == 5


def test_entry_folder_full_path():
    """Entry folder should be correctly placed under articles_dir"""
    articles_dir = Path("/tmp/test_articles")
    article = Article(
        title="Test Article",
        site="Test Site",
        url="https://example.com/test"
    )
    manager = EntryManager(articles_dir, article)

    # Full path should be articles_dir / slug-hash
    expected_parent = articles_dir
    assert manager.folder.parent == expected_parent


def test_file_paths_are_absolute():
    """File paths should be absolute paths"""
    articles_dir = Path("/tmp/test_articles")
    article = Article(
        title="Test",
        site="Test",
        url="https://example.com/test"
    )
    manager = EntryManager(articles_dir, article)

    # All file properties should return Path objects
    assert isinstance(manager.fetched_html, Path)
    assert isinstance(manager.extracted_txt, Path)
    assert isinstance(manager.llm_summary, Path)
    assert isinstance(manager.llm_debug, Path)
