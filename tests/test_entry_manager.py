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
