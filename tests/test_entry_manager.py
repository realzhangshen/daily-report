"""Tests for EntryManager class."""

import os
import tempfile
import time
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


def test_is_entry_valid_no_ttl():
    """Entry should be valid if exists and TTL is None"""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Test", site="Test", url="https://example.com/test")
        manager = EntryManager(articles_dir, article)
        manager.ensure_folder()

        # Should be valid when no TTL
        assert EntryManager.is_entry_valid(manager.folder, ttl_days=None)


def test_is_entry_valid_fresh():
    """Entry should be valid if within TTL"""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Test", site="Test", url="https://example.com/test")
        manager = EntryManager(articles_dir, article)
        manager.ensure_folder()

        # Should be valid for fresh entry
        assert EntryManager.is_entry_valid(manager.folder, ttl_days=7)


def test_is_entry_valid_expired():
    """Entry should be invalid if older than TTL"""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Test", site="Test", url="https://example.com/test")
        manager = EntryManager(articles_dir, article)
        manager.ensure_folder()

        # Modify folder mtime to be older than TTL
        old_time = time.time() - (8 * 86400)  # 8 days ago
        os.utime(manager.folder, (old_time, old_time))

        # Should be invalid with 7 day TTL
        assert not EntryManager.is_entry_valid(manager.folder, ttl_days=7)


def test_is_entry_valid_missing():
    """Missing entry should be invalid"""
    assert not EntryManager.is_entry_valid(Path("/nonexistent/folder"), ttl_days=7)
