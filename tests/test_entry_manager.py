"""Tests for EntryManager class."""

import os
import tempfile
import time
from pathlib import Path

from daily_feed.core.types import Article, ArticleSummary
from daily_feed.core.entry import EntryManager


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


def test_write_and_read_llm_summary():
    """Should write and read LLM summary JSON"""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Test", site="Test", url="https://example.com/test")
        manager = EntryManager(articles_dir, article)
        manager.ensure_folder()

        summary = ArticleSummary(
            article=article,
            bullets=["point 1", "point 2"],
            takeaway="Test takeaway",
            topic="AI",
            status="ok"
        )
        summary.meta = {"model": "test-model", "generated_at": "2026-02-05T00:00:00Z"}

        manager.write_llm_summary(summary)

        # Read it back
        loaded = manager.read_llm_summary()
        assert loaded["bullets"] == ["point 1", "point 2"]
        assert loaded["takeaway"] == "Test takeaway"
        assert loaded["topic"] == "AI"
        assert loaded["status"] == "ok"


def test_read_llm_summary_missing():
    """Should return None if summary file doesn't exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Test", site="Test", url="https://example.com/test")
        manager = EntryManager(articles_dir, article)

        assert manager.read_llm_summary() is None


def test_get_llm_logger_creates_logger():
    """get_llm_logger should create logger with JSONL formatter"""
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Test", site="Test", url="https://example.com/test")
        manager = EntryManager(articles_dir, article)
        manager.ensure_folder()

        logger = manager.get_llm_logger()

        assert logger is not None
        assert logger.name == f"daily_feed.entry.{manager.folder.name}"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)
        assert logger.propagate is False


def test_get_llm_logger_returns_none_if_no_folder():
    """get_llm_logger should return None if folder doesn't exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Test", site="Test", url="https://example.com/test")
        manager = EntryManager(articles_dir, article)

        # Don't create folder
        logger = manager.get_llm_logger()

        assert logger is None


def test_llm_logger_writes_jsonl():
    """LLM logger should write JSONL formatted logs"""
    import json
    import logging

    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Test", site="Test", url="https://example.com/test")
        manager = EntryManager(articles_dir, article)
        manager.ensure_folder()

        logger = manager.get_llm_logger()
        assert logger is not None

        # Log an event
        logger.info("Test event", extra={"key": "value", "number": 123})

        # Flush and close handler
        for handler in logger.handlers:
            handler.flush()
            handler.close()

        # Read log file
        log_content = manager.llm_debug.read_text(encoding="utf-8")
        lines = log_content.strip().split("\n")

        assert len(lines) == 1

        # Parse JSON line
        log_entry = json.loads(lines[0])
        assert log_entry["message"] == "Test event"
        assert log_entry["key"] == "value"
        assert log_entry["number"] == 123
        assert "timestamp" in log_entry
        assert log_entry["level"] == "INFO"
