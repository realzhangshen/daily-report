"""Tests for cache enable/disable behavior."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from daily_feed.config import AppConfig
from daily_feed.core.entry import EntryManager
from daily_feed.core.types import AnalysisResult, Article
from daily_feed.fetch.fetcher import FetchResult
from daily_feed import runner


def test_fetch_uses_cached_text_when_cache_enabled(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Cached Item", site="Test", url="https://example.com/cached")
        entry = EntryManager(articles_dir, article)
        entry.ensure_folder()
        entry.extracted_txt.write_text("cached text", encoding="utf-8")

        cfg = AppConfig()
        cfg.fetch.crawl4ai_api_url = "https://crawl4ai.example.com"
        cfg.cache.enabled = True

        called = 0

        async def fake_fetch(*args, **kwargs):
            nonlocal called
            called += 1
            return FetchResult(url=args[0], status_code=200, text="fetched text", error=None)

        monkeypatch.setattr(runner, "fetch_url_crawl4ai_api", fake_fetch)

        stats = runner.FetchStats(total=1)
        logger = logging.getLogger("test_cache_enabled")
        extracted = asyncio.run(
            runner._fetch_and_extract_api_async(  # noqa: SLF001
                [article], articles_dir, cfg, stats, logger
            )
        )

        assert called == 0
        assert stats.cache_hits == 1
        assert stats.api_success == 0
        assert extracted[0].text == "cached text"


def test_fetch_bypasses_cache_when_disabled(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="No Cache Item", site="Test", url="https://example.com/no-cache")
        entry = EntryManager(articles_dir, article)
        entry.ensure_folder()
        entry.extracted_txt.write_text("stale cache text", encoding="utf-8")

        cfg = AppConfig()
        cfg.fetch.crawl4ai_api_url = "https://crawl4ai.example.com"
        cfg.cache.enabled = False

        called = 0

        async def fake_fetch(*args, **kwargs):
            nonlocal called
            called += 1
            return FetchResult(url=args[0], status_code=200, text="fresh text " * 30, error=None)

        monkeypatch.setattr(runner, "fetch_url_crawl4ai_api", fake_fetch)

        stats = runner.FetchStats(total=1)
        logger = logging.getLogger("test_cache_disabled")
        extracted = asyncio.run(
            runner._fetch_and_extract_api_async(  # noqa: SLF001
                [article], articles_dir, cfg, stats, logger
            )
        )

        assert called == 1
        assert stats.cache_hits == 0
        assert stats.api_success == 1
        assert extracted[0].text == "fresh text " * 30
        assert entry.extracted_txt.read_text(encoding="utf-8") == "stale cache text"


def test_analysis_cache_respects_cache_enabled_switch():
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Analysis Item", site="Test", url="https://example.com/analysis")
        entry = EntryManager(articles_dir, article)
        entry.ensure_folder()
        entry.write_analysis_result(
            AnalysisResult(
                article=article,
                analysis="cached analysis",
                status="ok",
                meta={"model": "test-model"},
            )
        )

        cfg = AppConfig()
        cfg.cache.enabled = True
        cached = runner._read_cached_analysis(entry, cfg)  # noqa: SLF001
        assert cached is not None
        assert cached.get("analysis") == "cached analysis"

        cfg.cache.enabled = False
        cached_disabled = runner._read_cached_analysis(entry, cfg)  # noqa: SLF001
        assert cached_disabled is None
