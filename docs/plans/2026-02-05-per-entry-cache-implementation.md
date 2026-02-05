# Per-Entry Cache and Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure cache and logging to be organized per-entry (per article) with entry folders containing fetched.html, extracted.txt, llm_summary.json, and llm_debug.jsonl.

**Architecture:** Replace flat cache directory with per-entry folders under `articles/` directory. Each entry folder is named `{slug}-{shortHash}` and contains all cache files and LLM logs for that article. Global run.jsonl remains for pipeline-level events.

**Tech Stack:** Python 3.11+, pathlib, hashlib, pytest

---

## Task 1: Create EntryManager Class

**Files:**
- Create: `daily_feed/entry_manager.py`

**Step 1: Write the failing test**

Create `tests/test_entry_manager.py`:

```python
import pytest
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
    assert manager.folder.name == "openai-codex-app-launch-a1b2c"


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
    import tempfile
    import shutil

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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_entry_manager.py -v
```

Expected: `ModuleNotFoundError: No module named 'daily_feed.entry_manager'`

**Step 3: Write minimal implementation**

Create `daily_feed/entry_manager.py`:

```python
"""Entry manager for per-article cache and logging."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

from .types import Article


def slugify(text: str) -> str:
    """Convert text to URL-safe slug.

    Args:
        text: Input text to slugify

    Returns:
        Lowercase slug with hyphens replacing non-alphanumeric chars
    """
    # Convert to lowercase and replace non-alphanumeric with hyphens
    slug = re.sub(r'[^a-zA-Z0-9]+', '-', text.lower())
    # Remove leading/trailing hyphens and limit length
    slug = slug.strip('-')
    if len(slug) > 50:
        slug = slug[:50].rstrip('-')
    return slug


def short_hash(url: str) -> str:
    """Generate short hash from URL.

    Args:
        url: URL to hash

    Returns:
        First 5 characters of MD5 hash
    """
    return hashlib.md5(url.encode()).hexdigest()[:5]


class EntryManager:
    """Manages entry folder, cache files, and logs for a single article.

    Each article gets its own folder containing:
    - fetched.html: Raw HTML from fetch stage
    - extracted.txt: Extracted text content
    - llm_summary.json: LLM summary result
    - llm_debug.jsonl: LLM interaction logs
    """

    def __init__(self, articles_dir: Path, article: Article) -> None:
        """Initialize entry manager for an article.

        Args:
            articles_dir: Base directory for all article folders
            article: Article to manage
        """
        self.articles_dir = articles_dir
        self.article = article
        self._folder = self._entry_folder()

    def _entry_folder(self) -> Path:
        """Generate entry folder path: {slug}-{shortHash}/.

        Returns:
            Path to the entry folder
        """
        slug = slugify(self.article.title)
        hash_part = short_hash(self.article.url)
        return self.articles_dir / f"{slug}-{hash_part}"

    @property
    def folder(self) -> Path:
        """Get the entry folder path."""
        return self._folder

    @property
    def fetched_html(self) -> Path:
        """Path to fetched HTML cache file."""
        return self._folder / "fetched.html"

    @property
    def extracted_txt(self) -> Path:
        """Path to extracted text cache file."""
        return self._folder / "extracted.txt"

    @property
    def llm_summary(self) -> Path:
        """Path to LLM summary JSON file."""
        return self._folder / "llm_summary.json"

    @property
    def llm_debug(self) -> Path:
        """Path to LLM debug JSONL file."""
        return self._folder / "llm_debug.jsonl"

    def ensure_folder(self) -> None:
        """Create the entry folder if it doesn't exist."""
        self._folder.mkdir(parents=True, exist_ok=True)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_entry_manager.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add daily_feed/entry_manager.py tests/test_entry_manager.py
git commit -m "feat: add EntryManager class for per-article cache management"
```

---

## Task 2: Add Entry Validation Logic

**Files:**
- Modify: `daily_feed/entry_manager.py`
- Modify: `tests/test_entry_manager.py`

**Step 1: Write the failing test**

Add to `tests/test_entry_manager.py`:

```python
from datetime import datetime, timedelta, timezone


def test_is_entry_valid_no_ttl():
    """Entry should be valid if exists and TTL is None"""
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Test", site="Test", url="https://example.com/test")
        manager = EntryManager(articles_dir, article)
        manager.ensure_folder()

        # Should be valid when no TTL
        assert EntryManager.is_entry_valid(manager.folder, ttl_days=None)


def test_is_entry_valid_fresh():
    """Entry should be valid if within TTL"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Test", site="Test", url="https://example.com/test")
        manager = EntryManager(articles_dir, article)
        manager.ensure_folder()

        # Should be valid for fresh entry
        assert EntryManager.is_entry_valid(manager.folder, ttl_days=7)


def test_is_entry_valid_expired():
    """Entry should be invalid if older than TTL"""
    import tempfile
    import time

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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_entry_manager.py::test_is_entry_valid_no_ttl -v
```

Expected: `AttributeError: type object 'EntryManager' has no attribute 'is_entry_valid'`

**Step 3: Write minimal implementation**

Add to `daily_feed/entry_manager.py`:

```python
import os
from datetime import datetime, timezone


class EntryManager:
    # ... existing code ...

    @staticmethod
    def is_entry_valid(folder: Path, ttl_days: int | None) -> bool:
        """Check if entry cache is still valid based on TTL.

        Args:
            folder: Path to the entry folder
            ttl_days: TTL in days, or None for no expiration

        Returns:
            True if folder exists and is within TTL (or TTL is None)
        """
        if not folder.exists():
            return False
        if ttl_days is None:
            return True
        age_seconds = (datetime.now(timezone.utc).timestamp() -
                      folder.stat().st_mtime)
        return age_seconds <= ttl_days * 86400
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_entry_manager.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add daily_feed/entry_manager.py tests/test_entry_manager.py
git commit -m "feat: add entry validation with TTL support"
```

---

## Task 3: Add LLM Summary Read/Write Methods

**Files:**
- Modify: `daily_feed/entry_manager.py`
- Modify: `tests/test_entry_manager.py`

**Step 1: Write the failing test**

Add to `tests/test_entry_manager.py`:

```python
from daily_feed.types import ArticleSummary


def test_write_and_read_llm_summary():
    """Should write and read LLM summary JSON"""
    import tempfile

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
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)
        article = Article(title="Test", site="Test", url="https://example.com/test")
        manager = EntryManager(articles_dir, article)

        assert manager.read_llm_summary() is None
```

**Step 4: Run test to verify it fails**

```bash
pytest tests/test_entry_manager.py::test_write_and_read_llm_summary -v
```

Expected: `AttributeError: 'EntryManager' object has no attribute 'write_llm_summary'`

**Step 3: Write minimal implementation**

Add to `daily_feed/entry_manager.py`:

```python
import json


class EntryManager:
    # ... existing code ...

    def write_llm_summary(self, summary: ArticleSummary) -> None:
        """Write LLM summary to JSON file.

        Args:
            summary: ArticleSummary to write
        """
        self.ensure_folder()
        data = {
            "bullets": summary.bullets,
            "takeaway": summary.takeaway,
            "topic": summary.topic,
            "status": summary.status,
            "article": {
                "title": summary.article.title,
                "url": summary.article.url,
                "site": summary.article.site,
            },
        }
        # Include metadata if present
        if summary.meta:
            data["model"] = summary.meta.get("model", "")
            data["generated_at"] = summary.meta.get("generated_at", "")

        with self.llm_summary.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)

    def read_llm_summary(self) -> dict | None:
        """Read LLM summary from JSON file.

        Returns:
            Summary dict, or None if file doesn't exist
        """
        if not self.llm_summary.exists():
            return None
        with self.llm_summary.open("r", encoding="utf-8") as f:
            return json.load(f)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_entry_manager.py::test_write_and_read_llm_summary -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add daily_feed/entry_manager.py tests/test_entry_manager.py
git commit -m "feat: add LLM summary read/write methods to EntryManager"
```

---

## Task 4: Update Runner to Use EntryManager

**Files:**
- Modify: `daily_feed/runner.py`

**Step 1: Update imports**

Add at top of `daily_feed/runner.py`:

```python
from .entry_manager import EntryManager
```

**Step 2: Modify run_pipeline function**

Replace the cache directory setup with articles directory:

```python
def run_pipeline(
    input_path: Path,
    output_dir: Path,
    cfg: AppConfig,
    show_progress: bool = True,
    console: Console | None = None,
) -> Path:
    # ... existing code ...

    run_output_dir = _build_run_output_dir(output_dir, input_path, cfg)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # NEW: Use articles directory instead of cache
    articles_dir = run_output_dir / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)

    # Keep global logger for pipeline events
    logger = setup_logging(cfg.logging, run_output_dir)
    # Remove llm_logger - will be handled per-entry
    setup_langfuse(cfg.langfuse)
```

**Step 3: Commit**

```bash
git add daily_feed/runner.py
git commit -m "refactor: update runner to use articles directory"
```

---

## Task 5: Update Fetch Logic to Use EntryManager

**Files:**
- Modify: `daily_feed/runner.py`

**Step 1: Modify fetch function signature**

Update `_fetch_articles` and `_fetch_single_httpx` to accept and use EntryManager:

```python
def _fetch_articles(
    articles,
    articles_dir: Path,  # Changed from cache_dir
    cfg: AppConfig,
    logger,
    progress: Progress | None = None,
    fetch_task: int | None = None,
) -> tuple[list[ExtractedArticle], FetchStats]:
    """Fetch and extract content for all articles.

    Args:
        articles: List of Article objects to fetch
        articles_dir: Directory for entry folders
        cfg: Application configuration
        logger: Logger for events
        progress: Optional Rich progress bar
        fetch_task: Task ID for progress updates

    Returns:
        Tuple of (extracted articles, fetch statistics)
    """
    stats = FetchStats(total=len(articles))
    backend = (cfg.fetch.backend or "httpx").lower()
    if backend == "crawl4ai":
        extracted = _fetch_and_extract_crawl4ai(
            articles, articles_dir, cfg, stats, logger, progress, fetch_task
        )
        return extracted, stats
    extracted = _fetch_and_extract_httpx(
        articles, articles_dir, cfg, stats, logger, progress, fetch_task
    )
    return extracted, stats
```

**Step 2: Update _fetch_single_httpx**

```python
def _fetch_single_httpx(
    article,
    articles_dir: Path,
    cfg: AppConfig,
    stats: FetchStats,
    logger,
) -> ExtractedArticle:
    """Fetch a single article using httpx and extract content.

    Args:
        article: Article to fetch
        articles_dir: Directory for entry folders
        cfg: Application configuration
        stats: Statistics object to update
        logger: Logger for events

    Returns:
        ExtractedArticle with text or error populated
    """
    entry = EntryManager(articles_dir, article)
    entry.ensure_folder()

    # Check for cached extracted text (fastest path)
    if entry.extracted_txt.exists() and EntryManager.is_entry_valid(
        entry.folder, cfg.cache.ttl_days
    ):
        text = entry.extracted_txt.read_text(encoding="utf-8")
        stats.cache_hits += 1
        log_event(
            logger,
            "Cache hit",
            event="cache_hit",
            url=article.url,
            title=article.title,
            cache_type="extracted_txt",
        )
        return ExtractedArticle(article=article, text=text)

    # Check for cached HTML
    if entry.fetched_html.exists() and EntryManager.is_entry_valid(
        entry.folder, cfg.cache.ttl_days
    ):
        html = entry.fetched_html.read_text(encoding="utf-8", errors="ignore")
    else:
        log_event(
            logger,
            "Fetch start",
            event="fetch_start",
            url=article.url,
            title=article.title,
            backend="httpx",
        )
        result = fetch_url(
            article.url,
            timeout=cfg.fetch.timeout_seconds,
            retries=cfg.fetch.retries,
            user_agent=cfg.fetch.user_agent,
            trust_env=cfg.fetch.trust_env,
        )
        if result.text:
            html = result.text
            entry.fetched_html.write_text(html, encoding="utf-8")
        else:
            error_category = _categorize_error(result.error, result.status_code)
            stats.httpx_failed += 1
            log_event(
                logger,
                "Fetch failed",
                event="fetch_failed",
                url=article.url,
                title=article.title,
                backend="httpx",
                error=result.error,
                status_code=result.status_code,
                error_category=error_category,
            )
            return ExtractedArticle(article=article, text=None, error=result.error)

    text = extract_text(html, cfg.extract.primary, cfg.extract.fallback)
    if text and _is_placeholder_text(text):
        text = None
    if text:
        entry.extracted_txt.write_text(text, encoding="utf-8")
        stats.httpx_success += 1
        log_event(
            logger,
            "Extract success",
            event="extract_success",
            url=article.url,
            title=article.title,
        )
        return ExtractedArticle(article=article, text=text, error=None)
    html_size = len(html) if html else 0
    stats.httpx_failed += 1
    log_event(
        logger,
        "Extract failed",
        event="extract_failed",
        url=article.url,
        title=article.title,
        error="Empty extraction result",
        html_size=html_size,
        extraction_methods=[cfg.extract.primary] + cfg.extract.fallback,
    )
    return ExtractedArticle(article=article, text=None, error="Empty extraction result")
```

**Step 3: Commit**

```bash
git add daily_feed/runner.py
git commit -m "refactor: update fetch logic to use EntryManager"
```

---

## Task 6: Update Crawl4AI Fetch

**Files:**
- Modify: `daily_feed/runner.py`

**Step 1: Update _fetch_and_extract_crawl4ai_async**

Similar changes to Task 5, update the async fetch function:

```python
async def _fetch_and_extract_crawl4ai_async(
    articles,
    articles_dir: Path,
    cfg: AppConfig,
    stats: FetchStats,
    logger,
    progress: Progress | None = None,
    fetch_task: int | None = None,
) -> list[ExtractedArticle]:
    # ... existing setup code ...

    async def _fetch_single(article) -> ExtractedArticle:
        entry = EntryManager(articles_dir, article)
        entry.ensure_folder()

        text_cache = entry.extracted_txt  # Changed from cache_path
        if text_cache.exists() and EntryManager.is_entry_valid(
            entry.folder, cfg.cache.ttl_days
        ):
            text = text_cache.read_text(encoding="utf-8")
            stats.cache_hits += 1
            await _advance_progress()
            return ExtractedArticle(article=article, text=text)

        async with semaphore:
            log_event(
                logger,
                "Fetch start",
                event="fetch_start",
                url=article.url,
                title=article.title,
                backend="crawl4ai",
            )
            result = await fetch_url_crawl4ai(
                article.url,
                timeout=cfg.fetch.timeout_seconds,
                retries=cfg.fetch.retries,
            )

        text = result.text
        if text and _is_placeholder_text(text):
            text = None

        if text:
            text_cache.write_text(text, encoding="utf-8")
            stats.crawl4ai_success += 1
            await _advance_progress()
            return ExtractedArticle(article=article, text=text, error=None)

        # ... rest of error handling ...

    # ... rest of function ...
```

**Step 2: Commit**

```bash
git add daily_feed/runner.py
git commit -m "refactor: update crawl4ai fetch to use EntryManager"
```

---

## Task 7: Update Provider to Use Per-Entry Logging

**Files:**
- Modify: `daily_feed/providers/gemini.py`
- Modify: `daily_feed/entry_manager.py`

**Step 1: Add logger setup to EntryManager**

Add to `daily_feed/entry_manager.py`:

```python
import logging
from .logging_utils import JsonlFormatter


class EntryManager:
    # ... existing code ...

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
```

**Step 2: Update GeminiProvider**

Modify `daily_feed/providers/gemini.py` to accept optional entry_logger parameter:

```python
def summarize_article(
    self,
    article: Article,
    text: str,
    entry_logger: logging.Logger | None = None,
) -> ArticleSummary:
    """Generate summary for a single article.

    Args:
        article: Article to summarize
        text: Extracted article text
        entry_logger: Optional per-entry logger for LLM interactions

    Returns:
        ArticleSummary with bullets and takeaway
    """
    logger = entry_logger or self.llm_logger
    # ... rest of implementation using logger ...
```

**Step 3: Update runner to pass entry logger**

In `daily_feed/runner.py`, modify the summarization loop:

```python
for item in extracted:
    entry = EntryManager(articles_dir, item.article)
    entry.ensure_folder()
    entry_logger = entry.get_llm_logger()

    if not item.text:
        summary_text = item.article.summary or ""
        summaries.append(
            ArticleSummary(
                article=item.article,
                bullets=["Summary unavailable; used RSS summary."],
                takeaway=summary_text,
                status="summary_only",
            )
        )
        progress.advance(summarize_task, 1)
        continue

    summary = provider.summarize_article(item.article, item.text, entry_logger)

    # Save summary to entry folder
    entry.write_llm_summary(summary)

    if summary.status == "parse_error":
        log_event(
            logger,
            "LLM parse error",
            event="llm_parse_error",
            url=item.article.url,
            title=item.article.title,
        )
    summaries.append(summary)
    progress.advance(summarize_task, 1)
```

**Step 4: Commit**

```bash
git add daily_feed/entry_manager.py daily_feed/providers/gemini.py daily_feed/runner.py
git commit -m "feat: add per-entry LLM logging"
```

---

## Task 8: Remove Deprecated Cache Code

**Files:**
- Modify: `daily_feed/runner.py`
- Optionally: Keep `cache.py` for backward compatibility or mark deprecated

**Step 1: Remove unused imports from runner.py**

Remove or comment out:
```python
# from .cache import CacheIndex  # Deprecated: using EntryManager
```

**Step 2: Remove _build_cache_dir function**

This function is no longer needed since we use `articles_dir` directly.

**Step 3: Commit**

```bash
git add daily_feed/runner.py
git commit -m "refactor: remove deprecated cache directory logic"
```

---

## Task 9: Update Config Documentation

**Files:**
- Modify: `config.yaml`
- Modify: `config.example.yaml`

**Step 1: Update cache config section**

Update documentation to reflect new structure:

```yaml
cache:
  mode: run  # Options: run, shared (shared still uses old structure)
  ttl_days: null  # Cache TTL in days, null = no expiration
  # Note: Cache is now organized as articles/{slug}-{hash}/
  # with fetched.html, extracted.txt, llm_summary.json, llm_debug.jsonl
```

**Step 2: Commit**

```bash
git add config.yaml config.example.yaml
git commit -m "docs: update config documentation for per-entry cache"
```

---

## Task 10: Integration Tests

**Files:**
- Create: `tests/test_integration_per_entry.py`

**Step 1: Write integration test**

```python
"""Integration tests for per-entry cache and logging."""

import json
import tempfile
from pathlib import Path

from daily_feed.entry_manager import EntryManager
from daily_feed.types import Article, ArticleSummary


def test_full_pipeline_with_entry_manager():
    """Test full workflow: fetch -> extract -> summarize with entry folders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)

        # Create article
        article = Article(
            title="Test AI Article",
            site="Tech News",
            url="https://example.com/ai-article"
        )
        entry = EntryManager(articles_dir, article)
        entry.ensure_folder()

        # Simulate fetch - write HTML
        html = "<html><body><p>AI is advancing rapidly in 2026.</p></body></html>"
        entry.fetched_html.write_text(html, encoding="utf-8")

        # Simulate extract - write text
        text = "AI is advancing rapidly in 2026. New models are released."
        entry.extracted_txt.write_text(text, encoding="utf-8")

        # Simulate LLM summary
        summary = ArticleSummary(
            article=article,
            bullets=["AI models advancing in 2026", "New releases happening"],
            takeaway="AI technology is progressing quickly.",
            topic="AI",
            status="ok"
        )
        summary.meta = {"model": "test-model", "generated_at": "2026-02-05T00:00:00Z"}
        entry.write_llm_summary(summary)

        # Verify all files exist
        assert entry.fetched_html.exists()
        assert entry.extracted_txt.exists()
        assert entry.llm_summary.exists()

        # Verify summary content
        loaded = entry.read_llm_summary()
        assert loaded["bullets"] == ["AI models advancing in 2026", "New releases happening"]
        assert loaded["takeaway"] == "AI technology is progressing quickly."
        assert loaded["topic"] == "AI"


def test_entry_folder_naming_consistency():
    """Same article always produces same folder name."""
    articles_dir = Path("/tmp/test")

    article = Article(
        title="OpenAI Codex Launch",
        site="Tech",
        url="https://example.com/codex"
    )

    entry1 = EntryManager(articles_dir, article)
    entry2 = EntryManager(articles_dir, article)

    assert entry1.folder.name == entry2.folder.name
```

**Step 2: Run integration tests**

```bash
pytest tests/test_integration_per_entry.py -v
```

Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_integration_per_entry.py
git commit -m "test: add integration tests for per-entry cache"
```

---

## Task 11: End-to-End Test

**Files:**
- Manual test with real data

**Step 1: Run pipeline with test data**

```bash
# Using existing test data
daily-feed run --input data/folo-export-2026-02-03.json --output out/test-per-entry
```

**Step 2: Verify output structure**

```bash
ls -la out/test-per-entry/
ls -la out/test-per-entry/articles/
ls -la out/test-per-entry/articles/*/
cat out/test-per-entry/articles/*/llm_summary.json | head -20
```

Expected:
- `articles/` directory exists with entry folders
- Each entry folder has `fetched.html`, `extracted.txt`, `llm_summary.json`, `llm_debug.jsonl`
- `report.html` still generates correctly

**Step 3: Verify report renders**

Open `out/test-per-entry/report.html` in browser and verify it displays correctly.

**Step 4: Commit**

```bash
git add -A
git commit -m "test: verify end-to-end pipeline with per-entry cache"
```

---

## Task 12: Update README

**Files:**
- Modify: `README.md`

**Step 1: Update cache documentation**

Add section explaining new cache structure:

```markdown
## Cache Structure

Articles are cached in per-entry folders under `articles/`:

```
out/run-20260205/
├── report.html
├── run.jsonl              # Pipeline-level events
└── articles/              # Per-article cache
    ├── article-one-a1b2c/
    │   ├── fetched.html       # Raw HTML
    │   ├── extracted.txt      # Extracted text
    │   ├── llm_summary.json   # Summary result
    │   └── llm_debug.jsonl    # LLM logs
    └── article-two-c3d4e/
        └── ...
```

Each entry folder is named `{slug}-{shortHash}` where:
- `slug`: URL-safe version of article title
- `shortHash`: First 5 chars of MD5 hash of URL
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with per-entry cache documentation"
```

---

## Summary

This implementation:
1. ✅ Creates `EntryManager` class for per-article cache management
2. ✅ Organizes cache as `articles/{slug}-{shortHash}/` folders
3. ✅ Separates LLM logs per-entry instead of global
4. ✅ Keeps global `run.jsonl` for pipeline events
5. ✅ Maintains TTL validation for cache freshness
6. ✅ Provides clean API for reading/writing entry files

**Total commits:** ~12 commits for incremental, testable progress
