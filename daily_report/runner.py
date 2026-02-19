"""
Main pipeline orchestration for the Daily Feed Agent.

This module coordinates the entire workflow:
1. Parse JSON input
2. Deduplicate articles
3. Fetch and extract content (via remote Crawl4AI API only)
4. Extract structured metadata per entry (Pass 1)
5. Synthesize daily briefing (Pass 2)
6. Render output files

Supports both progress bar and quiet modes. Fetching is done exclusively
through the remote Crawl4AI API service.

Cross-repo contract notes:
- Input file is expected to follow folo_exporter JSON schema.
- Generated report folder is expected to be publishable into
  web_dailyreport/reports/<date>/.
"""

from __future__ import annotations

import logging
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .config import AppConfig
from .core.dedup import dedup_articles
from .core.entry import EntryManager
from .core.types import ExtractionResult, ExtractedArticle
from .config import get_crawl4ai_api_url, get_crawl4ai_api_auth
from .fetch.fetcher import fetch_url_crawl4ai_api
from .utils.logging import log_event, setup_logging
from .input.json_parser import parse_folo_json
from .llm.tracing import set_span_output, setup_langfuse, start_span
from .llm.providers import create_provider
from .analyzers.entry_analyzer import EntryAnalyzer
from .renderer import render_briefing, render_html, render_markdown
from .output.index_sync import sync_report_to_web_index
from .analyzers.synthesizer import Synthesizer


def _is_cache_enabled(cfg: AppConfig) -> bool:
    """Return whether cache read/write is enabled in config."""
    return cfg.cache.enabled


def _read_cached_extraction(entry: EntryManager, cfg: AppConfig) -> dict | None:
    """Read cached extraction only when cache is enabled and valid."""
    if not _is_cache_enabled(cfg):
        return None
    cached = entry.read_extraction_result()
    if cached and EntryManager.is_entry_valid(entry.folder, cfg.cache.ttl_days):
        return cached
    return None


def _build_cached_extraction(item: ExtractedArticle, cached: dict) -> ExtractionResult:
    # Rehydrate ExtractionResult from cache while keeping unknown keys in meta.
    # This preserves backward/forward compatibility when extraction schema grows.
    return ExtractionResult(
        article=item.article,
        one_line_summary=cached.get("one_line_summary", item.article.title),
        category=cached.get("category", "other"),
        tags=cached.get("tags", []),
        importance=cached.get("importance", 3),
        content_type=cached.get("content_type", "news"),
        key_takeaway=cached.get("key_takeaway", ""),
        status=cached.get("status", "ok"),
        meta={k: v for k, v in cached.items() if k not in {
            "one_line_summary", "category", "tags", "importance",
            "content_type", "key_takeaway", "status", "article"
        }},
    )


def _extract_entries(
    extracted: list[ExtractedArticle],
    articles_dir: Path,
    cfg: AppConfig,
    analyzer: EntryAnalyzer,
    logger,
    progress: Progress | None = None,
    extract_task: int | None = None,
) -> list[ExtractionResult]:
    concurrency = max(1, int(cfg.summary.analysis_concurrency))
    if concurrency > 1:
        log_event(logger, "Extract concurrency enabled", event="extract_concurrency_enabled", workers=concurrency)

    def _advance_progress() -> None:
        if progress and extract_task is not None:
            progress.advance(extract_task, 1)

    if concurrency == 1:
        # Sequential path is easier to debug and deterministic for local runs.
        results: list[ExtractionResult] = []
        for item in extracted:
            entry = EntryManager(articles_dir, item.article)
            cached = _read_cached_extraction(entry, cfg)
            if cached:
                result = _build_cached_extraction(item, cached)
                log_event(logger, "Extraction cache hit", event="extraction_cache_hit", url=item.article.url)
            else:
                result = analyzer.extract(item)
            results.append(result)
            _advance_progress()
        return results

    results_list: list[ExtractionResult | None] = [None] * len(extracted)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {}
        for idx, item in enumerate(extracted):
            entry = EntryManager(articles_dir, item.article)
            cached = _read_cached_extraction(entry, cfg)
            if cached:
                results_list[idx] = _build_cached_extraction(item, cached)
                log_event(logger, "Extraction cache hit", event="extraction_cache_hit", url=item.article.url)
                _advance_progress()
                continue
            # Copy current context (including tracing ids) into worker thread.
            ctx = copy_context()
            future = executor.submit(ctx.run, analyzer.extract, item)
            future_map[future] = idx

        for future in as_completed(future_map):
            idx = future_map[future]
            # Put result back to original index to keep output ordering stable.
            results_list[idx] = future.result()
            _advance_progress()

    if any(r is None for r in results_list):
        raise RuntimeError("Extraction results incomplete")
    return [r for r in results_list if r is not None]


def _categorize_error(error: str | None, status_code: int | None) -> str:
    """Categorize fetch errors for better logging.

    Args:
        error: Error message from fetch attempt
        status_code: HTTP status code if available

    Returns:
        Error category: "network_failed", "blocked", "timeout", "unknown"
    """
    if not error:
        return "unknown"
    error_lower = error.lower()
    if "timeout" in error_lower or "timed out" in error_lower:
        return "timeout"
    if "403" in str(status_code) or "blocked" in error_lower:
        return "blocked"
    if "connect" in error_lower or "connection" in error_lower:
        return "network_failed"
    return "unknown"


@dataclass
class FetchStats:
    """Statistics collected during the fetch/extract stage.

    Tracks success/failure rates and cache hit rate for performance monitoring.

    Attributes:
        total: Total number of articles to fetch
        cache_hits: Number served from cache
        api_success: Successful API fetches
        api_failed: Failed API fetches
    """
    total: int = 0
    cache_hits: int = 0
    api_success: int = 0
    api_failed: int = 0


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    cfg: AppConfig,
    show_progress: bool = True,
    console: Console | None = None,
) -> Path:
    """Run the complete daily feed processing pipeline.

    Orchestrates all stages from parsing to rendering, with optional
    progress bar display. Returns the path to the generated HTML report.

    Args:
        input_path: Path to the input markdown file (Folo format)
        output_dir: Directory for output files
        cfg: Application configuration
        show_progress: Whether to display progress bars
        console: Rich console for output (creates default if None)

    Returns:
        Path to the generated HTML report file
    """
    # Build deterministic run directory first so all subsequent artifacts
    # (report + per-article cache + logs) live under one folder.
    run_output_dir = _build_run_output_dir(output_dir, input_path, cfg)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    articles_dir = run_output_dir / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(cfg.logging, run_output_dir)
    # llm_logger is configured at entry level to keep per-article traces separate.
    setup_langfuse(cfg.langfuse)

    with start_span(
        "daily_report.run",
        kind="chain",
        input_value={"input_path": str(input_path), "output_dir": str(run_output_dir)},
        attributes={"run_folder_mode": cfg.output.run_folder_mode},
    ) as run_span:
        log_event(
            logger,
            "Pipeline start",
            event="pipeline_start",
            input=str(input_path),
            output=str(run_output_dir),
            articles_dir=str(articles_dir),
        )

        # Quiet mode is preferred in automation logs where progress bars
        # generate noisy output.
        if not show_progress:
            # Load JSON input file
            with open(input_path, encoding="utf-8") as f:
                data = json.load(f)

            articles = parse_folo_json(data)

            if cfg.dedup.enabled:
                articles = dedup_articles(articles, cfg.dedup.title_similarity_threshold)

            with start_span(
                "daily_report.fetch_extract",
                kind="chain",
                input_value={"count": len(articles)},
            ):
                extracted, stats = _fetch_articles(
                    articles, articles_dir, cfg, logger
                )
            _render_fetch_stats(stats, console or Console())

            # Stage 4: Extract (Pass 1) - structured per-entry metadata.
            provider = _build_provider(cfg, None)
            analyzer = EntryAnalyzer(cfg, provider, articles_dir, logger)
            with start_span(
                "daily_report.extract_batch",
                kind="chain",
                input_value={"count": len(extracted)},
            ):
                extractions = _extract_entries(extracted, articles_dir, cfg, analyzer, logger)

            # Stage 5: Synthesize (Pass 2) - aggregate daily briefing.
            synthesizer = Synthesizer(cfg, provider, logger)
            briefing = synthesizer.synthesize(extractions)

            # Stage 6: Render outputs and sync web index best-effort.
            title = f"Daily Feed - {input_path.stem}"
            html_path = run_output_dir / "report.html"
            render_briefing(briefing, extractions, html_path, title)
            _sync_web_report_index(html_path=html_path, title=title, input_path=input_path, logger=logger)

            log_event(
                logger,
                "Pipeline complete",
                event="pipeline_complete",
                output=str(html_path),
                total=len(extractions),
            )
            set_span_output(
                run_span, {"report": str(html_path), "total": len(extractions)}
            )
            return html_path

        console = console or Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            # Keep stage counter aligned with the conceptual six pipeline phases.
            stage_task = progress.add_task("Stages", total=6)

            # Load JSON input file
            with open(input_path, encoding="utf-8") as f:
                data = json.load(f)

            articles = parse_folo_json(data)
            progress.advance(stage_task, 1)

            if cfg.dedup.enabled:
                articles = dedup_articles(articles, cfg.dedup.title_similarity_threshold)
            progress.advance(stage_task, 1)

            fetch_task = progress.add_task("Fetch + Extract", total=len(articles))
            with start_span(
                "daily_report.fetch_extract",
                kind="chain",
                input_value={"count": len(articles)},
            ):
                extracted, stats = _fetch_articles(
                    articles, articles_dir, cfg, logger, progress, fetch_task
                )
            _render_fetch_stats(stats, console)
            progress.advance(stage_task, 1)

            # Stage 4: Extract (Pass 1) - run entry analyzer.
            provider = _build_provider(cfg, None)
            analyzer = EntryAnalyzer(cfg, provider, articles_dir, logger)

            extract_task = progress.add_task("Extract", total=len(extracted))
            with start_span(
                "daily_report.extract_batch",
                kind="chain",
                input_value={"count": len(extracted)},
            ):
                extractions = _extract_entries(
                    extracted,
                    articles_dir,
                    cfg,
                    analyzer,
                    logger,
                    progress=progress,
                    extract_task=extract_task,
                )
            progress.advance(stage_task, 1)

            # Stage 5: Synthesize (Pass 2) - build final briefing.
            synthesizer = Synthesizer(cfg, provider, logger)
            briefing = synthesizer.synthesize(extractions)

            # Stage 6: Render to report.html and optionally sync index.
            title = f"Daily Feed - {input_path.stem}"
            html_path = run_output_dir / "report.html"
            render_briefing(briefing, extractions, html_path, title)
            _sync_web_report_index(html_path=html_path, title=title, input_path=input_path, logger=logger)

            progress.advance(stage_task, 1)
            log_event(
                logger,
                "Pipeline complete",
                event="pipeline_complete",
                output=str(html_path),
                total=len(extractions),
            )
            set_span_output(
                run_span, {"report": str(html_path), "total": len(extractions)}
            )

        return html_path


def _fetch_and_extract(articles, articles_dir: Path, cfg: AppConfig):
    """Convenience function to fetch and extract without progress tracking.

    Used by tests or other code that doesn't need the full pipeline.

    Args:
        articles: List of Article objects to fetch
        articles_dir: Articles directory for storing fetched content
        cfg: Application configuration

    Returns:
        List of ExtractedArticle objects
    """
    logger = logging.getLogger("daily_report")
    extracted, _stats = _fetch_articles(articles, articles_dir, cfg, logger)
    return extracted


def _fetch_articles(
    articles,
    articles_dir: Path,
    cfg: AppConfig,
    logger,
    progress: Progress | None = None,
    fetch_task: int | None = None,
) -> tuple[list[ExtractedArticle], FetchStats]:
    """Fetch and extract content for all articles using remote Crawl4AI API.

    Args:
        articles: List of Article objects to fetch
        articles_dir: Articles directory for storing fetched content
        cfg: Application configuration
        logger: Logger for events
        progress: Optional Rich progress bar
        fetch_task: Task ID for progress updates

    Returns:
        Tuple of (extracted articles, fetch statistics)
    """
    stats = FetchStats(total=len(articles))
    extracted = _fetch_and_extract_api(
        articles, articles_dir, cfg, stats, logger, progress, fetch_task
    )
    return extracted, stats


def _fetch_and_extract_api(
    articles,
    articles_dir: Path,
    cfg: AppConfig,
    stats: FetchStats,
    logger,
    progress: Progress | None = None,
    fetch_task: int | None = None,
) -> list[ExtractedArticle]:
    """Fetch and extract using remote Crawl4AI API (async concurrent processing).

    Uses asyncio to process multiple articles concurrently.

    Args:
        articles: List of Article objects to fetch
        articles_dir: Articles directory for storing fetched content
        cfg: Application configuration
        stats: Statistics object to update
        logger: Logger for events
        progress: Optional Rich progress bar
        fetch_task: Task ID for progress updates

    Returns:
        List of ExtractedArticle objects
    """
    # Keep async implementation isolated while exposing sync API to callers.
    return asyncio.run(
        _fetch_and_extract_api_async(
            articles, articles_dir, cfg, stats, logger, progress, fetch_task
        )
    )


async def _fetch_and_extract_api_async(
    articles,
    articles_dir: Path,
    cfg: AppConfig,
    stats: FetchStats,
    logger,
    progress: Progress | None = None,
    fetch_task: int | None = None,
) -> list[ExtractedArticle]:
    """Async implementation of Crawl4AI API fetching with concurrent processing.

    Creates tasks for all articles and runs them concurrently.

    Args:
        articles: List of Article objects to fetch
        articles_dir: Articles directory for storing fetched content
        cfg: Application configuration
        stats: Statistics object to update
        logger: Logger for events
        progress: Optional Rich progress bar
        fetch_task: Task ID for progress updates

    Returns:
        List of ExtractedArticle objects
    """
    api_url = get_crawl4ai_api_url(cfg.fetch)
    if not api_url:
        raise ValueError(
            "Crawl4AI API URL is required. Set CRAWL4AI_API_URL environment variable "
            "or configure crawl4ai_api_url in config."
        )

    api_auth = get_crawl4ai_api_auth(cfg.fetch)

    # Rich progress updates are not thread-safe; serialize advances.
    progress_lock = asyncio.Lock()

    async def _advance_progress() -> None:
        if progress and fetch_task is not None:
            async with progress_lock:
                progress.advance(fetch_task, 1)

    async def _fetch_single(article) -> ExtractedArticle:
        entry = EntryManager(articles_dir, article)
        entry.ensure_folder()

        # Check for cached extracted text (fastest path)
        if _is_cache_enabled(cfg) and entry.extracted_txt.exists() and EntryManager.is_entry_valid(
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
            await _advance_progress()
            return ExtractedArticle(article=article, text=text)

        log_event(
            logger,
            "Fetch start",
            event="fetch_start",
            url=article.url,
            title=article.title,
            backend="crawl4ai_api",
        )

        result = await fetch_url_crawl4ai_api(
            article.url,
            api_url=api_url,
            timeout=cfg.fetch.timeout_seconds,
            retries=cfg.fetch.retries,
            user_agent=cfg.fetch.user_agent,
            stealth=cfg.fetch.crawl4ai_stealth,
            delay=cfg.fetch.crawl4ai_delay,
            simulate_user=cfg.fetch.crawl4ai_simulate_user,
            magic=cfg.fetch.crawl4ai_magic,
            auth=api_auth,
        )

        # result.text is already extracted text/markdown from Crawl4AI API.
        text = result.text
        if text and _is_placeholder_text(text):
            text = None

        if text:
            if _is_cache_enabled(cfg):
                entry.extracted_txt.write_text(text, encoding="utf-8")
            stats.api_success += 1
            await _advance_progress()
            return ExtractedArticle(article=article, text=text, error=None)

        # Distinguish between transport failures and empty extraction outcomes;
        # this split is important for operational dashboards and retries.
        if result.error:
            # Network/protocol error during fetch
            error_category = _categorize_error(result.error, result.status_code)
            stats.api_failed += 1
            log_event(
                logger,
                "Fetch failed",
                event="fetch_failed",
                url=article.url,
                title=article.title,
                backend="crawl4ai_api",
                error=result.error,
                status_code=result.status_code,
                error_category=error_category,
            )
        else:
            # Fetch succeeded but extraction produced empty content
            # Note: Crawl4AI returns pre-extracted markdown/text
            # When text is empty but no error, it means extraction failed
            markdown_size = 0  # No markdown content was extracted
            stats.api_failed += 1
            log_event(
                logger,
                "Extract failed",
                event="extract_failed",
                url=article.url,
                title=article.title,
                backend="crawl4ai_api",
                error="Empty extraction result",
                status_code=result.status_code,
                markdown_size=markdown_size,
                extraction_methods=["crawl4ai"],
            )
            await _advance_progress()
            return ExtractedArticle(article=article, text=None, error="Empty extraction result")

        await _advance_progress()
        return ExtractedArticle(article=article, text=None, error=result.error)

    # gather() preserves input order, which keeps downstream article ordering stable.
    tasks = [asyncio.create_task(_fetch_single(article)) for article in articles]
    return await asyncio.gather(*tasks)


def _render_fetch_stats(stats: FetchStats, console: Console) -> None:
    """Display fetch statistics to the console.

    Prints summary statistics for total, success, failed, and cache hits.

    Args:
        stats: Fetch statistics to display
        console: Rich console for output
    """
    console.print(
        "[bold]Fetch summary[/bold]: "
        f"total={stats.total}, success={stats.api_success}, failed={stats.api_failed}, "
        f"cache_hits={stats.cache_hits}"
    )


def _is_placeholder_text(text: str) -> bool:
    """Detect if extracted text is a placeholder or too short.

    JavaScript-disabled messages, Cloudflare challenges, and very short
    content are considered invalid placeholder text.

    Args:
        text: The extracted text to validate

    Returns:
        True if text appears to be placeholder content
    """
    lowered = text.lower()
    if "javascript is disabled" in lowered or "please enable javascript" in lowered:
        return True
    if "enable javascript to continue" in lowered:
        return True
    # Detect Cloudflare challenge pages (more specific patterns)
    if "verifying you are human" in lowered:
        return True
    if "just a moment..." in lowered:
        return True
    if "checking your browser before accessing" in lowered:
        return True
    # Only treat as Cloudflare challenge if it has both "ray id:" AND short content
    # (legitimate pages using Cloudflare protection have longer content)
    if "ray id:" in lowered and len(text.strip()) < 1000:
        return True
    return len(text.strip()) < 200


def _build_provider(cfg: AppConfig, llm_logger):
    """Build a pluggable LLM provider instance based on configuration."""
    return create_provider(cfg.provider, cfg.summary, cfg.logging, llm_logger)


def _build_run_output_dir(output_dir: Path, input_path: Path, cfg: AppConfig) -> Path:
    """Build the output directory name based on configured mode.

    Args:
        output_dir: Base output directory
        input_path: Path to input file (for extracting stem name)
        cfg: Application configuration

    Returns:
        Path to the run output directory

    Raises:
        ValueError: If run_folder_mode is not supported
    """
    stem = input_path.stem or "run"
    mode = (cfg.output.run_folder_mode or "input").lower()
    if mode == "input":
        # Stable folder name preferred by publish automation scripts.
        run_dir_name = stem
    elif mode == "timestamp":
        # Timestamp prefix helps with repeated runs for same input.
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir_name = f"{timestamp}-{stem}"
    elif mode == "input_timestamp":
        # Input-first naming keeps source grouping while still unique.
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir_name = f"{stem}-{timestamp}"
    else:
        raise ValueError(
            "Unsupported run_folder_mode. Use 'input', 'timestamp', or 'input_timestamp'."
        )
    return output_dir / run_dir_name


def _sync_web_report_index(*, html_path: Path, title: str, input_path: Path, logger) -> None:
    """Best-effort sync of generated report into web_dailyreport index."""
    try:
        # Non-fatal helper: pipeline success should not depend on web repo layout.
        index_path = sync_report_to_web_index(html_path=html_path, title=title, input_path=input_path)
    except Exception as exc:  # noqa: BLE001
        log_event(
            logger,
            "Web index sync failed",
            event="web_index_sync_failed",
            error=str(exc),
            output=str(html_path),
        )
        return

    if index_path is None:
        log_event(
            logger,
            "Web index sync skipped",
            event="web_index_sync_skipped",
            output=str(html_path),
        )
        return

    log_event(
        logger,
        "Web index synced",
        event="web_index_synced",
        index=str(index_path),
        output=str(html_path),
    )
