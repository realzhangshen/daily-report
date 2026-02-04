"""
Main pipeline orchestration for the Daily Feed Agent.

This module coordinates the entire workflow:
1. Parse markdown input
2. Deduplicate articles
3. Fetch and extract content
4. Generate summaries via LLM
5. Group by topic
6. Render output files

Supports both progress bar and quiet modes, with configurable
fetching backends (httpx or crawl4ai).
"""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import asyncio

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .config import AppConfig, get_api_key
from .cache import CacheIndex
from .dedup import dedup_articles
from .extractor import extract_text
from .fetcher import cache_path, fetch_url, fetch_url_crawl4ai
from .logging_utils import log_event, setup_llm_logger, setup_logging
from .parser import parse_folo_markdown
from .langfuse_utils import set_span_output, setup_langfuse, start_span
from .providers.gemini import GeminiProvider
from .renderer import render_html, render_markdown
from .types import ArticleSummary, ExtractedArticle


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

    Tracks success/failure rates for different backends and
    cache hit rate for performance monitoring.

    Attributes:
        total: Total number of articles to fetch
        cache_hits: Number served from cache
        crawl4ai_success: Successful crawl4ai fetches
        crawl4ai_failed: Failed crawl4ai fetches
        crawl4ai_fallback_used: Times httpx fallback was used
        httpx_success: Successful httpx fetches
        httpx_failed: Failed httpx fetches
    """
    total: int = 0
    cache_hits: int = 0
    crawl4ai_success: int = 0
    crawl4ai_failed: int = 0
    crawl4ai_fallback_used: int = 0
    httpx_success: int = 0
    httpx_failed: int = 0


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
    run_output_dir = _build_run_output_dir(output_dir, input_path, cfg)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _build_cache_dir(run_output_dir, output_dir, cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_index = CacheIndex(
        cache_dir, enabled=cfg.cache.write_index, filename=cfg.cache.index_filename
    )
    logger = setup_logging(cfg.logging, run_output_dir)
    llm_logger = setup_llm_logger(cfg.logging, run_output_dir)
    setup_langfuse(cfg.langfuse)

    with start_span(
        "daily_feed.run",
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
            cache_dir=str(cache_dir),
        )

        # Process without progress bar (quiet mode)
        if not show_progress:
            raw = input_path.read_text(encoding="utf-8")
            articles = parse_folo_markdown(raw)

            if cfg.dedup.enabled:
                articles = dedup_articles(articles, cfg.dedup.title_similarity_threshold)

            with start_span(
                "daily_feed.fetch_extract",
                kind="chain",
                input_value={"count": len(articles)},
            ):
                extracted, stats = _fetch_articles(
                    articles, cache_dir, cfg, cache_index, logger
                )
            _render_fetch_stats(stats, console or Console())
            provider = _build_provider(cfg, llm_logger)

            summaries: list[ArticleSummary] = []
            with start_span(
                "daily_feed.summarize_batch",
                kind="chain",
                input_value={"count": len(extracted)},
            ):
                for item in extracted:
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
                        continue
                    summary = provider.summarize_article(item.article, item.text)
                    if summary.status == "parse_error":
                        log_event(
                            logger,
                            "LLM parse error",
                            event="llm_parse_error",
                            url=item.article.url,
                            title=item.article.title,
                        )
                    summaries.append(summary)

            with start_span(
                "daily_feed.group_topics",
                kind="chain",
                input_value={"count": len(summaries)},
            ):
                grouped = provider.group_topics(summaries)
            if not grouped:
                for summary in summaries:
                    summary.topic = None
            else:
                used_titles = {s.article.title for group in grouped.values() for s in group}
                for summary in summaries:
                    if summary.article.title not in used_titles:
                        summary.topic = None

            title = f"Daily Feed Report - {input_path.stem}"
            html_path = run_output_dir / "report.html"
            render_html(summaries, html_path, title)

            if cfg.output.include_markdown:
                md_path = run_output_dir / "report.md"
                render_markdown(summaries, md_path, title)

            log_event(
                logger,
                "Pipeline complete",
                event="pipeline_complete",
                output=str(html_path),
                total=len(summaries),
            )
            set_span_output(
                run_span, {"report": str(html_path), "total": len(summaries)}
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
            stage_task = progress.add_task("Stages", total=5)

            raw = input_path.read_text(encoding="utf-8")
            articles = parse_folo_markdown(raw)
            progress.advance(stage_task, 1)

            if cfg.dedup.enabled:
                articles = dedup_articles(articles, cfg.dedup.title_similarity_threshold)
            progress.advance(stage_task, 1)

            fetch_task = progress.add_task("Fetch + Extract", total=len(articles))
            with start_span(
                "daily_feed.fetch_extract",
                kind="chain",
                input_value={"count": len(articles)},
            ):
                extracted, stats = _fetch_articles(
                    articles, cache_dir, cfg, cache_index, logger, progress, fetch_task
                )
            _render_fetch_stats(stats, console)
            progress.advance(stage_task, 1)

            provider = _build_provider(cfg, llm_logger)

            summarize_task = progress.add_task("Summarize", total=len(extracted))
            summaries: list[ArticleSummary] = []
            with start_span(
                "daily_feed.summarize_batch",
                kind="chain",
                input_value={"count": len(extracted)},
            ):
                for item in extracted:
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
                    summary = provider.summarize_article(item.article, item.text)
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
            progress.advance(stage_task, 1)

            with start_span(
                "daily_feed.group_topics",
                kind="chain",
                input_value={"count": len(summaries)},
            ):
                grouped = provider.group_topics(summaries)
            if not grouped:
                for summary in summaries:
                    summary.topic = None
            else:
                used_titles = {s.article.title for group in grouped.values() for s in group}
                for summary in summaries:
                    if summary.article.title not in used_titles:
                        summary.topic = None

            title = f"Daily Feed Report - {input_path.stem}"
            html_path = run_output_dir / "report.html"
            render_html(summaries, html_path, title)

            if cfg.output.include_markdown:
                md_path = run_output_dir / "report.md"
                render_markdown(summaries, md_path, title)

            progress.advance(stage_task, 1)
            log_event(
                logger,
                "Pipeline complete",
                event="pipeline_complete",
                output=str(html_path),
                total=len(summaries),
            )
            set_span_output(
                run_span, {"report": str(html_path), "total": len(summaries)}
            )

        return html_path


def _fetch_and_extract(articles, cache_dir: Path, cfg: AppConfig):
    """Convenience function to fetch and extract without progress tracking.

    Used by tests or other code that doesn't need the full pipeline.

    Args:
        articles: List of Article objects to fetch
        cache_dir: Cache directory for storing fetched content
        cfg: Application configuration

    Returns:
        List of ExtractedArticle objects
    """
    extracted, _stats = _fetch_articles(articles, cache_dir, cfg, None, None)
    return extracted


def _fetch_articles(
    articles,
    cache_dir: Path,
    cfg: AppConfig,
    cache_index: CacheIndex | None,
    logger,
    progress: Progress | None = None,
    fetch_task: int | None = None,
) -> tuple[list[ExtractedArticle], FetchStats]:
    """Fetch and extract content for all articles.

    Dispatches to the appropriate backend (httpx or crawl4ai) based
    on configuration. Returns extracted articles and statistics.

    Args:
        articles: List of Article objects to fetch
        cache_dir: Cache directory for storing fetched content
        cfg: Application configuration
        cache_index: Optional cache index for logging
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
            articles, cache_dir, cfg, stats, cache_index, logger, progress, fetch_task
        )
        return extracted, stats
    extracted = _fetch_and_extract_httpx(
        articles, cache_dir, cfg, stats, cache_index, logger, progress, fetch_task
    )
    return extracted, stats


def _fetch_and_extract_httpx(
    articles,
    cache_dir: Path,
    cfg: AppConfig,
    stats: FetchStats,
    cache_index: CacheIndex | None,
    logger,
    progress: Progress | None = None,
    fetch_task: int | None = None,
) -> list[ExtractedArticle]:
    """Fetch and extract using httpx backend (sequential processing).

    Processes each article sequentially, checking cache first, then
    fetching HTML and extracting text content.

    Args:
        articles: List of Article objects to fetch
        cache_dir: Cache directory for storing fetched content
        cfg: Application configuration
        stats: Statistics object to update
        cache_index: Optional cache index for logging
        logger: Logger for events
        progress: Optional Rich progress bar
        fetch_task: Task ID for progress updates

    Returns:
        List of ExtractedArticle objects
    """
    extracted: list[ExtractedArticle] = []
    for article in articles:
        extracted.append(_fetch_single_httpx(article, cache_dir, cfg, stats, cache_index, logger))
        if progress and fetch_task is not None:
            progress.advance(fetch_task, 1)
    return extracted


def _fetch_single_httpx(
    article,
    cache_dir: Path,
    cfg: AppConfig,
    stats: FetchStats,
    cache_index: CacheIndex | None,
    logger,
) -> ExtractedArticle:
    """Fetch a single article using httpx and extract content.

    Checks for cached text first, then cached HTML, then fetches
    fresh content if needed. Extracts text from HTML using the
    configured extraction chain.

    Args:
        article: Article to fetch
        cache_dir: Cache directory
        cfg: Application configuration
        stats: Statistics object to update
        cache_index: Optional cache index for logging
        logger: Logger for events

    Returns:
        ExtractedArticle with text or error populated
    """
    html_cache = cache_path(cache_dir, article.url, "html")
    text_cache = cache_path(cache_dir, article.url, "txt")

    # Check for cached extracted text (fastest path)
    if _is_cache_valid(text_cache, cfg):
        text = text_cache.read_text(encoding="utf-8")
        stats.cache_hits += 1
        _append_cache_index(
            cache_index,
            url=article.url,
            path=text_cache,
            kind="txt",
            source="cache",
            status_code=None,
            error=None,
        )
        return ExtractedArticle(article=article, text=text)

    if _is_cache_valid(html_cache, cfg):
        html = html_cache.read_text(encoding="utf-8", errors="ignore")
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
            html_cache.write_text(html, encoding="utf-8")
            _append_cache_index(
                cache_index,
                url=article.url,
                path=html_cache,
                kind="html",
                source="httpx",
                status_code=result.status_code,
                error=None,
            )
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
        text_cache.write_text(text, encoding="utf-8")
        _append_cache_index(
            cache_index,
            url=article.url,
            path=text_cache,
            kind="txt",
            source="extract",
            status_code=None,
            error=None,
        )
        stats.httpx_success += 1
        log_event(
            logger,
            "Extract success",
            event="extract_success",
            url=article.url,
            title=article.title,
        )
        return ExtractedArticle(article=article, text=text, error=None)
    stats.httpx_failed += 1
    log_event(
        logger,
        "Extract failed",
        event="extract_failed",
        url=article.url,
        title=article.title,
        error="Empty extraction result",
    )
    return ExtractedArticle(article=article, text=None, error="Empty extraction result")


def _fetch_and_extract_crawl4ai(
    articles,
    cache_dir: Path,
    cfg: AppConfig,
    stats: FetchStats,
    cache_index: CacheIndex | None,
    logger,
    progress: Progress | None = None,
    fetch_task: int | None = None,
) -> list[ExtractedArticle]:
    """Fetch and extract using crawl4ai backend (async concurrent processing).

    Uses asyncio to process multiple articles concurrently with
    semaphore-controlled concurrency limits. Falls back to httpx
    if configured and crawl4ai fails.

    Args:
        articles: List of Article objects to fetch
        cache_dir: Cache directory for storing fetched content
        cfg: Application configuration
        stats: Statistics object to update
        cache_index: Optional cache index for logging
        logger: Logger for events
        progress: Optional Rich progress bar
        fetch_task: Task ID for progress updates

    Returns:
        List of ExtractedArticle objects
    """
    return asyncio.run(
        _fetch_and_extract_crawl4ai_async(
            articles, cache_dir, cfg, stats, cache_index, logger, progress, fetch_task
        )
    )


async def _fetch_and_extract_crawl4ai_async(
    articles,
    cache_dir: Path,
    cfg: AppConfig,
    stats: FetchStats,
    cache_index: CacheIndex | None,
    logger,
    progress: Progress | None = None,
    fetch_task: int | None = None,
) -> list[ExtractedArticle]:
    """Async implementation of crawl4ai fetching with concurrent processing.

    Creates tasks for all articles and runs them with semaphore-controlled
    concurrency. Handles cache checking and fallback to httpx.

    Args:
        articles: List of Article objects to fetch
        cache_dir: Cache directory for storing fetched content
        cfg: Application configuration
        stats: Statistics object to update
        cache_index: Optional cache index for logging
        logger: Logger for events
        progress: Optional Rich progress bar
        fetch_task: Task ID for progress updates

    Returns:
        List of ExtractedArticle objects
    """
    # Semaphore limits concurrent requests to avoid overwhelming the server
    semaphore = asyncio.Semaphore(max(1, int(cfg.fetch.crawl4ai_concurrency or 1)))
    progress_lock = asyncio.Lock()

    async def _advance_progress() -> None:
        if progress and fetch_task is not None:
            async with progress_lock:
                progress.advance(fetch_task, 1)

    async def _fetch_single(article) -> ExtractedArticle:
        text_cache = cache_path(cache_dir, article.url, "txt")
        if _is_cache_valid(text_cache, cfg):
            text = text_cache.read_text(encoding="utf-8")
            stats.cache_hits += 1
            _append_cache_index(
                cache_index,
                url=article.url,
                path=text_cache,
                kind="txt",
                source="cache",
                status_code=None,
                error=None,
            )
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
            _append_cache_index(
                cache_index,
                url=article.url,
                path=text_cache,
                kind="txt",
                source="crawl4ai",
                status_code=result.status_code,
                error=None,
            )
            stats.crawl4ai_success += 1
            await _advance_progress()
            return ExtractedArticle(article=article, text=text, error=None)

        error_category = _categorize_error(result.error, result.status_code)
        stats.crawl4ai_failed += 1
        log_event(
            logger,
            "Fetch failed",
            event="fetch_failed",
            url=article.url,
            title=article.title,
            backend="crawl4ai",
            error=result.error,
            status_code=result.status_code,
            error_category=error_category,
        )
        if cfg.fetch.fallback_to_httpx:
            stats.crawl4ai_fallback_used += 1
            extracted = await asyncio.to_thread(
                _fetch_single_httpx, article, cache_dir, cfg, stats, cache_index, logger
            )
            await _advance_progress()
            return extracted

        await _advance_progress()
        return ExtractedArticle(article=article, text=None, error=result.error)

    tasks = [asyncio.create_task(_fetch_single(article)) for article in articles]
    return await asyncio.gather(*tasks)


def _render_fetch_stats(stats: FetchStats, console: Console) -> None:
    """Display fetch statistics to the console.

    Prints summary statistics for total, success, failed, and cache hits,
    broken down by backend (crawl4ai, httpx).

    Args:
        stats: Fetch statistics to display
        console: Rich console for output
    """
    failed = stats.crawl4ai_failed + stats.httpx_failed
    success = stats.crawl4ai_success + stats.httpx_success
    console.print(
        "[bold]Fetch summary[/bold]: "
        f"total={stats.total}, success={success}, failed={failed}, cache_hits={stats.cache_hits}"
    )
    if stats.crawl4ai_success or stats.crawl4ai_failed:
        console.print(
            "[bold]Crawl4AI[/bold]: "
            f"success={stats.crawl4ai_success}, failed={stats.crawl4ai_failed}, "
            f"fallback_used={stats.crawl4ai_fallback_used}"
        )
    if stats.httpx_success or stats.httpx_failed:
        console.print(
            f"[bold]httpx[/bold]: success={stats.httpx_success}, failed={stats.httpx_failed}"
        )


def _is_placeholder_text(text: str) -> bool:
    """Detect if extracted text is a placeholder or too short.

    JavaScript-disabled messages and very short content are
    considered invalid placeholder text.

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
    return len(text.strip()) < 200


def _build_provider(cfg: AppConfig, llm_logger):
    """Build the LLM provider instance based on configuration.

    Args:
        cfg: Application configuration
        llm_logger: Logger for LLM interactions

    Returns:
        Provider instance (e.g., GeminiProvider)

    Raises:
        ValueError: If provider name is not supported
    """
    if cfg.provider.name == "gemini":
        api_key = get_api_key(cfg.provider)
        return GeminiProvider(cfg.provider, cfg.summary, api_key, cfg.logging, llm_logger)
    raise ValueError(f"Unsupported provider: {cfg.provider.name}")


def _build_cache_dir(run_output_dir: Path, output_dir: Path, cfg: AppConfig) -> Path:
    """Determine the cache directory based on cache mode.

    Args:
        run_output_dir: The current run's output directory
        output_dir: Base output directory
        cfg: Application configuration

    Returns:
        Path to the cache directory
    """
    mode = (cfg.cache.mode or "run").lower()
    if mode == "shared":
        if cfg.cache.shared_dir:
            return Path(cfg.cache.shared_dir)
        return output_dir / "cache_shared"
    return run_output_dir / "cache"


def _is_cache_valid(path: Path, cfg: AppConfig) -> bool:
    """Check if a cache file is still valid based on TTL.

    Args:
        path: Path to the cache file
        cfg: Application configuration with TTL settings

    Returns:
        True if cache exists and is within TTL (or TTL is not set)
    """
    if not path.exists():
        return False
    if cfg.cache.ttl_days is None:
        return True
    age_seconds = max(0, (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds())
    return age_seconds <= cfg.cache.ttl_days * 86400


def _append_cache_index(
    cache_index: CacheIndex | None,
    url: str,
    path: Path,
    kind: str,
    source: str,
    status_code: int | None,
    error: str | None,
) -> None:
    """Append an entry to the cache index if enabled.

    Args:
        cache_index: CacheIndex instance (may be None)
        url: The URL being cached
        path: Path to the cache file
        kind: Type of content ("html" or "txt")
        source: Source of content ("cache", "httpx", "crawl4ai", "extract")
        status_code: HTTP status code if applicable
        error: Error message if applicable
    """
    if cache_index is None:
        return
    cache_index.append(
        {
            "url": url,
            "hash": path.name.split(".")[0],
            "kind": kind,
            "path": str(path),
            "source": source,
            "status_code": status_code,
            "error": error,
            "content_len": path.stat().st_size if path.exists() else None,
        }
    )


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
        run_dir_name = stem
    elif mode == "timestamp":
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir_name = f"{timestamp}-{stem}"
    elif mode == "input_timestamp":
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir_name = f"{stem}-{timestamp}"
    else:
        raise ValueError(
            "Unsupported run_folder_mode. Use 'input', 'timestamp', or 'input_timestamp'."
        )
    return output_dir / run_dir_name
