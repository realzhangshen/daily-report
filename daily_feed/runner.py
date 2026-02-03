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
from .providers.gemini import GeminiProvider
from .renderer import render_html, render_markdown
from .types import ArticleSummary, ExtractedArticle


@dataclass
class FetchStats:
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
    run_output_dir = _build_run_output_dir(output_dir, input_path, cfg)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _build_cache_dir(run_output_dir, output_dir, cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_index = CacheIndex(
        cache_dir, enabled=cfg.cache.write_index, filename=cfg.cache.index_filename
    )
    logger = setup_logging(cfg.logging, run_output_dir)
    llm_logger = setup_llm_logger(cfg.logging, run_output_dir)
    log_event(
        logger,
        "Pipeline start",
        event="pipeline_start",
        input=str(input_path),
        output=str(run_output_dir),
        cache_dir=str(cache_dir),
    )

    if not show_progress:
        raw = input_path.read_text(encoding="utf-8")
        articles = parse_folo_markdown(raw)

        if cfg.dedup.enabled:
            articles = dedup_articles(articles, cfg.dedup.title_similarity_threshold)

        extracted, stats = _fetch_articles(articles, cache_dir, cfg, cache_index, logger)
        _render_fetch_stats(stats, console or Console())
        provider = _build_provider(cfg, llm_logger)

        summaries: list[ArticleSummary] = []
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
        extracted, stats = _fetch_articles(
            articles, cache_dir, cfg, cache_index, logger, progress, fetch_task
        )
        _render_fetch_stats(stats, console)
        progress.advance(stage_task, 1)

        provider = _build_provider(cfg, llm_logger)

        summarize_task = progress.add_task("Summarize", total=len(extracted))
        summaries: list[ArticleSummary] = []
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

    return html_path


def _fetch_and_extract(articles, cache_dir: Path, cfg: AppConfig):
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
    html_cache = cache_path(cache_dir, article.url, "html")
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
            stats.httpx_failed += 1
            log_event(
                logger,
                "Fetch failed",
                event="fetch_failed",
                url=article.url,
                title=article.title,
                backend="httpx",
                error=result.error,
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

        stats.crawl4ai_failed += 1
        log_event(
            logger,
            "Fetch failed",
            event="fetch_failed",
            url=article.url,
            title=article.title,
            backend="crawl4ai",
            error=result.error,
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
    lowered = text.lower()
    if "javascript is disabled" in lowered or "please enable javascript" in lowered:
        return True
    if "enable javascript to continue" in lowered:
        return True
    return len(text.strip()) < 200


def _build_provider(cfg: AppConfig, llm_logger):
    if cfg.provider.name == "gemini":
        api_key = get_api_key(cfg.provider)
        return GeminiProvider(cfg.provider, cfg.summary, api_key, cfg.logging, llm_logger)
    raise ValueError(f"Unsupported provider: {cfg.provider.name}")


def _build_cache_dir(run_output_dir: Path, output_dir: Path, cfg: AppConfig) -> Path:
    mode = (cfg.cache.mode or "run").lower()
    if mode == "shared":
        if cfg.cache.shared_dir:
            return Path(cfg.cache.shared_dir)
        return output_dir / "cache_shared"
    return run_output_dir / "cache"


def _is_cache_valid(path: Path, cfg: AppConfig) -> bool:
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
