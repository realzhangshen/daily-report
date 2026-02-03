from __future__ import annotations

from datetime import datetime
from pathlib import Path

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
from .dedup import dedup_articles
from .extractor import extract_text
from .fetcher import cache_path, fetch_url
from .parser import parse_folo_markdown
from .providers.gemini import GeminiProvider
from .renderer import render_html, render_markdown
from .types import ArticleSummary, ExtractedArticle


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    cfg: AppConfig,
    show_progress: bool = True,
    console: Console | None = None,
) -> Path:
    run_output_dir = _build_run_output_dir(output_dir, input_path, cfg)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not show_progress:
        raw = input_path.read_text(encoding="utf-8")
        articles = parse_folo_markdown(raw)

        if cfg.dedup.enabled:
            articles = dedup_articles(articles, cfg.dedup.title_similarity_threshold)

        extracted = _fetch_and_extract(articles, cache_dir, cfg)
        provider = _build_provider(cfg)

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
            summaries.append(provider.summarize_article(item.article, item.text))

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
        extracted: list[ExtractedArticle] = []
        for article in articles:
            html_cache = cache_path(cache_dir, article.url, "html")
            text_cache = cache_path(cache_dir, article.url, "txt")

            if text_cache.exists():
                text = text_cache.read_text(encoding="utf-8")
                extracted.append(ExtractedArticle(article=article, text=text))
                progress.advance(fetch_task, 1)
                continue

            if html_cache.exists():
                html = html_cache.read_text(encoding="utf-8", errors="ignore")
            else:
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
                else:
                    extracted.append(
                        ExtractedArticle(article=article, text=None, error=result.error)
                    )
                    progress.advance(fetch_task, 1)
                    continue

            text = extract_text(html, cfg.extract.primary, cfg.extract.fallback)
            if text and _is_placeholder_text(text):
                text = None
            if text:
                text_cache.write_text(text, encoding="utf-8")
            extracted.append(ExtractedArticle(article=article, text=text, error=None))
            progress.advance(fetch_task, 1)
        progress.advance(stage_task, 1)

        provider = _build_provider(cfg)

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
            summaries.append(provider.summarize_article(item.article, item.text))
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

    return html_path


def _fetch_and_extract(articles, cache_dir: Path, cfg: AppConfig):
    extracted: list[ExtractedArticle] = []
    for article in articles:
        html_cache = cache_path(cache_dir, article.url, "html")
        text_cache = cache_path(cache_dir, article.url, "txt")

        if text_cache.exists():
            text = text_cache.read_text(encoding="utf-8")
            extracted.append(ExtractedArticle(article=article, text=text))
            continue

        if html_cache.exists():
            html = html_cache.read_text(encoding="utf-8", errors="ignore")
        else:
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
            else:
                extracted.append(ExtractedArticle(article=article, text=None, error=result.error))
                continue

        text = extract_text(html, cfg.extract.primary, cfg.extract.fallback)
        if text and _is_placeholder_text(text):
            text = None
        if text:
            text_cache.write_text(text, encoding="utf-8")
        extracted.append(ExtractedArticle(article=article, text=text, error=None))

    return extracted


def _is_placeholder_text(text: str) -> bool:
    lowered = text.lower()
    if "javascript is disabled" in lowered or "please enable javascript" in lowered:
        return True
    if "enable javascript to continue" in lowered:
        return True
    return len(text.strip()) < 200


def _build_provider(cfg: AppConfig):
    if cfg.provider.name == "gemini":
        api_key = get_api_key(cfg.provider)
        return GeminiProvider(cfg.provider, cfg.summary, api_key)
    raise ValueError(f"Unsupported provider: {cfg.provider.name}")


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
