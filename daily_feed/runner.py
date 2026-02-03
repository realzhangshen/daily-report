from __future__ import annotations

from pathlib import Path

from .config import AppConfig, get_api_key
from .dedup import dedup_articles
from .extractor import extract_text
from .fetcher import cache_path, fetch_url
from .parser import parse_folo_markdown
from .providers.gemini import GeminiProvider
from .renderer import render_html, render_markdown
from .types import ArticleSummary, ExtractedArticle


def run_pipeline(input_path: Path, output_dir: Path, cfg: AppConfig) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

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
    html_path = output_dir / "report.html"
    render_html(summaries, html_path, title)

    if cfg.output.include_markdown:
        md_path = output_dir / "report.md"
        render_markdown(summaries, md_path, title)

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
            )
            if result.text:
                html = result.text
                html_cache.write_text(html, encoding="utf-8")
            else:
                extracted.append(ExtractedArticle(article=article, text=None, error=result.error))
                continue

        text = extract_text(html, cfg.extract.primary, cfg.extract.fallback)
        if text:
            text_cache.write_text(text, encoding="utf-8")
        extracted.append(ExtractedArticle(article=article, text=text, error=None))

    return extracted


def _build_provider(cfg: AppConfig):
    if cfg.provider.name == "gemini":
        api_key = get_api_key(cfg.provider)
        return GeminiProvider(cfg.provider, cfg.summary, api_key)
    raise ValueError(f"Unsupported provider: {cfg.provider.name}")
