"""Entry analyzer for per-article open-ended analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass
import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from ..config import AppConfig, get_crawl4ai_api_auth, get_crawl4ai_api_url
from ..core.entry import EntryManager
from ..core.types import AnalysisResult, ExtractedArticle
from ..fetch.fetcher import fetch_url_crawl4ai_api
from ..llm.providers.base import AnalysisProvider
from ..utils.logging import log_event


_URL_RE = re.compile(r"https?://[^\s)\]]+")


@dataclass
class DeepFetchDecision:
    """Decision object for deep fetching."""

    need_deep_fetch: bool
    urls: list[str]
    rationale: str = ""


class EntryAnalyzer:
    """Analyze a single entry with optional deep fetch."""

    def __init__(
        self,
        cfg: AppConfig,
        provider: AnalysisProvider,
        articles_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.provider = provider
        self.articles_dir = articles_dir
        self.logger = logger

    def analyze(self, item: ExtractedArticle) -> AnalysisResult:
        article = item.article
        entry = EntryManager(self.articles_dir, article)
        entry.ensure_folder()
        entry_logger = entry.get_llm_logger()
        analysis_logger = entry.get_analysis_logger()

        base_text = item.text or article.summary or ""
        candidate_links = _extract_links(base_text)

        decision_raw: dict[str, Any]
        if self.cfg.summary.enable_deep_fetch_decision and candidate_links:
            decision_raw = self.provider.decide_deep_fetch(
                article, base_text, candidate_links, entry_logger=entry_logger
            )
        elif not self.cfg.summary.enable_deep_fetch_decision:
            decision_raw = {
                "need_deep_fetch": False,
                "urls": [],
                "rationale": "disabled_by_config",
            }
        else:
            decision_raw = {
                "need_deep_fetch": False,
                "urls": [],
                "rationale": "no_candidate_links",
            }
        decision = DeepFetchDecision(
            need_deep_fetch=bool(decision_raw.get("need_deep_fetch")),
            urls=list(decision_raw.get("urls") or []),
            rationale=str(decision_raw.get("rationale") or ""),
        )

        log_event(
            analysis_logger,
            "Deep fetch decision",
            event="deep_fetch_decision",
            url=article.url,
            title=article.title,
            need_deep_fetch=decision.need_deep_fetch,
            urls=decision.urls,
            rationale=decision.rationale,
        )

        deep_texts: list[str] = []
        deep_links: list[str] = []
        if decision.need_deep_fetch:
            urls = _select_deep_links(
                decision.urls, candidate_links, article.url, self.cfg.fetch.deep_fetch_max_links
            )
            if urls:
                deep_results = asyncio.run(self._fetch_deep(urls))
                deep_links = [r["url"] for r in deep_results if r.get("text")]
                deep_texts = [r["text"] for r in deep_results if r.get("text")]
                _write_deep_fetch(entry.folder, deep_results)
                log_event(
                    analysis_logger,
                    "Deep fetch complete",
                    event="deep_fetch_complete",
                    url=article.url,
                    title=article.title,
                    deep_count=len(deep_links),
                )

        analysis_text = self.provider.analyze_entry(
            article, base_text, deep_texts, decision_raw, entry_logger=entry_logger
        )
        result = AnalysisResult(
            article=article,
            analysis=analysis_text,
            status="ok" if analysis_text else "analysis_only",
            meta={
                "model": self.provider.cfg.model,
                "deep_fetch": decision.need_deep_fetch,
                "deep_links": deep_links,
                "deep_fetch_rationale": decision.rationale,
            },
        )
        entry.write_analysis_result(result)
        return result

    async def _fetch_deep(self, urls: list[str]) -> list[dict[str, Any]]:
        api_url = get_crawl4ai_api_url(self.cfg.fetch)
        if not api_url:
            raise ValueError(
                "Crawl4AI API URL is required. Set CRAWL4AI_API_URL environment variable "
                "or configure crawl4ai_api_url in config."
            )

        api_auth = get_crawl4ai_api_auth(self.cfg.fetch)

        async def _fetch_one(url: str) -> dict[str, Any]:
            result = await fetch_url_crawl4ai_api(
                url,
                api_url=api_url,
                timeout=self.cfg.fetch.timeout_seconds,
                retries=self.cfg.fetch.retries,
                user_agent=self.cfg.fetch.user_agent,
                stealth=self.cfg.fetch.crawl4ai_stealth,
                delay=self.cfg.fetch.crawl4ai_delay,
                simulate_user=self.cfg.fetch.crawl4ai_simulate_user,
                magic=self.cfg.fetch.crawl4ai_magic,
                auth=api_auth,
            )
            text = result.text
            if text and _is_placeholder_text(text):
                text = None
            return {
                "url": url,
                "text": text,
                "status_code": result.status_code,
                "error": result.error,
            }

        tasks = [asyncio.create_task(_fetch_one(url)) for url in urls]
        return await asyncio.gather(*tasks)


def _select_deep_links(
    decision_urls: list[str],
    candidate_links: list[str],
    original_url: str,
    max_links: int,
) -> list[str]:
    ordered: list[str] = []
    seen = {original_url}
    for url in decision_urls + candidate_links:
        if url in seen:
            continue
        seen.add(url)
        ordered.append(url)
        if len(ordered) >= max_links:
            break
    return ordered


def _extract_links(text: str) -> list[str]:
    if not text:
        return []
    urls = _URL_RE.findall(text)
    cleaned = []
    seen = set()
    for url in urls:
        cleaned_url = url.rstrip(").,;\"'")
        if cleaned_url in seen:
            continue
        seen.add(cleaned_url)
        cleaned.append(cleaned_url)
    return cleaned


def _write_deep_fetch(folder: Path, results: list[dict[str, Any]]) -> None:
    deep_dir = folder / "deep"
    deep_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []
    for item in results:
        url = item.get("url") or ""
        text = item.get("text")
        filename = f"{_short_hash(url)}.txt"
        if text:
            (deep_dir / filename).write_text(text, encoding="utf-8")
        index.append(
            {
                "url": url,
                "file": f"deep/{filename}" if text else None,
                "status_code": item.get("status_code"),
                "error": item.get("error"),
            }
        )
    with open(folder / "deep_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def _short_hash(value: str) -> str:
    return hashlib.md5(value.encode()).hexdigest()[:8]


def _is_placeholder_text(text: str) -> bool:
    lowered = text.lower()
    if "javascript is disabled" in lowered or "please enable javascript" in lowered:
        return True
    if "enable javascript to continue" in lowered:
        return True
    if "verifying you are human" in lowered:
        return True
    if "just a moment..." in lowered:
        return True
    if "checking your browser before accessing" in lowered:
        return True
    if "ray id:" in lowered and len(text.strip()) < 1000:
        return True
    return len(text.strip()) < 200
