"""Entry analyzer for per-article extraction workflow."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import AppConfig
from ..core.entry import EntryManager
from ..core.types import ExtractionResult, ExtractedArticle
from ..llm.providers.base import AnalysisProvider


class EntryAnalyzer:
    """Extract structured metadata from a single entry."""

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

    def extract(self, item: ExtractedArticle) -> ExtractionResult:
        article = item.article
        entry = EntryManager(self.articles_dir, article)
        entry.ensure_folder()
        entry_logger = entry.get_llm_logger()

        base_text = item.text or article.summary or ""

        raw = self.provider.extract_entry(
            article, base_text, entry_logger=entry_logger
        )

        result = ExtractionResult(
            article=article,
            one_line_summary=str(raw.get("one_line_summary", article.title)),
            category=str(raw.get("category", "other")),
            tags=list(raw.get("tags") or []),
            importance=int(raw.get("importance", 3)),
            content_type=str(raw.get("content_type", "news")),
            key_takeaway=str(raw.get("key_takeaway", "")),
            status="ok",
            meta={"model": self.provider.cfg.model},
        )

        entry.write_extraction_result(result)
        return result
