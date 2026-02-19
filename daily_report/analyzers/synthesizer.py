"""Synthesizer for producing daily briefings from extraction results."""

from __future__ import annotations

import logging
import re
from typing import Any

from ..config import AppConfig
from ..core.types import BriefingResult, BriefingSection, ExtractionResult
from ..llm.providers.base import AnalysisProvider


class Synthesizer:
    """Produce a daily briefing from a batch of extraction results."""

    def __init__(
        self,
        cfg: AppConfig,
        provider: AnalysisProvider,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.provider = provider
        self.logger = logger

    def synthesize(self, extractions: list[ExtractionResult]) -> BriefingResult:
        # Reduce rich extraction objects into a prompt-friendly payload.
        # Keep key fields explicit so prompt templates remain stable.
        payloads = []
        for ext in extractions:
            payloads.append({
                "title": ext.article.title,
                "url": ext.article.url,
                "site": ext.article.site,
                "author": ext.article.author or "",
                "one_line_summary": ext.one_line_summary,
                "category": ext.category,
                "tags": ext.tags,
                "importance": ext.importance,
                "content_type": ext.content_type,
                "key_takeaway": ext.key_takeaway,
            })

        raw_text = self.provider.synthesize(payloads, logger=self.logger)
        return parse_briefing_markdown(raw_text)


def parse_briefing_markdown(text: str) -> BriefingResult:
    """Parse LLM markdown into BriefingResult sections.

    Expected high-level markdown shape:
    - ## 今日概要
    - ## 重点报道
    - ## 主题板块
    - ## 速览
    """
    result = BriefingResult(raw_text=text)

    sections = re.split(r'^## ', text, flags=re.MULTILINE)

    for section in sections:
        if not section.strip():
            continue

        lines = section.strip().split('\n', 1)
        heading = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        # Use keyword matching instead of exact heading equality to remain
        # tolerant of minor prompt/model heading variations.
        if '\u4eca\u65e5\u6982\u8981' in heading:
            result.executive_summary = body
        elif '\u91cd\u70b9\u62a5\u9053' in heading:
            result.top_stories = _parse_top_stories(body)
        elif '\u4e3b\u9898\u677f\u5757' in heading:
            result.sections = _parse_themed_sections(body)
        elif '\u901f\u89c8' in heading:
            result.quick_mentions = _parse_quick_mentions(body)

    return result


def _parse_top_stories(body: str) -> list[dict[str, Any]]:
    """Parse '### <title>' blocks into top story list."""
    stories = []
    parts = re.split(r'^### ', body, flags=re.MULTILINE)
    for part in parts:
        if not part.strip():
            continue
        lines = part.strip().split('\n', 1)
        title = lines[0].strip()
        analysis = lines[1].strip() if len(lines) > 1 else ""
        stories.append({"title": title, "analysis": analysis})
    return stories


def _parse_themed_sections(body: str) -> list[BriefingSection]:
    """Parse themed section blocks from markdown."""
    sections = []
    parts = re.split(r'^### ', body, flags=re.MULTILINE)
    for part in parts:
        if not part.strip():
            continue
        lines = part.strip().split('\n', 1)
        theme = lines[0].strip()
        description = lines[1].strip() if len(lines) > 1 else ""
        sections.append(BriefingSection(theme=theme, description=description))
    return sections


def _parse_quick_mentions(body: str) -> list[dict[str, Any]]:
    """Parse bullet quick mentions.

    Preferred line format:
    - **标题**: 描述
    """
    mentions = []
    for line in body.strip().split('\n'):
        line = line.strip()
        if line.startswith('- '):
            line = line[2:]
        elif line.startswith('* '):
            line = line[2:]
        else:
            continue
        m = re.match(r'\*\*(.+?)\*\*[:\uff1a]\s*(.*)', line)
        if m:
            mentions.append({"title": m.group(1).strip(), "description": m.group(2).strip()})
        else:
            mentions.append({"title": line, "description": ""})
    return mentions
