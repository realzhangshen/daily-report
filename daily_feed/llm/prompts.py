"""Prompt loading and rendering helpers for LLM providers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import SummaryConfig
from ..core.types import Article


_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=None)
def _load_template(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()


def _render_template(name: str, **values: str) -> str:
    template = _load_template(name)
    return template.format(**values)


def build_deep_fetch_prompt(
    article: Article,
    text: str,
    candidate_links: list[str],
    cfg: SummaryConfig,
) -> str:
    trimmed = text[: cfg.max_chars]
    links_block = "\n".join(f"- {link}" for link in candidate_links[:50])
    if not links_block:
        links_block = "- (no obvious links extracted)"

    return _render_template(
        "deep_fetch",
        title=article.title,
        site=article.site,
        author=article.author or "",
        candidate_links=links_block,
        content=trimmed,
    )


def build_analysis_prompt(
    article: Article,
    base_text: str,
    deep_texts: list[str],
    decision: dict[str, Any],
    cfg: SummaryConfig,
) -> str:
    base_trimmed = base_text[: cfg.max_chars]
    deep_chunks = []
    for idx, text in enumerate(deep_texts):
        if not text:
            continue
        deep_chunks.append(f"[Deep Source {idx + 1}]\n{text[: cfg.max_chars]}")
    deep_block = "\n\n".join(deep_chunks) or "(none)"
    decision_note = (
        f"Deep fetch decision: {decision.get('need_deep_fetch')}, "
        f"URLs: {decision.get('urls')}, Rationale: {decision.get('rationale')}"
    )

    return _render_template(
        "analysis",
        title=article.title,
        site=article.site,
        author=article.author or "",
        decision_note=decision_note,
        primary_content=base_trimmed,
        additional_sources=deep_block,
    )
