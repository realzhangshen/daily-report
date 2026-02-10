# Two-Pass Briefing Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-pass per-article analysis with a two-pass pipeline (structured extraction + synthesis) that produces a Chinese-language daily briefing with executive summary, top stories, themed sections, and quick mentions.

**Architecture:** Pass 1 extracts structured JSON metadata from each article (concurrently, cached per-article). Pass 2 feeds all extractions into a single LLM call that produces a complete Chinese daily briefing in markdown format. The briefing markdown is parsed by section headings and rendered into a new HTML template.

**Tech Stack:** Python 3.10+, httpx, Jinja2, existing Gemini/OpenAI provider infrastructure.

---

### Task 1: Add new data types

**Files:**
- Modify: `daily_feed/core/types.py`

**Step 1: Add ExtractionResult and BriefingSection and BriefingResult dataclasses**

Add after the existing `AnalysisResult` class:

```python
@dataclass
class ExtractionResult:
    """Structured metadata extracted from an article by Pass 1."""
    article: Article
    one_line_summary: str = ""
    category: str = ""
    tags: list[str] = field(default_factory=list)
    importance: int = 3
    content_type: str = ""  # product_launch, technical, news, opinion, announcement, tutorial
    key_takeaway: str = ""
    status: str = "ok"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BriefingSection:
    """A themed section in the daily briefing."""
    theme: str
    description: str
    items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BriefingResult:
    """Complete daily briefing produced by Pass 2 synthesis."""
    executive_summary: str = ""
    top_stories: list[dict[str, Any]] = field(default_factory=list)
    sections: list[BriefingSection] = field(default_factory=list)
    quick_mentions: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
```

**Step 2: Commit**

```bash
git add daily_feed/core/types.py
git commit -m "feat: add ExtractionResult and BriefingResult data types"
```

---

### Task 2: Update config with new fields

**Files:**
- Modify: `daily_feed/config.py`
- Modify: `config.yaml`

**Step 1: Update SummaryConfig**

In `daily_feed/config.py`, replace the `SummaryConfig` class with:

```python
@dataclass
class SummaryConfig:
    """Configuration for LLM summarization.

    Attributes:
        bullets_min: Minimum number of bullet points (legacy, unused)
        bullets_max: Maximum number of bullet points (legacy, unused)
        max_chars: Maximum characters of article text to send to LLM
        extraction_max_output_tokens: Max output tokens for per-entry extraction (Pass 1)
        synthesis_max_output_tokens: Max output tokens for synthesis call (Pass 2)
        analysis_concurrency: Number of parallel workers for per-entry extraction
        enable_deep_fetch_decision: Whether to run LLM deep-fetch decision step
        analysis_max_output_tokens: Legacy field, unused
    """

    bullets_min: int = 3
    bullets_max: int = 6
    max_chars: int = 10000
    extraction_max_output_tokens: int = 300
    synthesis_max_output_tokens: int = 4096
    analysis_concurrency: int = 4
    enable_deep_fetch_decision: bool = False
    analysis_max_output_tokens: int = 1200  # legacy compat
```

Wait - we can't have two fields with the same name. Instead, keep `analysis_max_output_tokens` and add the new fields:

```python
@dataclass
class SummaryConfig:
    bullets_min: int = 3
    bullets_max: int = 6
    max_chars: int = 10000
    analysis_max_output_tokens: int = 1200
    extraction_max_output_tokens: int = 300
    synthesis_max_output_tokens: int = 4096
    analysis_concurrency: int = 4
    enable_deep_fetch_decision: bool = False
```

**Step 2: Update `_asdict` and `_fromdict`**

In `_asdict`, add the new fields to the `"summary"` dict:
```python
"extraction_max_output_tokens": cfg.summary.extraction_max_output_tokens,
"synthesis_max_output_tokens": cfg.summary.synthesis_max_output_tokens,
```

In `_fromdict`, add backward compat to strip unknown keys from old configs. The existing `SummaryConfig(**data["summary"])` will handle new defaults automatically since dataclass fields have defaults.

**Step 3: Update config.yaml**

Replace the summary section:
```yaml
summary:
  max_chars: 10000
  extraction_max_output_tokens: 300
  synthesis_max_output_tokens: 4096
  analysis_concurrency: 4
  enable_deep_fetch_decision: false
```

**Step 4: Commit**

```bash
git add daily_feed/config.py config.yaml
git commit -m "feat: add extraction and synthesis config fields"
```

---

### Task 3: Create extraction prompt template

**Files:**
- Create: `daily_feed/prompts/extraction.md`

**Step 1: Write the extraction prompt**

```markdown
You are extracting structured metadata from an article for a daily tech briefing.
Return strict JSON only. No markdown fences, no explanation.

JSON schema:
{
  "one_line_summary": "One sentence summarizing what this article is about",
  "category": "One of: ai, dev-tools, product, startup, open-source, security, science, business, design, other",
  "tags": ["3-5 short lowercase tags"],
  "importance": 3,
  "content_type": "One of: product_launch, technical, news, opinion, announcement, tutorial",
  "key_takeaway": "The single most important or novel point from this article"
}

Importance scale (1-5):
1 = Routine/minor update, low general interest
2 = Somewhat interesting but not urgent
3 = Noteworthy, worth a brief mention
4 = Important development, most readers would care
5 = Major news, industry-shifting or highly impactful

Title: {title}
Site: {site}
Author: {author}

Content:
{content}
```

**Step 2: Commit**

```bash
git add daily_feed/prompts/extraction.md
git commit -m "feat: add extraction prompt template"
```

---

### Task 4: Create synthesis prompt template

**Files:**
- Create: `daily_feed/prompts/synthesis.md`

**Step 1: Write the synthesis prompt**

```markdown
你是一位专业的科技日报编辑。根据以下文章的结构化摘要，撰写一份中文日报简报。

要求：
1. 用中文撰写全部内容
2. 按主题聚类文章，而非按来源分组
3. 相关文章之间要有连接叙述
4. 重要性高的文章详细分析，低的简要提及
5. 语言简洁有力，避免空话套话

输出格式（严格遵守以下 Markdown 结构）：

## 今日概要
3-5句话概括今天最重要的主题和趋势。

## 重点报道
针对 importance >= 4 的文章，每篇写2-3句分析：
### [文章标题]
分析内容...

## 主题板块
将剩余文章按主题聚类：
### [主题名称]
连接叙述，提及该主题下的多篇文章...

## 速览
importance <= 2 的文章，每篇一行：
- **[标题]**: 一句话描述

---

以下是今天的 {count} 篇文章摘要：

{extractions_json}
```

**Step 2: Commit**

```bash
git add daily_feed/prompts/synthesis.md
git commit -m "feat: add Chinese synthesis prompt template"
```

---

### Task 5: Update prompt builders

**Files:**
- Modify: `daily_feed/llm/prompts.py`

**Step 1: Add build_extraction_prompt and build_synthesis_prompt functions**

Add these two functions after the existing `build_analysis_prompt`:

```python
def build_extraction_prompt(
    article: Article,
    base_text: str,
    cfg: SummaryConfig,
) -> str:
    trimmed = base_text[: cfg.max_chars]
    return _render_template(
        "extraction",
        title=article.title,
        site=article.site,
        author=article.author or "",
        content=trimmed,
    )


def build_synthesis_prompt(
    extractions: list[dict[str, Any]],
    cfg: SummaryConfig,
) -> str:
    import json
    extractions_json = json.dumps(extractions, ensure_ascii=False, indent=2)
    return _render_template(
        "synthesis",
        count=str(len(extractions)),
        extractions_json=extractions_json,
    )
```

Also add `from typing import Any` to the imports.

**Step 2: Commit**

```bash
git add daily_feed/llm/prompts.py
git commit -m "feat: add extraction and synthesis prompt builders"
```

---

### Task 6: Update provider interface

**Files:**
- Modify: `daily_feed/llm/providers/base.py`

**Step 1: Add extract_entry and synthesize abstract methods**

Add to the `AnalysisProvider` class:

```python
@abstractmethod
def extract_entry(
    self,
    article: Article,
    base_text: str,
    entry_logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Return structured extraction JSON dict."""
    raise NotImplementedError

@abstractmethod
def synthesize(
    self,
    extractions: list[dict[str, Any]],
    logger: logging.Logger | None = None,
) -> str:
    """Return synthesis markdown text."""
    raise NotImplementedError
```

**Step 2: Commit**

```bash
git add daily_feed/llm/providers/base.py
git commit -m "feat: add extract_entry and synthesize to provider ABC"
```

---

### Task 7: Implement extract_entry and synthesize in Gemini provider

**Files:**
- Modify: `daily_feed/llm/providers/gemini.py`

**Step 1: Add imports**

Add to the imports at top:
```python
from ..prompts import build_analysis_prompt, build_deep_fetch_prompt, build_extraction_prompt, build_synthesis_prompt
```

**Step 2: Add extract_entry method**

Add after `analyze_entry`:

```python
def extract_entry(
    self,
    article: Article,
    base_text: str,
    entry_logger: logging.Logger | None = None,
) -> dict[str, Any]:
    prompt = build_extraction_prompt(article, base_text, self.summary_cfg)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": self.summary_cfg.extraction_max_output_tokens,
        },
    }
    with start_span(
        "gemini.extract_entry",
        kind="llm",
        input_value=prompt,
        attributes={
            "llm.model": self.cfg.model,
            "llm.provider": "gemini",
            "article.title": article.title,
            "article.site": article.site,
        },
    ) as span:
        try:
            data = self._post(payload)
            content = _extract_text(data)
            set_span_output(span, content)
            obj = _parse_json_response(content)
            self._log_llm_response(
                article=article,
                event="llm_extraction_response",
                status="ok",
                content=content,
                prompt=prompt,
                logger=entry_logger,
            )
            return obj
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            record_span_error(span, exc)
            self._log_llm_response(
                article=article,
                event="llm_extraction_response",
                status="error",
                content=str(exc),
                prompt=prompt,
                logger=entry_logger,
            )
            return {
                "one_line_summary": article.title,
                "category": "other",
                "tags": [],
                "importance": 3,
                "content_type": "news",
                "key_takeaway": article.summary or article.title,
            }
```

**Step 3: Add synthesize method**

Add after `extract_entry`:

```python
def synthesize(
    self,
    extractions: list[dict[str, Any]],
    logger: logging.Logger | None = None,
) -> str:
    prompt = build_synthesis_prompt(extractions, self.summary_cfg)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": self.summary_cfg.synthesis_max_output_tokens,
        },
    }
    with start_span(
        "gemini.synthesize",
        kind="llm",
        input_value=prompt,
        attributes={
            "llm.model": self.cfg.model,
            "llm.provider": "gemini",
        },
    ) as span:
        try:
            data = self._post(payload, timeout=120.0)
            content = _extract_text(data)
            set_span_output(span, content)
            if logger:
                from ...utils.logging import log_event
                log_event(logger, "Synthesis complete", event="llm_synthesis_response", status="ok")
            return content.strip()
        except httpx.HTTPError as exc:
            record_span_error(span, exc)
            if logger:
                from ...utils.logging import log_event
                log_event(logger, "Synthesis failed", event="llm_synthesis_response", status="error", error=str(exc))
            return f"Synthesis error: {type(exc).__name__}"
```

**Step 4: Update `_post` to accept optional timeout**

Change the `_post` method signature and body:

```python
def _post(self, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    url = f"{self.cfg.base_url}/v1beta/models/{self.cfg.model}:generateContent"
    params = {"key": self.api_key}
    with httpx.Client(timeout=timeout, trust_env=self.cfg.trust_env) as client:
        resp = client.post(url, params=params, json=payload)
        resp.raise_for_status()
        return resp.json()
```

**Step 5: Commit**

```bash
git add daily_feed/llm/providers/gemini.py
git commit -m "feat: implement extract_entry and synthesize in Gemini provider"
```

---

### Task 8: Implement extract_entry and synthesize in OpenAI provider

**Files:**
- Modify: `daily_feed/llm/providers/openai_compatible.py`

**Step 1: Add imports**

Update imports:
```python
from ..prompts import build_analysis_prompt, build_deep_fetch_prompt, build_extraction_prompt, build_synthesis_prompt
```

**Step 2: Add extract_entry method**

Same logic as Gemini but using OpenAI chat completions format:

```python
def extract_entry(
    self,
    article: Article,
    base_text: str,
    entry_logger: logging.Logger | None = None,
) -> dict[str, Any]:
    prompt = build_extraction_prompt(article, base_text, self.summary_cfg)
    payload = {
        "model": self.cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": self.summary_cfg.extraction_max_output_tokens,
    }
    with start_span(
        "openai_compatible.extract_entry",
        kind="llm",
        input_value=prompt,
        attributes={
            "llm.model": self.cfg.model,
            "llm.provider": self.cfg.name,
            "article.title": article.title,
            "article.site": article.site,
        },
    ) as span:
        try:
            data = self._post(payload)
            content = _extract_text(data)
            set_span_output(span, content)
            obj = _parse_json_response(content)
            self._log_llm_response(
                article=article,
                event="llm_extraction_response",
                status="ok",
                content=content,
                prompt=prompt,
                logger=entry_logger,
            )
            return obj
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            record_span_error(span, exc)
            self._log_llm_response(
                article=article,
                event="llm_extraction_response",
                status="error",
                content=str(exc),
                prompt=prompt,
                logger=entry_logger,
            )
            return {
                "one_line_summary": article.title,
                "category": "other",
                "tags": [],
                "importance": 3,
                "content_type": "news",
                "key_takeaway": article.summary or article.title,
            }
```

**Step 3: Add synthesize method**

```python
def synthesize(
    self,
    extractions: list[dict[str, Any]],
    logger: logging.Logger | None = None,
) -> str:
    prompt = build_synthesis_prompt(extractions, self.summary_cfg)
    payload = {
        "model": self.cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": self.summary_cfg.synthesis_max_output_tokens,
    }
    with start_span(
        "openai_compatible.synthesize",
        kind="llm",
        input_value=prompt,
        attributes={
            "llm.model": self.cfg.model,
            "llm.provider": self.cfg.name,
        },
    ) as span:
        try:
            data = self._post(payload, timeout=120.0)
            content = _extract_text(data)
            set_span_output(span, content)
            if logger:
                from ...utils.logging import log_event
                log_event(logger, "Synthesis complete", event="llm_synthesis_response", status="ok")
            return content.strip()
        except httpx.HTTPError as exc:
            record_span_error(span, exc)
            if logger:
                from ...utils.logging import log_event
                log_event(logger, "Synthesis failed", event="llm_synthesis_response", status="error", error=str(exc))
            return f"Synthesis error: {type(exc).__name__}"
```

**Step 4: Update `_post` to accept optional timeout**

```python
def _post(self, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    base = self.cfg.base_url.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=timeout, trust_env=self.cfg.trust_env) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()
```

**Step 5: Commit**

```bash
git add daily_feed/llm/providers/openai_compatible.py
git commit -m "feat: implement extract_entry and synthesize in OpenAI provider"
```

---

### Task 9: Update EntryAnalyzer to do extraction instead of analysis

**Files:**
- Modify: `daily_feed/analyzers/entry_analyzer.py`

**Step 1: Replace analyze() with extract()**

Replace the `analyze` method with a new `extract` method. The deep fetch logic stays the same but the final LLM call changes from `analyze_entry` to `extract_entry`:

```python
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

    # Write extraction cache
    entry.write_extraction_result(result)
    return result
```

Also update the imports at the top of the file:
```python
from ..core.types import ExtractionResult, ExtractedArticle
```

Remove the unused imports: `AnalysisResult`, `DeepFetchDecision`, `asyncio`, `_URL_RE`, and all the deep fetch helper functions. Keep them only if `enable_deep_fetch_decision` is still needed - but since we're replacing the pipeline, simplify by removing all deep fetch code from this file. The deep fetch infrastructure can be re-added later if needed.

The simplified file should just have:
- `EntryAnalyzer.__init__` (same)
- `EntryAnalyzer.extract` (new, replaces analyze)

Remove: `DeepFetchDecision`, `_extract_links`, `_select_deep_links`, `_write_deep_fetch`, `_short_hash`, `_is_placeholder_text` (the one in entry_analyzer.py; the one in runner.py stays), and all deep fetch logic in `analyze`.

**Step 2: Commit**

```bash
git add daily_feed/analyzers/entry_analyzer.py
git commit -m "feat: replace analyze() with extract() in EntryAnalyzer"
```

---

### Task 10: Update EntryManager with extraction cache support

**Files:**
- Modify: `daily_feed/core/entry.py`

**Step 1: Add extraction cache properties and methods**

Add new properties after existing ones:

```python
@property
def extraction_raw(self) -> Path:
    return self.folder / "extraction.json"
```

Add write/read methods:

```python
def write_extraction_result(self, result) -> None:
    """Write extraction result to cache."""
    from daily_feed.core.types import ExtractionResult
    if not isinstance(result, ExtractionResult):
        return
    data = {
        "one_line_summary": result.one_line_summary,
        "category": result.category,
        "tags": result.tags,
        "importance": result.importance,
        "content_type": result.content_type,
        "key_takeaway": result.key_takeaway,
        "status": result.status,
        "article": {
            "title": result.article.title,
            "url": result.article.url,
            "site": result.article.site,
        },
    }
    data.update(result.meta or {})
    with open(self.extraction_raw, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def read_extraction_result(self) -> dict[str, Any] | None:
    """Read extraction result from cache."""
    if not self.extraction_raw.exists():
        return None
    with open(self.extraction_raw, "r", encoding="utf-8") as f:
        return json.load(f)
```

**Step 2: Commit**

```bash
git add daily_feed/core/entry.py
git commit -m "feat: add extraction cache to EntryManager"
```

---

### Task 11: Create the Synthesizer class

**Files:**
- Create: `daily_feed/analyzers/synthesizer.py`

**Step 1: Write the synthesizer**

```python
"""Synthesizer for producing daily briefings from extraction results."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
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
    """Parse the LLM synthesis markdown into a BriefingResult."""
    result = BriefingResult(raw_text=text)

    sections = re.split(r'^## ', text, flags=re.MULTILINE)

    for section in sections:
        if not section.strip():
            continue

        lines = section.strip().split('\n', 1)
        heading = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        if '今日概要' in heading:
            result.executive_summary = body
        elif '重点报道' in heading:
            result.top_stories = _parse_top_stories(body)
        elif '主题板块' in heading:
            result.sections = _parse_themed_sections(body)
        elif '速览' in heading:
            result.quick_mentions = _parse_quick_mentions(body)

    return result


def _parse_top_stories(body: str) -> list[dict[str, Any]]:
    """Parse ### headed top stories."""
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
    """Parse ### headed themed sections."""
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
    """Parse bullet list quick mentions."""
    mentions = []
    for line in body.strip().split('\n'):
        line = line.strip()
        if line.startswith('- '):
            line = line[2:]
        elif line.startswith('* '):
            line = line[2:]
        else:
            continue
        # Try to extract **title**: description format
        m = re.match(r'\*\*(.+?)\*\*[:\uff1a]\s*(.*)', line)
        if m:
            mentions.append({"title": m.group(1).strip(), "description": m.group(2).strip()})
        else:
            mentions.append({"title": line, "description": ""})
    return mentions
```

**Step 2: Commit**

```bash
git add daily_feed/analyzers/synthesizer.py
git commit -m "feat: add Synthesizer class with markdown parser"
```

---

### Task 12: Create the briefing HTML template

**Files:**
- Create: `daily_feed/output/templates/briefing.html`

**Step 1: Write the briefing template**

```html
{#
  Daily Briefing Report Template

  Template Variables:
    title: Report title
    generated_at: ISO timestamp of report generation
    total: Total number of articles
    briefing: BriefingResult object
    articles_by_url: dict mapping URL to article metadata for linking
#}
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>
  <style>
    :root {
      --bg: #f6f5f2;
      --card: #ffffff;
      --text: #1a1a1a;
      --muted: #666;
      --accent: #0f766e;
      --accent-light: #f0fdfa;
      --border: #e4e4e4;
      --importance-5: #dc2626;
      --importance-4: #ea580c;
      --importance-3: #0f766e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, "SF Pro Text", "Helvetica Neue", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      background: radial-gradient(circle at top, #ffffff 0%, #f6f5f2 40%, #efece6 100%);
      color: var(--text);
      line-height: 1.8;
    }
    header {
      padding: 2.5rem 1.5rem 1.5rem;
      text-align: center;
    }
    header h1 { margin: 0; font-size: 2rem; }
    header p { margin: 0.5rem 0 0; color: var(--muted); }
    main {
      max-width: 860px;
      margin: 0 auto 3rem;
      padding: 0 1.5rem;
    }
    .card {
      margin-bottom: 1.5rem;
      border: 1px solid var(--border);
      background: var(--card);
      border-radius: 12px;
      padding: 1.5rem;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
    }
    .executive-summary {
      border-left: 4px solid var(--accent);
      background: var(--accent-light);
    }
    .executive-summary p { margin: 0; }
    .section-title {
      margin: 2rem 0 1rem;
      font-size: 1.3rem;
      color: var(--accent);
      border-bottom: 2px solid var(--border);
      padding-bottom: 0.4rem;
    }
    .top-story { margin-bottom: 1.2rem; }
    .top-story h3 {
      margin: 0 0 0.3rem;
      font-size: 1.1rem;
    }
    .top-story h3 a {
      color: var(--text);
      text-decoration: none;
    }
    .top-story h3 a:hover { color: var(--accent); }
    .top-story .analysis { color: #333; margin: 0; }
    .theme-section { margin-bottom: 1.5rem; }
    .theme-section h3 {
      margin: 0 0 0.4rem;
      font-size: 1.1rem;
      color: var(--accent);
    }
    .theme-section .description {
      margin: 0;
      white-space: pre-wrap;
    }
    .quick-mentions ul {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .quick-mentions li {
      padding: 0.4rem 0;
      border-bottom: 1px dashed var(--border);
    }
    .quick-mentions li:last-child { border-bottom: none; }
    .quick-mentions .mention-title { font-weight: 600; }
    .meta { color: var(--muted); font-size: 0.85rem; }
    .meta a { color: var(--accent); text-decoration: none; }
    @media (max-width: 600px) {
      header h1 { font-size: 1.5rem; }
      .card { padding: 1rem; }
    }
  </style>
</head>
<body>
  <header>
    <h1>{{ title }}</h1>
    <p>{{ total }} 篇文章 · 生成于 {{ generated_at }}</p>
  </header>
  <main>
    {% if briefing.executive_summary %}
    <div class="card executive-summary">
      <p>{{ briefing.executive_summary }}</p>
    </div>
    {% endif %}

    {% if briefing.top_stories %}
    <h2 class="section-title">重点报道</h2>
    <div class="card">
      {% for story in briefing.top_stories %}
      <div class="top-story">
        <h3>
          {% if articles_by_title.get(story.title) %}
          <a href="{{ articles_by_title[story.title].url }}" target="_blank" rel="noopener">{{ story.title }}</a>
          {% else %}
          {{ story.title }}
          {% endif %}
        </h3>
        <p class="analysis">{{ story.analysis }}</p>
      </div>
      {% endfor %}
    </div>
    {% endif %}

    {% if briefing.sections %}
    <h2 class="section-title">主题板块</h2>
    {% for section in briefing.sections %}
    <div class="card theme-section">
      <h3>{{ section.theme }}</h3>
      <div class="description">{{ section.description }}</div>
    </div>
    {% endfor %}
    {% endif %}

    {% if briefing.quick_mentions %}
    <h2 class="section-title">速览</h2>
    <div class="card quick-mentions">
      <ul>
        {% for mention in briefing.quick_mentions %}
        <li>
          <span class="mention-title">{{ mention.title }}</span>{% if mention.description %}：{{ mention.description }}{% endif %}
        </li>
        {% endfor %}
      </ul>
    </div>
    {% endif %}

    {% if not briefing.executive_summary and not briefing.top_stories %}
    <div class="card">
      <pre style="white-space: pre-wrap;">{{ briefing.raw_text }}</pre>
    </div>
    {% endif %}
  </main>
</body>
</html>
```

**Step 2: Commit**

```bash
git add daily_feed/output/templates/briefing.html
git commit -m "feat: add briefing HTML template"
```

---

### Task 13: Update renderer with render_briefing

**Files:**
- Modify: `daily_feed/output/renderer.py`

**Step 1: Add render_briefing function**

Add after the existing `render_markdown` function:

```python
def render_briefing(
    briefing: "BriefingResult",
    extractions: list["ExtractionResult"],
    output_path: Path,
    title: str,
) -> None:
    """Render a BriefingResult as an HTML report.

    Args:
        briefing: The synthesized briefing result
        extractions: All extraction results (for article URL lookup)
        output_path: Path where the HTML file will be written
        title: Report title
    """
    from ..core.types import BriefingResult, ExtractionResult

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("briefing.html")

    # Build title->article lookup for linking
    articles_by_title = {}
    for ext in extractions:
        articles_by_title[ext.article.title] = ext.article

    html = template.render(
        title=title,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        total=len(extractions),
        briefing=briefing,
        articles_by_title=articles_by_title,
    )
    output_path.write_text(html, encoding="utf-8")
```

**Step 2: Commit**

```bash
git add daily_feed/output/renderer.py
git commit -m "feat: add render_briefing to renderer"
```

---

### Task 14: Rewire the pipeline in runner.py

This is the largest task. Replace the analysis stage with extraction + synthesis.

**Files:**
- Modify: `daily_feed/runner.py`

**Step 1: Update imports**

Replace:
```python
from .core.types import AnalysisResult, ExtractedArticle
from .output.renderer import render_html, render_markdown
```

With:
```python
from .core.types import AnalysisResult, ExtractionResult, ExtractedArticle, BriefingResult
from .output.renderer import render_briefing, render_html, render_markdown
from .analyzers.synthesizer import Synthesizer
```

**Step 2: Replace _analyze_entries with _extract_entries**

Replace the entire `_analyze_entries` function with `_extract_entries` that does extraction instead of analysis:

```python
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

    def _advance_progress() -> None:
        if progress and extract_task is not None:
            progress.advance(extract_task, 1)

    if concurrency == 1:
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

    results: list[ExtractionResult | None] = [None] * len(extracted)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {}
        for idx, item in enumerate(extracted):
            entry = EntryManager(articles_dir, item.article)
            cached = _read_cached_extraction(entry, cfg)
            if cached:
                results[idx] = _build_cached_extraction(item, cached)
                log_event(logger, "Extraction cache hit", event="extraction_cache_hit", url=item.article.url)
                _advance_progress()
                continue
            ctx = copy_context()
            future = executor.submit(ctx.run, analyzer.extract, item)
            future_map[future] = idx

        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
            _advance_progress()

    if any(r is None for r in results):
        raise RuntimeError("Extraction results incomplete")
    return [r for r in results if r is not None]
```

**Step 3: Add cached extraction helpers**

```python
def _read_cached_extraction(entry: EntryManager, cfg: AppConfig) -> dict | None:
    if not _is_cache_enabled(cfg):
        return None
    cached = entry.read_extraction_result()
    if cached and EntryManager.is_entry_valid(entry.folder, cfg.cache.ttl_days):
        return cached
    return None


def _build_cached_extraction(item: ExtractedArticle, cached: dict) -> ExtractionResult:
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
```

**Step 4: Rewire the pipeline stages**

In `run_pipeline`, for both the progress and quiet code paths, replace the analysis stage + render stage with:

```python
# Stage 4: Extract (Pass 1)
provider = _build_provider(cfg, None)
analyzer = EntryAnalyzer(cfg, provider, articles_dir, logger)
extractions = _extract_entries(extracted, articles_dir, cfg, analyzer, logger)

# Stage 5: Synthesize (Pass 2)
synthesizer = Synthesizer(cfg, provider, logger)
briefing = synthesizer.synthesize(extractions)

# Stage 6: Render
title = f"Daily Feed - {input_path.stem}"
html_path = run_output_dir / "report.html"
render_briefing(briefing, extractions, html_path, title)
```

Keep the old `render_html` and `render_markdown` functions in place (they just won't be called from the main pipeline). Remove `_analyze_entries`, `_read_cached_analysis`, `_build_cached_result`.

**Step 5: Commit**

```bash
git add daily_feed/runner.py
git commit -m "feat: rewire pipeline to two-pass extraction + synthesis"
```

---

### Task 15: Write tests for the new components

**Files:**
- Create: `tests/test_synthesis.py`

**Step 1: Test parse_briefing_markdown**

```python
"""Tests for the synthesis module."""

from daily_feed.analyzers.synthesizer import parse_briefing_markdown


def test_parse_briefing_basic():
    md = """## 今日概要
今天AI工具非常火爆。

## 重点报道
### Skillkit
这是一个很重要的工具。

### Nativeline
原生Swift开发。

## 主题板块
### AI开发工具
多个AI工具发布了。

### 产品发布
今天有很多新产品。

## 速览
- **Felsius**: 双单位天气显示
- **PinMe**: 截图固定工具
"""
    result = parse_briefing_markdown(md)
    assert "AI工具" in result.executive_summary
    assert len(result.top_stories) == 2
    assert result.top_stories[0]["title"] == "Skillkit"
    assert len(result.sections) == 2
    assert result.sections[0].theme == "AI开发工具"
    assert len(result.quick_mentions) == 2
    assert result.quick_mentions[0]["title"] == "Felsius"


def test_parse_briefing_empty():
    result = parse_briefing_markdown("")
    assert result.executive_summary == ""
    assert result.top_stories == []
    assert result.sections == []
    assert result.quick_mentions == []
    assert result.raw_text == ""


def test_parse_briefing_fallback_raw():
    md = "Some random LLM output without proper headings"
    result = parse_briefing_markdown(md)
    assert result.raw_text == md
```

**Step 2: Run tests**

```bash
pytest tests/test_synthesis.py -v
```

**Step 3: Test prompt builders**

Add to `tests/test_synthesis.py`:

```python
from daily_feed.llm.prompts import build_extraction_prompt, build_synthesis_prompt
from daily_feed.core.types import Article
from daily_feed.config import SummaryConfig


def test_build_extraction_prompt():
    article = Article(title="Test", site="HN", url="https://example.com")
    cfg = SummaryConfig(max_chars=500)
    prompt = build_extraction_prompt(article, "Some article text", cfg)
    assert "Test" in prompt
    assert "HN" in prompt
    assert "Some article text" in prompt
    assert "one_line_summary" in prompt


def test_build_synthesis_prompt():
    cfg = SummaryConfig()
    extractions = [{"title": "Test", "importance": 4}]
    prompt = build_synthesis_prompt(extractions, cfg)
    assert "1" in prompt  # count
    assert "Test" in prompt
    assert "今日概要" in prompt
```

**Step 4: Run all tests**

```bash
pytest tests/test_synthesis.py -v
```

**Step 5: Commit**

```bash
git add tests/test_synthesis.py
git commit -m "test: add tests for synthesis parser and prompt builders"
```

---

### Task 16: Clean up old analysis code and run full test suite

**Files:**
- Modify: `daily_feed/runner.py` (remove old _analyze_entries, _read_cached_analysis, _build_cached_result if not already done)

**Step 1: Remove dead code**

Remove from `runner.py`:
- `_read_cached_analysis` function
- `_build_cached_result` function
- `_analyze_entries` function (replaced by `_extract_entries`)

Keep `_fetch_and_extract` as it's used by tests.

**Step 2: Run existing tests to check for regressions**

```bash
pytest tests/ -v --tb=short 2>&1 | head -80
```

Fix any imports or references that break. The main things to watch for:
- `tests/test_integration_per_entry.py` likely imports `AnalysisResult` and calls `analyzer.analyze()` - these need updating to use `extract()` and `ExtractionResult`
- `tests/test_entry_manager.py` may reference `write_analysis_result` - keep that method since it still works

**Step 3: Fix broken tests**

Update test files as needed to use the new API (`extract` instead of `analyze`, `ExtractionResult` instead of `AnalysisResult` where applicable).

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: clean up old analysis code, fix tests"
```

---

### Task 17: End-to-end validation

**Step 1: Run the pipeline with existing data**

```bash
cd /Users/deming/Projects/personal_projects/agents_projects/daily_feed
python -m daily_feed.cli run -i data/folo-export-2026-02-08-11-40.json -o out --no-use-cache
```

Check the generated report visually. Verify:
- Executive summary appears in Chinese
- Top stories section has 3-5 highlighted articles
- Themed sections group related articles
- Quick mentions are compact
- All links work

**Step 2: If output looks wrong, iterate on prompts**

The synthesis prompt may need tuning. Adjust `prompts/synthesis.md` based on the actual output quality.

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete two-pass briefing pipeline"
```
