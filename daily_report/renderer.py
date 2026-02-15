"""Report renderer for Daily Report."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

from .core.types import ExtractionResult


def render_briefing(
    briefing: Any,
    extractions: list[ExtractionResult],
    output_path: Path,
    title: str,
) -> None:
    """Render full HTML briefing page to disk."""
    output_path.write_text(render_html(briefing, extractions, title), encoding="utf-8")


def render_html(briefing: Any, extractions: list[ExtractionResult], title: str) -> str:
    """Render full HTML page."""
    briefing_html = _format_briefing(briefing)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    toc_items = ['<li><a href="#briefing">今日综述</a></li>']
    article_cards: list[str] = []

    for idx, ex in enumerate(extractions, start=1):
        article_id = f"article-{idx}"
        safe_title = escape(ex.article.title or f"文章 {idx}")
        safe_site = escape(ex.article.site or "Unknown")
        safe_url = escape(ex.article.url or "#", quote=True)
        summary = _format_inline(ex.one_line_summary or "")
        takeaway = _format_inline(ex.key_takeaway or "")
        category = escape((ex.category or "general").upper())
        importance = max(1, int(ex.importance or 1))
        importance_badge = "★" * min(5, importance)
        tags = "".join(
            f'<span class="tag">{_format_inline(tag)}</span>' for tag in (ex.tags or [])
        )
        content_type = escape(ex.content_type or "article")

        toc_items.append(f'<li><a href="#{article_id}">{safe_title}</a></li>')

        article_cards.append(
            f"""
            <article class="card" id="{article_id}">
              <div class="meta-row">
                <span class="chip">{category}</span>
                <span class="importance">{importance_badge}</span>
              </div>
              <h2><a href="{safe_url}" target="_blank" rel="noopener noreferrer">{safe_title}</a></h2>
              <div class="tags">{tags}</div>
              <p class="summary">{summary}</p>
              <p class="takeaway"><strong>关键观点：</strong>{takeaway}</p>
              <div class="meta-row muted">
                <span>来源：{safe_site}</span>
                <span>类型：{content_type}</span>
              </div>
            </article>
            """
        )

    articles_html = "\n".join(article_cards)
    toc_html = "\n".join(toc_items)
    safe_title = escape(title)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --text: #142033;
      --muted: #607089;
      --accent: #1f6feb;
      --line: #dfe7f2;
      --chip-bg: #edf4ff;
      --chip-text: #1e4f9e;
      --shadow: 0 14px 36px rgba(20, 32, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--text);
      background: radial-gradient(circle at 10% -20%, #dde9ff 0%, transparent 40%), var(--bg);
      font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Arial, sans-serif;
      line-height: 1.6;
    }}
    .layout {{
      max-width: 1300px;
      margin: 0 auto;
      padding: 28px 24px 48px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 280px;
      gap: 28px;
    }}
    .main {{
      min-width: 0;
    }}
    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
    }}
    .home-btn {{
      display: inline-block;
      padding: 9px 14px;
      background: var(--accent);
      color: #fff;
      text-decoration: none;
      border-radius: 10px;
      font-size: 14px;
      font-weight: 600;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 24px 26px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(1.5rem, 2.7vw, 2.1rem);
      line-height: 1.25;
    }}
    .muted {{
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .briefing {{
      padding: 24px 26px;
      margin-bottom: 18px;
    }}
    .briefing h2,
    .briefing h3,
    .briefing h4 {{
      margin: 0 0 10px;
      line-height: 1.4;
    }}
    .briefing p {{
      margin: 0 0 12px;
    }}
    .briefing ul {{
      margin: 0 0 12px;
      padding-left: 20px;
    }}
    .articles {{
      display: grid;
      gap: 14px;
    }}
    .card {{
      padding: 18px 20px;
    }}
    .card h2 {{
      font-size: 1.1rem;
      margin: 8px 0;
      line-height: 1.5;
    }}
    .card a {{
      color: inherit;
      text-decoration: none;
    }}
    .card a:hover {{
      color: var(--accent);
    }}
    .meta-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .chip {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 999px;
      background: var(--chip-bg);
      color: var(--chip-text);
      font-size: 12px;
      font-weight: 700;
      border: 0.5px solid #bfd7ff;
      letter-spacing: 0.04em;
    }}
    .importance {{
      color: #b42318;
      font-size: 13px;
      letter-spacing: 0.08em;
    }}
    .tags {{
      margin: 8px 0 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .tag {{
      display: inline-block;
      padding: 2px 9px;
      border-radius: 999px;
      background: #f8fbff;
      border: 0.5px solid #ccd7ea;
      color: #324763;
      font-size: 12px;
    }}
    .summary {{
      margin: 0 0 10px;
    }}
    .takeaway {{
      margin: 0;
      padding: 10px 12px;
      border-left: 3px solid var(--accent);
      border-radius: 10px;
      background: #f6f9ff;
    }}
    .sidebar {{
      position: sticky;
      top: 18px;
      align-self: start;
      padding: 16px;
      max-height: calc(100vh - 36px);
      overflow: auto;
    }}
    .sidebar h3 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
    .toc {{
      margin: 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 6px;
    }}
    .toc a {{
      color: var(--muted);
      text-decoration: none;
      display: block;
      padding: 5px 8px;
      border-radius: 8px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .toc a:hover {{
      color: var(--accent);
      background: #f1f6ff;
    }}
    @media (max-width: 1080px) {{
      .layout {{
        grid-template-columns: minmax(0, 1fr);
      }}
      .sidebar {{
        position: static;
        order: -1;
        max-height: none;
      }}
      .toc {{
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      }}
    }}
    @media (max-width: 640px) {{
      .layout {{
        padding: 16px 12px 32px;
        gap: 12px;
      }}
      .hero, .briefing, .card, .sidebar {{
        padding: 14px;
        border-radius: 14px;
      }}
      .topbar {{
        margin-bottom: 12px;
      }}
      .home-btn {{
        padding: 8px 11px;
        font-size: 13px;
      }}
      .toc {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <main class="main">
      <div class="topbar">
        <a class="home-btn" href="/">返回主页</a>
      </div>

      <header class="panel hero">
        <h1>{safe_title}</h1>
        <div class="muted">生成时间：{generated_at}</div>
      </header>

      <section class="panel briefing" id="briefing">
        {briefing_html}
      </section>

      <section class="articles">
        {articles_html}
      </section>
    </main>

    <aside class="panel sidebar" aria-label="目录">
      <a class="home-btn" href="/" style="display:block;text-align:center;margin-bottom:12px;">返回主页</a>
      <h3>目录</h3>
      <ul class="toc">
        {toc_html}
      </ul>
    </aside>
  </div>
</body>
</html>
"""


def render_markdown(
    briefing: Any,
    extractions: list[ExtractionResult],
    output_path: Path | None = None,
    title: str = "Daily Report",
) -> str:
    """Render report as markdown and optionally write to disk."""
    briefing_text = _get_briefing_text(briefing).strip()

    lines = [f"# {title}", "", "## 今日综述", "", briefing_text or "(无)"]
    for idx, ex in enumerate(extractions, start=1):
        lines.extend(
            [
                "",
                f"## {idx}. {ex.article.title}",
                f"- URL: {ex.article.url}",
                f"- 来源: {ex.article.site}",
                f"- 分类: {ex.category or 'general'}",
                f"- 重要度: {ex.importance}",
                f"- 摘要: {ex.one_line_summary or '(无)'}",
                f"- 关键观点: {ex.key_takeaway or '(无)'}",
            ]
        )
        if ex.tags:
            lines.append(f"- 标签: {', '.join(ex.tags)}")

    text = "\n".join(lines) + "\n"
    if output_path is not None:
        output_path.write_text(text, encoding="utf-8")
    return text


def _format_briefing(briefing: Any) -> str:
    """Convert briefing markdown-like text into safe HTML."""
    text = _get_briefing_text(briefing).strip()
    if not text:
        return "<h2>今日综述</h2><p>暂无综述内容。</p>"

    out = ["<h2>今日综述</h2>"]
    paragraph_lines: list[str] = []
    list_items: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            joined = "<br>".join(_format_inline(line) for line in paragraph_lines)
            out.append(f"<p>{joined}</p>")
            paragraph_lines.clear()

    def flush_list() -> None:
        if list_items:
            out.append(f"<ul>{''.join(list_items)}</ul>")
            list_items.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_list()
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            level = len(heading.group(1))
            html_level = min(6, level + 1)
            out.append(f"<h{html_level}>{_format_inline(heading.group(2).strip())}</h{html_level}>")
            continue

        item = re.match(r"^[-*]\s+(.+)$", line)
        if item:
            flush_paragraph()
            list_items.append(f"<li>{_format_inline(item.group(1).strip())}</li>")
            continue

        flush_list()
        paragraph_lines.append(line)

    flush_paragraph()
    flush_list()
    return "".join(out)


def _get_briefing_text(briefing: Any) -> str:
    raw_text = getattr(briefing, "raw_text", None)
    if isinstance(raw_text, str) and raw_text.strip():
        return raw_text
    if isinstance(briefing, str):
        return briefing
    if raw_text is not None:
        return str(raw_text)
    return str(briefing or "")


def _format_inline(text: str) -> str:
    """Render limited inline markdown safely."""
    escaped = escape(text)

    code_pattern = re.compile(r"`([^`]+)`")
    code_store: dict[str, str] = {}

    def store_code(match: re.Match[str]) -> str:
        token = f"__CODE_{len(code_store)}__"
        code_store[token] = f"<code>{match.group(1)}</code>"
        return token

    escaped = code_pattern.sub(store_code, escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        lambda m: (
            f'<a href="{escape(m.group(2), quote=True)}" target="_blank" '
            f'rel="noopener noreferrer">{m.group(1)}</a>'
        ),
        escaped,
    )
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)

    for token, value in code_store.items():
        escaped = escaped.replace(token, value)

    return escaped
