from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .types import ArticleSummary


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    cleaned = []
    last_dash = False
    for ch in lowered:
        if ch.isalnum():
            cleaned.append(ch)
            last_dash = False
        else:
            if not last_dash:
                cleaned.append("-")
                last_dash = True
    slug = "".join(cleaned).strip("-")
    return slug or "section"


def render_html(summaries: list[ArticleSummary], output_path: Path, title: str) -> None:
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    grouped = defaultdict(list)
    for summary in summaries:
        group = summary.topic or summary.article.site
        grouped[group].append(summary)

    sorted_groups = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0].lower()))
    used_ids: dict[str, int] = {}
    groups = []
    for group_name, items in sorted_groups:
        base_id = _slugify(group_name)
        count = used_ids.get(base_id, 0)
        used_ids[base_id] = count + 1
        group_id = f"{base_id}-{count + 1}" if count else base_id
        article_ids: dict[str, int] = {}
        enriched_items = []
        for item in items:
            title_base = _slugify(item.article.title or "article")
            title_count = article_ids.get(title_base, 0)
            article_ids[title_base] = title_count + 1
            article_id = (
                f"{group_id}-{title_base}-{title_count + 1}"
                if title_count
                else f"{group_id}-{title_base}"
            )
            enriched_items.append({"id": article_id, "summary": item})
        groups.append(
            {
                "id": group_id,
                "name": group_name,
                "articles": enriched_items,
                "count": len(items),
            }
        )

    html = template.render(
        title=title,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        groups=groups,
        toc=[
            {
                "id": group["id"],
                "name": group["name"],
                "count": group["count"],
                "toc_items": [
                    {"id": item["id"], "title": item["summary"].article.title}
                    for item in group["articles"]
                ],
            }
            for group in groups
        ],
        total=len(summaries),
    )
    output_path.write_text(html, encoding="utf-8")


def render_markdown(summaries: list[ArticleSummary], output_path: Path, title: str) -> None:
    grouped = defaultdict(list)
    for summary in summaries:
        group = summary.topic or summary.article.site
        grouped[group].append(summary)

    lines = [f"# {title}", "", f"Total: {len(summaries)}", ""]
    for group, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0].lower())):
        lines.append(f"## {group}")
        lines.append("")
        for summary in items:
            art = summary.article
            lines.append(f"### {art.title}")
            lines.append(f"- Source: {art.site}{(' @' + art.author) if art.author else ''}")
            if art.time:
                lines.append(f"- Time: {art.time}")
            lines.append(f"- Link: {art.url}")
            if summary.bullets:
                lines.append("- Key Points:")
                for bullet in summary.bullets:
                    lines.append(f"  - {bullet}")
            if summary.takeaway:
                lines.append(f"- Takeaway: {summary.takeaway}")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
