from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .types import ArticleSummary


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

    html = template.render(
        title=title,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        groups=sorted_groups,
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
