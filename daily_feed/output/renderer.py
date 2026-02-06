"""
Report rendering for HTML and Markdown output.

This module generates the final reports using Jinja2 templates for HTML
and custom formatting for Markdown.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .core.types import ArticleSummary


def _slugify(value: str) -> str:
    """Convert a string to a URL-safe slug.

    Converts to lowercase, replaces non-alphanumeric characters with
    hyphens, and collapses consecutive hyphens.

    Args:
        value: The string to slugify

    Returns:
        A URL-safe slug string

    Examples:
        >>> _slugify("Hello World!")
        "hello-world"
        >>> _slugify("Tech & Science")
        "tech-science"
    """
    lowered = value.strip().lower()
    cleaned = []
    last_dash = False
    for ch in lowered:
        if ch.isalnum():
            cleaned.append(ch)
            last_dash = False
        else:
            # Only add dash if previous char wasn't a dash (avoid consecutive dashes)
            if not last_dash:
                cleaned.append("-")
                last_dash = True
    slug = "".join(cleaned).strip("-")
    return slug or "section"


def render_html(summaries: list[ArticleSummary], output_path: Path, title: str) -> None:
    """Render summaries as an HTML report using Jinja2 template.

    Groups summaries by topic (or site if no topic), generates a table
    of contents, and creates anchor links for navigation.

    Args:
        summaries: List of ArticleSummary objects to render
        output_path: Path where the HTML file will be written
        title: Report title displayed in the header
    """
    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent / "templates")),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    # Group by topic or site
    grouped = defaultdict(list)
    for summary in summaries:
        group = summary.topic or summary.article.site
        grouped[group].append(summary)

    # Sort groups by size (descending), then alphabetically
    sorted_groups = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0].lower()))
    used_ids: dict[str, int] = {}
    groups = []
    for group_name, items in sorted_groups:
        # Generate unique ID for each group (handle duplicate slugs)
        base_id = _slugify(group_name)
        count = used_ids.get(base_id, 0)
        used_ids[base_id] = count + 1
        group_id = f"{base_id}-{count + 1}" if count else base_id
        article_ids: dict[str, int] = {}
        enriched_items = []
        for item in items:
            # Generate unique ID for each article within the group
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

    # Render the template with all data
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
    """Render summaries as a Markdown report.

    Groups summaries by topic (or site if no topic) and formats
    with standard Markdown headings and lists.

    Args:
        summaries: List of ArticleSummary objects to render
        output_path: Path where the Markdown file will be written
        title: Report title for the top-level heading
    """
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
