from pathlib import Path

from daily_report.core.types import (
    AnalysisResult,
    Article,
    BriefingResult,
    BriefingSection,
    ExtractionResult,
)
from daily_report.output.renderer import render_briefing, render_markdown


def _sample_article(*, title: str, site: str = "Example Site", url: str = "https://example.com") -> Article:
    return Article(title=title, site=site, url=url)


def test_render_markdown_outputs_grouped_sections(tmp_path: Path) -> None:
    output_path = tmp_path / "report.md"
    results = [
        AnalysisResult(
            article=_sample_article(title="A1", site="Tech"),
            analysis="分析A1",
        ),
        AnalysisResult(
            article=_sample_article(title="A2", site="Tech"),
            analysis="分析A2",
        ),
        AnalysisResult(
            article=_sample_article(title="B1", site="News"),
            analysis="分析B1",
        ),
    ]

    render_markdown(results=results, output_path=output_path, title="日报标题")
    text = output_path.read_text(encoding="utf-8")

    assert "# 日报标题" in text
    assert "## Tech" in text
    assert "## News" in text
    assert "### A1" in text
    assert "### B1" in text
    assert "分析A2" in text


def test_render_briefing_links_top_story_and_escapes_html(tmp_path: Path) -> None:
    output_path = tmp_path / "briefing.html"
    article = _sample_article(title="故事A", url="https://example.com/story-a")
    extractions = [ExtractionResult(article=article)]
    briefing = BriefingResult(
        executive_summary="总结<script>alert(1)</script>",
        top_stories=[{"title": "故事A", "analysis": "重点分析"}],
        sections=[BriefingSection(theme="趋势", description="描述内容")],
        quick_mentions=[{"title": "速览A", "description": "描述A"}],
    )

    render_briefing(
        briefing=briefing,
        extractions=extractions,
        output_path=output_path,
        title="日报",
    )
    html = output_path.read_text(encoding="utf-8")

    assert '<a href="https://example.com/story-a"' in html
    assert "重点分析" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<h2>趋势</h2>" in html
    assert "速览A" in html


def test_render_briefing_fallback_uses_raw_text(tmp_path: Path) -> None:
    output_path = tmp_path / "briefing-fallback.html"
    briefing = BriefingResult(raw_text="回退文本内容")

    render_briefing(
        briefing=briefing,
        extractions=[],
        output_path=output_path,
        title="日报",
    )
    html = output_path.read_text(encoding="utf-8")

    assert "回退文本内容" in html
