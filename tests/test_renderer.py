from pathlib import Path

from daily_report.core.types import Article, BriefingResult, ExtractionResult
from daily_report.output.renderer import _format_briefing, render_markdown


def _sample_extractions() -> list[ExtractionResult]:
    article = Article(title="t", site="s", url="https://example.com")
    return [ExtractionResult(article=article)]


def test_format_briefing_paragraphs_and_lists():
    briefing = "第一段第一行\n第一段第二行\n\n- 条目A\n- 条目B\n\n第二段"
    rendered = _format_briefing(briefing)

    assert "<p>第一段第一行<br>第一段第二行</p>" in rendered
    assert "<ul><li>条目A</li><li>条目B</li></ul>" in rendered
    assert "<p>第二段</p>" in rendered


def test_format_briefing_inline_markdown_and_escape():
    briefing = "**加粗** *斜体* `code` [链接](https://example.com) <script>"
    rendered = _format_briefing(briefing)

    assert "<strong>加粗</strong>" in rendered
    assert "<em>斜体</em>" in rendered
    assert "<code>code</code>" in rendered
    assert 'href="https://example.com"' in rendered
    assert "&lt;script&gt;" in rendered


def test_format_briefing_accepts_briefing_result():
    result = BriefingResult(raw_text="## 小结\n内容段落")
    rendered = _format_briefing(result)

    assert "<h3>小结</h3>" in rendered
    assert "<p>内容段落</p>" in rendered


def test_render_markdown_uses_briefing_text_instead_of_repr(tmp_path: Path):
    output_path = tmp_path / "report.md"
    briefing = BriefingResult(raw_text="今日概览\n- A")
    render_markdown(
        briefing=briefing,
        extractions=_sample_extractions(),
        output_path=output_path,
        title="Title",
    )

    text = output_path.read_text(encoding="utf-8")
    assert "BriefingResult(" not in text
    assert "今日概览" in text
