"""Tests for the synthesis module and prompt builders."""

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


def test_parse_quick_mentions_chinese_colon():
    """Test that Chinese colon is also handled."""
    from daily_feed.analyzers.synthesizer import _parse_quick_mentions
    body = "- **测试**：这是描述\n- **另一个**: 英文冒号"
    mentions = _parse_quick_mentions(body)
    assert len(mentions) == 2
    assert mentions[0]["title"] == "测试"
    assert mentions[0]["description"] == "这是描述"
    assert mentions[1]["title"] == "另一个"


# --- Prompt builder tests ---

from daily_feed.llm.prompts import build_extraction_prompt, build_synthesis_prompt
from daily_feed.core.types import Article
from daily_feed.config import SummaryConfig


def test_build_extraction_prompt():
    article = Article(title="Test Article", site="HN", url="https://example.com")
    cfg = SummaryConfig(max_chars=500)
    prompt = build_extraction_prompt(article, "Some article text here", cfg)
    assert "Test Article" in prompt
    assert "HN" in prompt
    assert "Some article text here" in prompt
    assert "one_line_summary" in prompt


def test_build_extraction_prompt_truncates():
    article = Article(title="Test", site="HN", url="https://example.com")
    cfg = SummaryConfig(max_chars=10)
    prompt = build_extraction_prompt(article, "A" * 100, cfg)
    # Content should be truncated to 10 chars
    assert "A" * 100 not in prompt
    assert "A" * 10 in prompt


def test_build_synthesis_prompt():
    cfg = SummaryConfig()
    extractions = [{"title": "Test", "importance": 4, "category": "ai"}]
    prompt = build_synthesis_prompt(extractions, cfg)
    assert "1" in prompt  # count
    assert "Test" in prompt
    assert "今日概要" in prompt
