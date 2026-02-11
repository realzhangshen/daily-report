"""Regression tests for Gemini response parsing and extraction status handling."""

from __future__ import annotations

from daily_report.analyzers.entry_analyzer import EntryAnalyzer
from daily_report.config import AppConfig
from daily_report.core.types import Article, ExtractedArticle
from daily_report.llm.providers.gemini import _extract_text


class _DummyProvider:
    """Minimal provider stub for EntryAnalyzer tests."""

    def __init__(self, raw_payload):
        self._raw_payload = raw_payload
        self.cfg = type("Cfg", (), {"model": "dummy-model"})()

    def extract_entry(self, article, base_text, entry_logger=None):  # noqa: ANN001
        return self._raw_payload


def test_extract_text_joins_non_thought_parts():
    data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "internal reasoning"},
                        {"text": '{"one_line_summary": "A"'},
                        {"text": ', "importance": 4}'},
                    ]
                }
            }
        ]
    }

    assert _extract_text(data) == '{"one_line_summary": "A", "importance": 4}'


def test_extract_text_falls_back_to_all_text_when_only_thought():
    data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "first"},
                        {"thought": True, "text": " second"},
                    ]
                }
            }
        ]
    }

    assert _extract_text(data) == "first second"


def test_entry_analyzer_propagates_provider_error_status(tmp_path):
    article = Article(
        title="Example",
        site="Site",
        url="https://example.com/article",
    )
    item = ExtractedArticle(article=article, text="content")
    provider = _DummyProvider({"summary": "fallback summary", "error": "parse_error"})
    analyzer = EntryAnalyzer(AppConfig(), provider, tmp_path)

    result = analyzer.extract(item)

    assert result.one_line_summary == "fallback summary"
    assert result.status == "parse_error"
    assert result.meta["error"] == "parse_error"


def test_entry_analyzer_normalizes_invalid_fields(tmp_path):
    article = Article(
        title="Fallback Title",
        site="Site",
        url="https://example.com/fallback",
    )
    item = ExtractedArticle(article=article, text="content")
    provider = _DummyProvider(
        {
            "one_line_summary": "",
            "category": "",
            "tags": "not-a-list",
            "importance": "99",
            "content_type": "",
            "key_takeaway": "",
            "status": "provider_error",
            "error": "provider_error",
        }
    )
    analyzer = EntryAnalyzer(AppConfig(), provider, tmp_path)

    result = analyzer.extract(item)

    assert result.one_line_summary == "Fallback Title"
    assert result.category == "other"
    assert result.tags == []
    assert result.importance == 5
    assert result.content_type == "news"
    assert result.status == "provider_error"
