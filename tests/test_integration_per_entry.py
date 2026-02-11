"""Integration tests for per-entry cache and logging."""

import tempfile
from pathlib import Path

from daily_report.core.entry import EntryManager
from daily_report.core.types import AnalysisResult, Article


def test_full_pipeline_with_entry_manager():
    """Test workflow: fetch -> extract -> analyze with entry folders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)

        article = Article(
            title="Test AI Article",
            site="Tech News",
            url="https://example.com/ai-article",
        )
        entry = EntryManager(articles_dir, article)
        entry.ensure_folder()

        html = "<html><body><p>AI is advancing rapidly in 2026.</p></body></html>"
        entry.fetched_html.write_text(html, encoding="utf-8")

        text = "AI is advancing rapidly in 2026. New models are released."
        entry.extracted_txt.write_text(text, encoding="utf-8")

        result = AnalysisResult(
            article=article,
            analysis="AI technology is progressing quickly with frequent model releases.",
            status="ok",
            meta={"model": "test-model", "deep_fetch": False},
        )
        entry.write_analysis_result(result)

        assert entry.fetched_html.exists()
        assert entry.extracted_txt.exists()
        assert entry.analysis_raw.exists()
        assert entry.analysis_meta.exists()

        loaded = entry.read_analysis_result()
        assert loaded is not None
        assert loaded["analysis"] == result.analysis
        assert loaded["status"] == "ok"
        assert loaded["model"] == "test-model"


def test_entry_folder_naming_consistency():
    """Same article always produces same folder name."""
    articles_dir = Path("/tmp/test")

    article = Article(title="OpenAI Codex Launch", site="Tech", url="https://example.com/codex")
    entry1 = EntryManager(articles_dir, article)
    entry2 = EntryManager(articles_dir, article)

    assert entry1.folder.name == entry2.folder.name


def test_multiple_articles_different_folders():
    """Different articles should produce different folder names."""
    articles_dir = Path("/tmp/test")

    article1 = Article(
        title="AI Advances in 2026",
        site="Tech News",
        url="https://example.com/ai-2026",
    )
    article2 = Article(
        title="Quantum Computing Breakthrough",
        site="Science Daily",
        url="https://example.com/quantum",
    )

    entry1 = EntryManager(articles_dir, article1)
    entry2 = EntryManager(articles_dir, article2)

    assert entry1.folder.name != entry2.folder.name
    assert "ai-advances-in-2026" in entry1.folder.name
    assert "quantum-computing-breakthrough" in entry2.folder.name


def test_pipeline_with_analysis_error_status():
    """Analysis artifacts should still be writable on error status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)

        article = Article(title="Test Article", site="Test Site", url="https://example.com/test")
        entry = EntryManager(articles_dir, article)
        entry.ensure_folder()

        entry.fetched_html.write_text("<html><body><p>Content here</p></body></html>", encoding="utf-8")
        entry.extracted_txt.write_text("", encoding="utf-8")

        result = AnalysisResult(
            article=article,
            analysis="Provider error: TimeoutError",
            status="provider_error",
            meta={"model": "test-model", "error": "timeout"},
        )
        entry.write_analysis_result(result)

        loaded = entry.read_analysis_result()
        assert loaded is not None
        assert loaded["status"] == "provider_error"
        assert loaded["error"] == "timeout"


def test_entry_folder_persistence():
    """Entry data should persist across manager instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)

        article = Article(title="Persistent Article", site="Test", url="https://example.com/persistent")

        entry1 = EntryManager(articles_dir, article)
        entry1.ensure_folder()
        result = AnalysisResult(
            article=article,
            analysis="Persistent analysis",
            status="ok",
            meta={"model": "test-model"},
        )
        entry1.write_analysis_result(result)

        entry2 = EntryManager(articles_dir, article)
        loaded = entry2.read_analysis_result()
        assert loaded is not None
        assert loaded["analysis"] == "Persistent analysis"
        assert loaded["status"] == "ok"


def test_article_with_special_characters():
    """Articles with special characters should produce safe folder names."""
    articles_dir = Path("/tmp/test")

    article = Article(
        title="AI & Machine Learning: What's Next? (2026)",
        site="Tech News",
        url="https://example.com/ai-ml-2026",
    )
    entry = EntryManager(articles_dir, article)

    assert "&" not in entry.folder.name
    assert "?" not in entry.folder.name
    assert "(" not in entry.folder.name
    assert ")" not in entry.folder.name
    assert "ai-machine-learning" in entry.folder.name
