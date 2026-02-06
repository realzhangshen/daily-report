"""Integration tests for per-entry cache and logging."""

import tempfile
from pathlib import Path

from daily_feed.core.entry import EntryManager
from daily_feed.core.types import Article, ArticleSummary


def test_full_pipeline_with_entry_manager():
    """Test full workflow: fetch -> extract -> summarize with entry folders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)

        # Create article
        article = Article(
            title="Test AI Article",
            site="Tech News",
            url="https://example.com/ai-article"
        )
        entry = EntryManager(articles_dir, article)
        entry.ensure_folder()

        # Simulate fetch - write HTML
        html = "<html><body><p>AI is advancing rapidly in 2026.</p></body></html>"
        entry.fetched_html.write_text(html, encoding="utf-8")

        # Simulate extract - write text
        text = "AI is advancing rapidly in 2026. New models are released."
        entry.extracted_txt.write_text(text, encoding="utf-8")

        # Simulate LLM summary
        summary = ArticleSummary(
            article=article,
            bullets=["AI models advancing in 2026", "New releases happening"],
            takeaway="AI technology is progressing quickly.",
            topic="AI",
            status="ok"
        )
        summary.meta = {"model": "test-model", "generated_at": "2026-02-05T00:00:00Z"}
        entry.write_llm_summary(summary)

        # Verify all files exist
        assert entry.fetched_html.exists()
        assert entry.extracted_txt.exists()
        assert entry.llm_summary.exists()

        # Verify summary content
        loaded = entry.read_llm_summary()
        assert loaded["bullets"] == ["AI models advancing in 2026", "New releases happening"]
        assert loaded["takeaway"] == "AI technology is progressing quickly."
        assert loaded["topic"] == "AI"


def test_entry_folder_naming_consistency():
    """Same article always produces same folder name."""
    articles_dir = Path("/tmp/test")

    article = Article(
        title="OpenAI Codex Launch",
        site="Tech",
        url="https://example.com/codex"
    )

    entry1 = EntryManager(articles_dir, article)
    entry2 = EntryManager(articles_dir, article)

    assert entry1.folder.name == entry2.folder.name


def test_multiple_articles_different_folders():
    """Different articles should produce different folder names."""
    articles_dir = Path("/tmp/test")

    article1 = Article(
        title="AI Advances in 2026",
        site="Tech News",
        url="https://example.com/ai-2026"
    )

    article2 = Article(
        title="Quantum Computing Breakthrough",
        site="Science Daily",
        url="https://example.com/quantum"
    )

    entry1 = EntryManager(articles_dir, article1)
    entry2 = EntryManager(articles_dir, article2)

    # Different articles should have different folder names
    assert entry1.folder.name != entry2.folder.name

    # Folder names should contain the slug
    assert "ai-advances-in-2026" in entry1.folder.name
    assert "quantum-computing-breakthrough" in entry2.folder.name


def test_full_pipeline_with_error_handling():
    """Test pipeline with error scenarios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)

        article = Article(
            title="Test Article",
            site="Test Site",
            url="https://example.com/test"
        )
        entry = EntryManager(articles_dir, article)
        entry.ensure_folder()

        # Simulate fetch - write HTML
        html = "<html><body><p>Content here</p></body></html>"
        entry.fetched_html.write_text(html, encoding="utf-8")

        # Simulate extraction failure
        entry.extracted_txt.write_text("", encoding="utf-8")

        # Verify files exist
        assert entry.fetched_html.exists()
        assert entry.extracted_txt.exists()

        # Summary should still be writable even with empty extraction
        summary = ArticleSummary(
            article=article,
            bullets=["Error extracting content"],
            takeaway="Could not extract article content.",
            topic="Error",
            status="provider_error"
        )
        summary.meta = {"model": "test-model", "error": "extraction_failed"}
        entry.write_llm_summary(summary)

        # Verify summary was written
        loaded = entry.read_llm_summary()
        assert loaded["status"] == "provider_error"
        assert loaded["topic"] == "Error"


def test_entry_folder_persistence():
    """Test that entry data persists across manager instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        articles_dir = Path(tmpdir)

        article = Article(
            title="Persistent Article",
            site="Test",
            url="https://example.com/persistent"
        )

        # First manager - write data
        entry1 = EntryManager(articles_dir, article)
        entry1.ensure_folder()

        summary = ArticleSummary(
            article=article,
            bullets=["Point 1", "Point 2"],
            takeaway="Test takeaway",
            topic="Test",
            status="ok"
        )
        summary.meta = {"model": "test-model"}
        entry1.write_llm_summary(summary)

        # Second manager - read data (same folder)
        entry2 = EntryManager(articles_dir, article)

        # Should read the same data
        loaded = entry2.read_llm_summary()
        assert loaded is not None
        assert loaded["bullets"] == ["Point 1", "Point 2"]
        assert loaded["takeaway"] == "Test takeaway"


def test_article_with_special_characters():
    """Test that articles with special characters are handled correctly."""
    articles_dir = Path("/tmp/test")

    article = Article(
        title="AI & Machine Learning: What's Next? (2026)",
        site="Tech News",
        url="https://example.com/ai-ml-2026"
    )

    entry = EntryManager(articles_dir, article)

    # Should sanitize special characters
    assert "&" not in entry.folder.name
    assert "?" not in entry.folder.name
    assert "(" not in entry.folder.name
    assert ")" not in entry.folder.name
    assert "ai-machine-learning" in entry.folder.name
