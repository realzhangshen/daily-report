"""Tests for daily export rebucket utilities."""

from datetime import date, timezone

from daily_feed.input.daily_rebucket import (
    bucket_date_for_timestamp,
    deduplicate_articles,
    parse_iso8601,
    rebucket_articles,
)


def test_bucket_date_boundary_at_cutoff():
    """Entries at/after cutoff should belong to the next feed date."""
    tz = timezone.utc

    before_cutoff = parse_iso8601("2026-02-08T21:59:59Z")
    at_cutoff = parse_iso8601("2026-02-08T22:00:00Z")

    assert bucket_date_for_timestamp(before_cutoff, tz, cutoff_hour=22, cutoff_minute=0) == date(
        2026, 2, 8
    )
    assert bucket_date_for_timestamp(at_cutoff, tz, cutoff_hour=22, cutoff_minute=0) == date(
        2026, 2, 9
    )


def test_deduplicate_articles_prefers_newer_by_id():
    """ID collisions should keep the newer record."""
    articles = [
        {
            "id": "same-id",
            "url": "https://example.com/item",
            "insertedAt": "2026-02-08T10:00:00Z",
            "title": "old",
        },
        {
            "id": "same-id",
            "url": "https://example.com/item",
            "insertedAt": "2026-02-08T10:30:00Z",
            "title": "new",
        },
    ]

    deduped = deduplicate_articles(articles)

    assert len(deduped) == 1
    assert deduped[0]["title"] == "new"


def test_deduplicate_articles_collapses_same_url_after_id_dedup():
    """Different IDs with same URL should still collapse to one record."""
    articles = [
        {
            "id": "a",
            "url": "https://example.com/shared",
            "insertedAt": "2026-02-08T10:00:00Z",
            "title": "first",
        },
        {
            "id": "b",
            "url": "https://example.com/shared",
            "insertedAt": "2026-02-08T10:05:00Z",
            "title": "second",
        },
    ]

    deduped = deduplicate_articles(articles)

    assert len(deduped) == 1
    assert deduped[0]["id"] == "b"


def test_rebucket_uses_timestamp_priority_and_strips_internal_fields():
    """Rebucket should use provided timestamp priority and remove internal keys."""
    articles = [
        {
            "id": "late",
            "url": "https://example.com/late",
            "insertedAt": "2026-02-08T22:30:00Z",
            "publishedAt": "2026-02-08T08:00:00Z",
            "_source_file": "export-1.json",
        },
        {
            "id": "fallback",
            "url": "https://example.com/fallback",
            "publishedAt": "2026-02-08T21:00:00Z",
            "_source_file": "export-1.json",
        },
    ]

    buckets, missing = rebucket_articles(
        articles,
        tz=timezone.utc,
        cutoff_hour=22,
        cutoff_minute=0,
        timestamp_fields=("insertedAt", "publishedAt"),
    )

    assert missing == 0
    assert set(buckets.keys()) == {date(2026, 2, 8), date(2026, 2, 9)}
    for day_articles in buckets.values():
        assert all("_source_file" not in article for article in day_articles)

