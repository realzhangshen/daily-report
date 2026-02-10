"""Utilities for rebucketing Folo exports into daily feeds by cutoff time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
import json

from zoneinfo import ZoneInfo

DEFAULT_TIMESTAMP_FIELDS = ("insertedAt", "publishedAt")


@dataclass
class RebucketStats:
    """Stats emitted by the rebucket workflow."""

    source_files: int
    source_articles: int
    deduped_articles: int
    missing_timestamp: int
    bucket_count: int


def parse_iso8601(value: str) -> datetime:
    """Parse an ISO 8601 timestamp string into a timezone-aware datetime."""
    raw = value.strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def resolve_timezone(timezone_name: str | None) -> tzinfo:
    """Resolve local/system timezone or a specific IANA timezone name."""
    if timezone_name:
        return ZoneInfo(timezone_name)

    local = datetime.now().astimezone().tzinfo
    if local is None:
        return timezone.utc
    return local


def timezone_label(tz: tzinfo) -> str:
    """Return a human-readable timezone label for output metadata."""
    zone_key = getattr(tz, "key", None)
    if zone_key:
        return str(zone_key)

    now = datetime.now(tz)
    name = now.tzname()
    if name:
        return str(name)
    return str(tz)


def list_export_files(input_dir: Path) -> list[Path]:
    """List top-level JSON export files from an input directory."""
    return sorted(path for path in input_dir.glob("*.json") if path.is_file())


def load_export_articles(files: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """Load article dictionaries from all export files."""
    articles: list[dict[str, Any]] = []
    source_names = [path.name for path in files]

    for path in files:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)

        export_time = payload.get("exportTime")
        raw_articles = payload.get("articles", [])
        if not isinstance(raw_articles, list):
            continue

        for raw_article in raw_articles:
            if not isinstance(raw_article, dict):
                continue
            article = dict(raw_article)
            article["_source_file"] = path.name
            article["_source_export_time"] = export_time
            articles.append(article)

    return articles, source_names


def _select_timestamp(
    article: dict[str, Any], timestamp_fields: tuple[str, ...]
) -> datetime | None:
    for field in timestamp_fields:
        value = article.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            return parse_iso8601(value)
        except ValueError:
            continue
    return None


def _article_recency(article: dict[str, Any], timestamp_fields: tuple[str, ...]) -> datetime:
    ts = _select_timestamp(article, timestamp_fields)
    if ts is not None:
        return ts.astimezone(timezone.utc)

    export_time = article.get("_source_export_time")
    if isinstance(export_time, str) and export_time.strip():
        try:
            return parse_iso8601(export_time).astimezone(timezone.utc)
        except ValueError:
            pass

    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _is_newer(
    candidate: dict[str, Any],
    existing: dict[str, Any],
    timestamp_fields: tuple[str, ...],
) -> bool:
    candidate_time = _article_recency(candidate, timestamp_fields)
    existing_time = _article_recency(existing, timestamp_fields)
    if candidate_time != existing_time:
        return candidate_time > existing_time

    candidate_file = str(candidate.get("_source_file", ""))
    existing_file = str(existing.get("_source_file", ""))
    return candidate_file > existing_file


def deduplicate_articles(
    articles: list[dict[str, Any]],
    timestamp_fields: tuple[str, ...] = DEFAULT_TIMESTAMP_FIELDS,
) -> list[dict[str, Any]]:
    """Deduplicate across exports, preferring newer copies for collisions."""
    by_id: dict[str, dict[str, Any]] = {}
    no_id_articles: list[dict[str, Any]] = []

    for article in articles:
        article_id = article.get("id")
        if isinstance(article_id, str) and article_id.strip():
            existing = by_id.get(article_id)
            if existing is None or _is_newer(article, existing, timestamp_fields):
                by_id[article_id] = article
            continue
        no_id_articles.append(article)

    merged = list(by_id.values()) + no_id_articles

    by_url_index: dict[str, int] = {}
    deduped: list[dict[str, Any]] = []
    for article in merged:
        url = article.get("url")
        if not isinstance(url, str) or not url.strip():
            continue

        existing_index = by_url_index.get(url)
        if existing_index is None:
            by_url_index[url] = len(deduped)
            deduped.append(article)
            continue

        if _is_newer(article, deduped[existing_index], timestamp_fields):
            deduped[existing_index] = article

    return deduped


def bucket_date_for_timestamp(
    timestamp_utc: datetime,
    tz: tzinfo,
    cutoff_hour: int = 22,
    cutoff_minute: int = 0,
) -> date:
    """Convert UTC timestamp to local feed date using a local cutoff boundary."""
    local_dt = timestamp_utc.astimezone(tz)
    day = local_dt.date()

    local_cutoff = time(hour=cutoff_hour, minute=cutoff_minute)
    current_local = local_dt.timetz().replace(tzinfo=None)
    if current_local >= local_cutoff:
        return day + timedelta(days=1)
    return day


def rebucket_articles(
    articles: list[dict[str, Any]],
    tz: tzinfo,
    cutoff_hour: int = 22,
    cutoff_minute: int = 0,
    timestamp_fields: tuple[str, ...] = DEFAULT_TIMESTAMP_FIELDS,
) -> tuple[dict[date, list[dict[str, Any]]], int]:
    """Group article records into feed dates by local cutoff boundary."""
    bucket_items: dict[date, list[tuple[datetime, dict[str, Any]]]] = {}
    missing_timestamp = 0

    for article in articles:
        timestamp = _select_timestamp(article, timestamp_fields)
        if timestamp is None:
            missing_timestamp += 1
            continue

        feed_date = bucket_date_for_timestamp(timestamp, tz, cutoff_hour, cutoff_minute)
        clean_article = {k: v for k, v in article.items() if not k.startswith("_")}
        bucket_items.setdefault(feed_date, []).append((timestamp, clean_article))

    buckets: dict[date, list[dict[str, Any]]] = {}
    for feed_date, items in bucket_items.items():
        sorted_items = sorted(items, key=lambda item: item[0], reverse=True)
        buckets[feed_date] = [article for _, article in sorted_items]

    return buckets, missing_timestamp


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _window_bounds(
    feed_date: date,
    tz: tzinfo,
    cutoff_hour: int,
    cutoff_minute: int,
) -> tuple[str, str]:
    start = datetime.combine(
        feed_date - timedelta(days=1),
        time(hour=cutoff_hour, minute=cutoff_minute),
        tzinfo=tz,
    )
    end = start + timedelta(days=1) - timedelta(milliseconds=1)
    return start.isoformat(), end.isoformat()


def write_daily_feeds(
    buckets: dict[date, list[dict[str, Any]]],
    output_dir: Path,
    source_files: list[str],
    tz: tzinfo,
    cutoff_hour: int = 22,
    cutoff_minute: int = 0,
) -> list[Path]:
    """Write rebucketed daily feeds to disk as feed-YYYY-MM-DD.json files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for existing in output_dir.glob("feed-*.json"):
        existing.unlink()

    written: list[Path] = []
    generated_at = _utc_now_z()
    tz_name = timezone_label(tz)
    cutoff_text = f"{cutoff_hour:02d}:{cutoff_minute:02d}"

    for feed_date in sorted(buckets):
        articles = buckets[feed_date]
        window_start, window_end = _window_bounds(feed_date, tz, cutoff_hour, cutoff_minute)
        payload = {
            "exportTime": generated_at,
            "total": len(articles),
            "date": feed_date.isoformat(),
            "timezone": tz_name,
            "cutoffLocalTime": cutoff_text,
            "windowStart": window_start,
            "windowEnd": window_end,
            "sourceFiles": source_files,
            "articles": articles,
        }
        path = output_dir / f"feed-{feed_date.isoformat()}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        written.append(path)

    return written


def build_stats(
    source_files: list[str],
    source_articles: int,
    deduped_articles: int,
    missing_timestamp: int,
    bucket_count: int,
) -> RebucketStats:
    """Create a reusable stats payload for reporting and manifest writing."""
    return RebucketStats(
        source_files=len(source_files),
        source_articles=source_articles,
        deduped_articles=deduped_articles,
        missing_timestamp=missing_timestamp,
        bucket_count=bucket_count,
    )

