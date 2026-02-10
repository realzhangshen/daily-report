#!/usr/bin/env python3
"""Rebucket raw exports into daily feeds using a configurable local cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from daily_feed.input.daily_rebucket import (
    build_stats,
    deduplicate_articles,
    list_export_files,
    load_export_articles,
    rebucket_articles,
    resolve_timezone,
    timezone_label,
    write_daily_feeds,
)


def parse_cutoff(value: str) -> tuple[int, int]:
    """Parse HH:MM cutoff string into hour and minute."""
    raw = value.strip()
    try:
        hour_text, minute_text = raw.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--cutoff must use HH:MM format") from exc

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise argparse.ArgumentTypeError("--cutoff must be a valid 24-hour time")
    return hour, minute


def parse_timestamp_fields(value: str) -> tuple[str, ...]:
    """Parse comma-separated timestamp fields."""
    fields = tuple(field.strip() for field in value.split(",") if field.strip())
    if not fields:
        raise argparse.ArgumentTypeError("--timestamp-fields cannot be empty")
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all JSON exports under data/, deduplicate entries, "
            "and rebucket into daily feeds with a local cutoff boundary."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing raw export JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/daily_feeds"),
        help="Directory for generated daily feed JSON files.",
    )
    parser.add_argument(
        "--cutoff",
        type=parse_cutoff,
        default=(22, 0),
        help="Daily cutoff in local time (HH:MM). Entries at/after cutoff go to next day.",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="local",
        help="IANA timezone name, or 'local' for system local timezone.",
    )
    parser.add_argument(
        "--timestamp-fields",
        type=parse_timestamp_fields,
        default=("insertedAt", "publishedAt"),
        help="Priority order for timestamp fields, comma-separated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    cutoff_hour, cutoff_minute = args.cutoff
    timestamp_fields: tuple[str, ...] = args.timestamp_fields

    tz_name = None if args.timezone == "local" else args.timezone
    tz = resolve_timezone(tz_name)

    files = list_export_files(input_dir)
    if not files:
        print(f"No JSON exports found in {input_dir}")
        return 1

    source_articles, source_file_names = load_export_articles(files)
    deduped_articles = deduplicate_articles(source_articles, timestamp_fields=timestamp_fields)
    buckets, missing_timestamp = rebucket_articles(
        deduped_articles,
        tz=tz,
        cutoff_hour=cutoff_hour,
        cutoff_minute=cutoff_minute,
        timestamp_fields=timestamp_fields,
    )
    written_files = write_daily_feeds(
        buckets,
        output_dir=output_dir,
        source_files=source_file_names,
        tz=tz,
        cutoff_hour=cutoff_hour,
        cutoff_minute=cutoff_minute,
    )

    stats = build_stats(
        source_files=source_file_names,
        source_articles=len(source_articles),
        deduped_articles=len(deduped_articles),
        missing_timestamp=missing_timestamp,
        bucket_count=len(buckets),
    )
    bucket_counts = {day.isoformat(): len(day_articles) for day, day_articles in buckets.items()}
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "inputDir": str(input_dir),
        "outputDir": str(output_dir),
        "timezone": timezone_label(tz),
        "cutoffLocalTime": f"{cutoff_hour:02d}:{cutoff_minute:02d}",
        "timestampFields": list(timestamp_fields),
        "stats": {
            "sourceFiles": stats.source_files,
            "sourceArticles": stats.source_articles,
            "dedupedArticles": stats.deduped_articles,
            "missingTimestamp": stats.missing_timestamp,
            "bucketCount": stats.bucket_count,
        },
        "feeds": [
            {
                "date": path.stem.removeprefix("feed-"),
                "file": path.name,
                "total": bucket_counts[path.stem.removeprefix("feed-")],
            }
            for path in written_files
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"Input exports: {stats.source_files}")
    print(f"Source articles: {stats.source_articles}")
    print(f"Deduped articles: {stats.deduped_articles}")
    print(f"Missing timestamp: {stats.missing_timestamp}")
    print(f"Daily feeds written: {len(written_files)}")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
