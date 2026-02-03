"""
Cache index tracking for fetched content.

This module provides JSONL-based index logging for cache operations,
enabling debugging and analysis of cache hits/misses.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CacheIndex:
    """Tracks cache operations in a JSONL index file.

    Each cache operation (read, write, source) is logged as a JSON line
    with timestamp, URL, file path, status code, and error info.

    Attributes:
        cache_dir: Directory where cache files and index are stored
        enabled: Whether index writing is enabled
        path: Full path to the index file
    """

    def __init__(self, cache_dir: Path, enabled: bool = True, filename: str = "index.jsonl"):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.path = cache_dir / filename

    def append(self, payload: dict[str, Any]) -> None:
        """Append an entry to the cache index.

        Adds a timestamp if not present and writes the entry as a JSON line.

        Args:
            payload: Dictionary containing cache operation details including
                     url, hash, kind, path, source, status_code, error, etc.
        """
        if not self.enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")
