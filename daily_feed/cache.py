from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any



class CacheIndex:
    def __init__(self, cache_dir: Path, enabled: bool = True, filename: str = "index.jsonl"):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.path = cache_dir / filename

    def append(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")
