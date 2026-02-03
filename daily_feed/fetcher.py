from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import httpx


@dataclass
class FetchResult:
    url: str
    status_code: int | None
    text: str | None
    error: str | None


def fetch_url(url: str, timeout: float, retries: int, user_agent: str) -> FetchResult:
    headers = {"User-Agent": user_agent}
    last_error: str | None = None

    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
                resp = client.get(url)
                return FetchResult(url=url, status_code=resp.status_code, text=resp.text, error=None)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))

    return FetchResult(url=url, status_code=None, text=None, error=last_error)


def cache_path(cache_dir: Path, url: str, suffix: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.{suffix}"
