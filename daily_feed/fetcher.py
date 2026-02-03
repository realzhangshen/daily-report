from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import httpx
import asyncio


@dataclass
class FetchResult:
    url: str
    status_code: int | None
    text: str | None
    error: str | None


def fetch_url(
    url: str,
    timeout: float,
    retries: int,
    user_agent: str,
    trust_env: bool,
) -> FetchResult:
    headers = {"User-Agent": user_agent}
    last_error: str | None = None

    for attempt in range(retries + 1):
        try:
            with httpx.Client(
                timeout=timeout,
                headers=headers,
                follow_redirects=True,
                trust_env=trust_env,
            ) as client:
                resp = client.get(url)
                return FetchResult(url=url, status_code=resp.status_code, text=resp.text, error=None)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))

    return FetchResult(url=url, status_code=None, text=None, error=last_error)


async def fetch_url_crawl4ai(
    url: str,
    timeout: float,
    retries: int,
) -> FetchResult:
    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    except Exception as exc:  # noqa: BLE001
        return FetchResult(url=url, status_code=None, text=None, error=f"ImportError: {exc}")

    last_error: str | None = None

    for attempt in range(retries + 1):
        try:
            run_cfg = CrawlerRunConfig(page_timeout=int(timeout * 1000))
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=run_cfg)

            success = getattr(result, "success", True)
            if not success:
                status_code = getattr(result, "status_code", None)
                error_message = getattr(result, "error_message", None) or "Crawl failed"
                last_error = f"Crawl4AIError: {error_message}"
                if attempt < retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                continue

            status_code = getattr(result, "status_code", None)
            markdown = getattr(result, "markdown", None)
            if markdown is None:
                text = None
            elif hasattr(markdown, "raw_markdown"):
                text = markdown.raw_markdown
            elif isinstance(markdown, str):
                text = markdown
            else:
                text = str(markdown)

            if not text or not text.strip():
                last_error = "Crawl4AIError: empty markdown"
                if attempt < retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                continue

            return FetchResult(url=url, status_code=status_code, text=text, error=None)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await asyncio.sleep(0.5 * (attempt + 1))

    return FetchResult(url=url, status_code=None, text=None, error=last_error)


def cache_path(cache_dir: Path, url: str, suffix: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.{suffix}"
