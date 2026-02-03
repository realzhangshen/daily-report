"""
HTTP content fetching with multiple backend support.

This module provides two fetching backends:
1. httpx: Fast synchronous HTTP client (default)
2. crawl4ai: Async client with JavaScript rendering for dynamic content

Both support retry logic, timeout configuration, and environment proxy support.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import httpx
import asyncio


@dataclass
class FetchResult:
    """Result of an HTTP fetch operation.

    Either text will be populated (success) or error will be populated (failure),
    but never both. status_code may be None for network-level failures.

    Attributes:
        url: The URL that was fetched
        status_code: HTTP status code, or None if request failed before getting response
        text: The response body text, or None on error
        error: Error message if fetch failed, None on success
    """
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
    """Fetch a URL using httpx with retry logic.

    Uses a synchronous HTTP client that follows redirects and respects
    system proxy settings when trust_env is enabled.

    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds
        retries: Number of retry attempts after initial failure
        user_agent: User-Agent header string
        trust_env: Whether to respect system proxy settings from environment

    Returns:
        FetchResult with text on success or error message on failure
    """
    headers = {"User-Agent": user_agent}
    last_error: str | None = None

    # Attempt the request with exponential backoff between retries
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
                # Exponential backoff: 0.5s, 1.0s, 1.5s...
                time.sleep(0.5 * (attempt + 1))

    return FetchResult(url=url, status_code=None, text=None, error=last_error)


async def fetch_url_crawl4ai(
    url: str,
    timeout: float,
    retries: int,
) -> FetchResult:
    """Fetch a URL using Crawl4AI with JavaScript rendering.

    Uses AsyncWebCrawler (Playwright backend) to render JavaScript-heavy
    pages that don't work well with simple HTTP clients. Returns the
    page content as markdown.

    Args:
        url: The URL to fetch
        timeout: Page timeout in seconds (converted to ms for Crawl4AI)
        retries: Number of retry attempts after initial failure

    Returns:
        FetchResult with markdown text on success or error message on failure
    """
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

            # Check if crawl succeeded
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
            # Extract markdown text from various return formats
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
    """Generate a cache file path for a URL using SHA256 hashing.

    The URL is hashed to create a unique filename that's safe for all
    filesystems. This allows caching content without filename issues
    from special characters in URLs.

    Args:
        cache_dir: The directory where cache files are stored
        url: The URL being cached
        suffix: File extension for the cache (e.g., "html", "txt")

    Returns:
        Path object for the cache file

    Example:
        >>> cache_path(Path("/cache"), "https://example.com", "txt")
        Path('/cache/a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146.txt')
    """
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.{suffix}"
