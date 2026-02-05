"""
HTTP content fetching using remote Crawl4AI API.

This module provides fetching via a remote Crawl4AI service which handles
JavaScript rendering and anti-bot detection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

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


async def fetch_url_crawl4ai_api(
    url: str,
    api_url: str,
    timeout: float,
    retries: int,
    user_agent: str | None = None,
    stealth: bool = True,
    delay: float = 2.0,
    simulate_user: bool = True,
    magic: bool = True,
    auth: tuple[str, str] | None = None,
) -> FetchResult:
    """Fetch a URL using remote Crawl4AI API.

    Makes HTTP requests to a remote Crawl4AI service. This is the only
    supported fetch method - local backends and fallbacks are not available.

    Args:
        url: The URL to fetch
        api_url: The base URL of the remote Crawl4AI API service
        timeout: Page timeout in seconds
        retries: Number of retry attempts after initial failure
        user_agent: Custom user agent string (overrides default)
        stealth: Enable stealth mode to bypass bot detection
        delay: Delay before returning HTML (allows challenges to complete)
        simulate_user: Simulate user behavior for anti-bot
        magic: Enable anti-detection "magic" mode
        auth: Optional (username, password) tuple for HTTP Basic Auth

    Returns:
        FetchResult with markdown text on success or error message on failure
    """
    import json
    import base64

    # Ensure api_url doesn't have trailing slash
    api_url = api_url.rstrip("/")
    endpoint = f"{api_url}/crawl"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Add Basic Auth header if credentials provided
    if auth:
        username, password = auth
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
        headers["Authorization"] = f"Basic {credentials}"

    payload = {
        "url": url,
        "timeout": int(timeout * 1000),  # Convert to milliseconds
        "delay_before_return_html": delay,
        "simulate_user": simulate_user,
        "magic": magic,
        "stealth": stealth,
    }

    if user_agent:
        payload["user_agent"] = user_agent

    last_error: str | None = None

    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )

                if resp.status_code == 200:
                    data = resp.json()

                    # Check for success field in response
                    if not data.get("success", True):
                        error_msg = data.get("error", "Unknown API error")
                        last_error = f"Crawl4AI API Error: {error_msg}"
                        if attempt < retries:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue

                    # Extract markdown from response
                    text = None
                    if "markdown" in data:
                        markdown = data["markdown"]
                        if hasattr(markdown, "raw_markdown"):
                            text = markdown.raw_markdown
                        elif isinstance(markdown, str):
                            text = markdown
                        else:
                            text = str(markdown)
                    elif "html" in data:
                        # Fallback to HTML if markdown not available
                        text = data["html"]

                    if text and text.strip():
                        status_code = data.get("status_code", 200)
                        return FetchResult(url=url, status_code=status_code, text=text, error=None)
                    else:
                        last_error = "Crawl4AI API Error: empty response"
                        if attempt < retries:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue
                else:
                    last_error = f"Crawl4AI API HTTP Error: {resp.status_code} {resp.text}"
                    if attempt < retries:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue

        except httpx.TimeoutException as exc:
            last_error = f"TimeoutError: {exc}"
            if attempt < retries:
                await asyncio.sleep(0.5 * (attempt + 1))
        except json.JSONDecodeError as exc:
            last_error = f"JSONDecodeError: {exc} - Response: {resp.text[:200] if 'resp' in locals() else 'N/A'}"
            if attempt < retries:
                await asyncio.sleep(0.5 * (attempt + 1))
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
