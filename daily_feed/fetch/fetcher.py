"""
HTTP content fetching using remote Crawl4AI API.

This module provides fetching via a remote Crawl4AI service which handles
JavaScript rendering and anti-bot detection.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        "urls": [url],  # API expects 'urls' as a list
        "timeout": int(timeout * 1000),  # Convert to milliseconds
        "delay_before_return_html": delay,
        "simulate_user": simulate_user,
        "magic": magic,
        "stealth": stealth,
    }

    if user_agent:
        payload["user_agent"] = user_agent

    last_error: str | None = None

    # Configure separate timeouts: connect is short, read is long (for slow crawls)
    # httpx.Timeout(connect=5, read=60, write=10, pool=10)
    timeout_config = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_config, trust_env=True) as client:
                resp = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=timeout_config,
                )

                if resp.status_code == 200:
                    data = resp.json()

                    # API returns a dict with 'results' array when using 'urls'
                    # Extract the first result from the results array
                    if isinstance(data, dict) and "results" in data:
                        results = data["results"]
                        if isinstance(results, list) and len(results) > 0:
                            data = results[0]  # Get first (and only) result
                        else:
                            last_error = "Crawl4AI API Error: empty results array"
                            if attempt < retries:
                                await asyncio.sleep(0.5 * (attempt + 1))
                                continue
                    # Handle case where API returns a list directly (unlikely)
                    elif isinstance(data, list) and len(data) > 0:
                        data = data[0]
                    elif isinstance(data, list):
                        last_error = "Crawl4AI API Error: empty response list"
                        if attempt < retries:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue

                    # Check for success field in response (after extracting result)
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
                    elif "result" in data:
                        # Some API versions use 'result' field
                        result = data["result"]
                        if isinstance(result, str):
                            text = result
                        elif hasattr(result, "markdown"):
                            text = result.markdown

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
