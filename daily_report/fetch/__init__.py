"""
Article fetching via remote Crawl4AI API.

This package exposes the fetch result type and Crawl4AI
API fetch function used by the pipeline.
"""

from .fetcher import FetchResult, fetch_url_crawl4ai_api

__all__ = [
    "fetch_url_crawl4ai_api",
    "FetchResult",
]
