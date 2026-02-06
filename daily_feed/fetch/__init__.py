"""
Article fetching and extraction.

This package handles HTTP fetching, content extraction,
and caching for article processing.
"""

from .fetcher import fetch_url_crawl4ai_api, FetchResult, cache_path
from .extractor import extract_text
from .cache import CacheIndex

__all__ = [
    "fetch_url_crawl4ai_api",
    "FetchResult",
    "cache_path",
    "extract_text",
    "CacheIndex",
]
