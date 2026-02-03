"""
HTML content extraction with multiple fallback strategies.

This module provides a chain of extraction methods:
1. trafilatura: Fast, purpose-built for article content (default)
2. readability: Mozilla's readability algorithm (fallback)
3. bs4: BeautifulSoup plain text extraction (last resort)
"""

from __future__ import annotations

from typing import Callable

from bs4 import BeautifulSoup
import trafilatura
from readability import Document


def extract_text(html: str, primary: str, fallback: list[str]) -> str | None:
    """Extract plain text from HTML using a chain of extractors.

    Tries each extraction method in order until one produces non-empty
    output. This provides robustness against different HTML structures.

    Args:
        html: The HTML content to extract text from
        primary: Name of the primary extraction method to try first
        fallback: List of fallback method names to try if primary fails

    Returns:
        Extracted plain text with leading/trailing whitespace stripped,
        or None if all methods fail

    Examples:
        >>> extract_text(html, "trafilatura", ["readability", "bs4"])
        "Article content here..."
    """
    order = [primary] + [name for name in fallback if name != primary]
    for method in order:
        extractor = _get_extractor(method)
        if not extractor:
            continue
        text = extractor(html)
        if text:
            return text.strip()
    return None


def _get_extractor(name: str) -> Callable[[str], str | None] | None:
    """Get the extractor function for a given method name.

    Args:
        name: The name of the extraction method ("trafilatura", "readability", "bs4")

    Returns:
        The corresponding extractor function, or None if name is unrecognized
    """
    if name == "trafilatura":
        return _extract_trafilatura
    if name == "readability":
        return _extract_readability
    if name == "bs4":
        return _extract_bs4
    return None


def _extract_trafilatura(html: str) -> str | None:
    """Extract article content using trafilatura.

    Trafilatura is a fast, purpose-built tool for extracting main content
    from web pages, handling navigation, footers, ads, etc.

    Args:
        html: The HTML content to extract from

    Returns:
        Extracted plain text or None if extraction fails
    """
    return trafilatura.extract(html)


def _extract_readability(html: str) -> str | None:
    """Extract article content using Mozilla's readability algorithm.

    Readability is the same algorithm used in Firefox's Reader View.
    It identifies the main content block and returns simplified HTML.

    Args:
        html: The HTML content to extract from

    Returns:
        Extracted plain text or None if extraction fails
    """
    doc = Document(html)
    content_html = doc.summary()
    # Use bs4 to convert the readability HTML to plain text
    return _extract_bs4(content_html)


def _extract_bs4(html: str) -> str | None:
    """Extract plain text from HTML using BeautifulSoup.

    This is the most basic extraction method - it removes script/style
    tags and gets all text content. Used as a last resort fallback.

    Args:
        html: The HTML content to extract from

    Returns:
        Extracted plain text with non-empty lines only, or None if empty
    """
    soup = BeautifulSoup(html, "html.parser")
    # Remove non-content tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Clean up: remove empty lines and strip whitespace
    cleaned = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return cleaned if cleaned else None
