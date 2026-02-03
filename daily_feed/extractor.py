from __future__ import annotations

from typing import Callable

from bs4 import BeautifulSoup
import trafilatura
from readability import Document


def extract_text(html: str, primary: str, fallback: list[str]) -> str | None:
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
    if name == "trafilatura":
        return _extract_trafilatura
    if name == "readability":
        return _extract_readability
    if name == "bs4":
        return _extract_bs4
    return None


def _extract_trafilatura(html: str) -> str | None:
    return trafilatura.extract(html)


def _extract_readability(html: str) -> str | None:
    doc = Document(html)
    content_html = doc.summary()
    return _extract_bs4(content_html)


def _extract_bs4(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    cleaned = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return cleaned if cleaned else None
