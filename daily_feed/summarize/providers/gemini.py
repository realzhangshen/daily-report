"""
Google Gemini LLM provider implementation.

This module integrates with Google's Generative Language API for
article summarization and topic grouping.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ...config import LoggingConfig, ProviderConfig, SummaryConfig
from ...logging_utils import log_event, redact_text, redact_value, truncate_text
from ..tracing import record_span_error, set_span_output, start_span
from ...core.types import Article, ArticleSummary
from .base import Provider


class GeminiProvider(Provider):
    """Google Gemini API integration for summarization and grouping.

    Uses the Gemini API via REST for:
    1. Single article summarization (bullets + takeaway)
    2. Topic grouping of multiple article summaries
    """

    def __init__(
        self,
        cfg: ProviderConfig,
        summary_cfg: SummaryConfig,
        api_key: str | None,
        log_cfg: LoggingConfig,
        llm_logger,
    ):
        """Initialize the Gemini provider.

        Args:
            cfg: Provider configuration (model, base URL, etc.)
            summary_cfg: Summarization settings (bullet count, max chars)
            api_key: Google API key for authentication
            log_cfg: Logging configuration for LLM response logging
            llm_logger: Logger instance for LLM interactions

        Raises:
            ValueError: If api_key is None
        """
        if not api_key:
            raise ValueError("Missing Google API key")
        self.cfg = cfg
        self.summary_cfg = summary_cfg
        self.api_key = api_key
        self.log_cfg = log_cfg
        self.llm_logger = llm_logger

    def summarize_article(
        self,
        article: Article,
        text: str,
        entry_logger: logging.Logger | None = None,
    ) -> ArticleSummary:
        """Generate a summary for a single article using Gemini.

        Sends the article title, metadata, and content to Gemini with a prompt
        requesting JSON output with bullets and takeaway. Handles various
        error cases gracefully.

        Args:
            article: The article to summarize (contains metadata)
            text: The full text content of the article

        Returns:
            ArticleSummary with bullets, takeaway, and status. Status can be:
            - "ok": Successful summarization
            - "provider_error": HTTP/network error calling the API
            - "parse_error": JSON parsing failed (fallback extraction attempted)
        """
        prompt = _summary_prompt(article, text, self.summary_cfg)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
            },
        }
        with start_span(
            "gemini.summarize",
            kind="llm",
            input_value=prompt,
            attributes={
                "llm.model": self.cfg.model,
                "llm.provider": "gemini",
                "article.title": article.title,
                "article.site": article.site,
            },
        ) as span:
            try:
                data = self._post(payload)
                content = _extract_text(data)
                set_span_output(span, content)
                obj = _parse_json_response(content)
                self._log_llm_response(
                    article=article,
                    event="llm_response",
                    status="ok",
                    content=content,
                    prompt=prompt,
                    logger=entry_logger,
                )
            except httpx.HTTPError as exc:
                record_span_error(span, exc)
                self._log_llm_response(
                    article=article,
                    event="llm_response",
                    status="provider_error",
                    content=str(exc),
                    prompt=prompt,
                    logger=entry_logger,
                )
                return ArticleSummary(
                    article=article,
                    bullets=[f"Provider error: {type(exc).__name__}"],
                    takeaway=article.summary or "",
                    status="provider_error",
                )
            except json.JSONDecodeError as exc:
                record_span_error(span, exc)
                self._log_llm_response(
                    article=article,
                    event="llm_response",
                    status="parse_error",
                    content=content,
                    prompt=prompt,
                    logger=entry_logger,
                )
                fallback = _fallback_from_text(content)
                return ArticleSummary(
                    article=article,
                    bullets=fallback["bullets"],
                    takeaway=fallback["takeaway"],
                    status="parse_error",
                    meta={"raw_response": content},
                )

        bullets = obj.get("bullets") or []
        takeaway = obj.get("takeaway") or ""
        return ArticleSummary(
            article=article,
            bullets=[str(b).strip() for b in bullets if str(b).strip()],
            takeaway=str(takeaway).strip(),
            status="ok",
            meta={"model": self.cfg.model},
        )

    def group_topics(self, summaries: list[ArticleSummary]) -> dict[str, list[ArticleSummary]]:
        """Group articles by topic using Gemini.

        Sends all article summaries to Gemini and asks it to assign
        a concise topic label to each. Returns articles grouped by topic.

        Args:
            summaries: List of ArticleSummary objects to group

        Returns:
            Dictionary mapping topic names to lists of summaries.
            Returns empty dict if grouping fails.
        """
        prompt = _group_prompt(summaries)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1024,
            },
        }
        with start_span(
            "gemini.group_topics",
            kind="llm",
            input_value=prompt,
            attributes={"llm.model": self.cfg.model, "llm.provider": "gemini"},
        ) as span:
            try:
                data = self._post(payload)
                content = _extract_text(data)
                set_span_output(span, content)
                obj = json.loads(content)
                self._log_llm_group_response(content, status="ok", prompt=prompt)
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                record_span_error(span, exc)
                self._log_llm_group_response(
                    content if "content" in locals() else "", status="error", prompt=prompt
                )
                return {}

        # Match returned topics back to summaries by title
        grouped: dict[str, list[ArticleSummary]] = {}
        for item in obj.get("items", []):
            title = item.get("title")
            topic = item.get("topic")
            if not title or not topic:
                continue
            for summary in summaries:
                if summary.article.title == title:
                    summary.topic = topic
                    grouped.setdefault(topic, []).append(summary)
                    break
        return grouped

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request to the Gemini API.

        Args:
            payload: Request body as a dictionary

        Returns:
            Parsed JSON response from the API

        Raises:
            httpx.HTTPError: If the request fails
        """
        url = f"{self.cfg.base_url}/v1beta/models/{self.cfg.model}:generateContent"
        params = {"key": self.api_key}
        with httpx.Client(timeout=30.0, trust_env=self.cfg.trust_env) as client:
            resp = client.post(url, params=params, json=payload)
            resp.raise_for_status()
            return resp.json()

    def _log_llm_response(
        self,
        article: Article,
        event: str,
        status: str,
        content: str,
        prompt: str,
        logger: logging.Logger | None = None,
    ) -> None:
        """Log an LLM summarization response.

        Applies redaction and truncation based on configuration,
        then logs to the LLM-specific log file.

        Args:
            article: The article being summarized
            event: Event type for logging
            status: Status of the request (ok, provider_error, parse_error)
            content: The LLM response content
            prompt: The prompt sent to the LLM
            logger: Logger to use (falls back to self.llm_logger if None)
        """
        active_logger = logger or self.llm_logger
        if active_logger is None:
            return
        detail = self.log_cfg.llm_log_detail
        redaction = self.log_cfg.llm_log_redaction
        payload = {
            "event": event,
            "status": status,
            "model": self.cfg.model,
            "article_title": article.title,
            "article_site": article.site,
            "article_url": redact_value(article.url, redaction),
            "author": redact_value(article.author, redaction),
        }
        if detail == "summary_only":
            payload["raw_response"] = ""
        elif detail == "prompt_response":
            payload["raw_prompt"] = truncate_text(redact_text(prompt, redaction))
            payload["raw_response"] = truncate_text(redact_text(content, redaction))
        else:
            payload["raw_response"] = truncate_text(redact_text(content, redaction))
        log_event(active_logger, "LLM response", **payload)

    def _log_llm_group_response(self, content: str, status: str, prompt: str) -> None:
        """Log an LLM topic grouping response.

        Args:
            content: The LLM response content
            status: Status of the request (ok or error)
            prompt: The prompt sent to the LLM
        """
        if self.llm_logger is None:
            return
        redaction = self.log_cfg.llm_log_redaction
        payload = {
            "event": "llm_group_response",
            "status": status,
            "model": self.cfg.model,
            "raw_response": truncate_text(redact_text(content or "", redaction)),
        }
        if self.log_cfg.llm_log_detail == "prompt_response":
            payload["raw_prompt"] = truncate_text(redact_text(prompt, redaction))
        log_event(self.llm_logger, "LLM group response", **payload)


def _extract_text(data: dict[str, Any]) -> str:
    """Extract text content from Gemini API response.

    Navigates the Gemini response structure to find the actual
    generated text content.

    Args:
        data: Parsed JSON response from Gemini API

    Returns:
        The text content, or empty string if not found
    """
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:  # noqa: BLE001
        return ""


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse JSON from LLM response with fenced code block handling.

    Gemini may return JSON wrapped in markdown code blocks. This
    function handles both direct JSON and fenced formats.

    Args:
        content: The raw text content from the LLM response

    Returns:
        Parsed JSON as a dictionary

    Raises:
        json.JSONDecodeError: If JSON cannot be extracted or parsed
    """
    if not content:
        raise json.JSONDecodeError("Empty content", content or "", 0)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        extracted = _extract_json_snippet(content)
        return json.loads(extracted)


def _extract_json_snippet(content: str) -> str:
    """Extract JSON from text, handling fenced code blocks.

    First tries to extract from a ```json...``` fenced block,
    then falls back to finding the outermost { } braces.

    Args:
        content: Text potentially containing JSON

    Returns:
        Extracted JSON string

    Raises:
        json.JSONDecodeError: If no JSON-like content is found
    """
    fence = _extract_fenced_json(content)
    if fence:
        return fence
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", content, 0)
    return content[start : end + 1]


def _extract_fenced_json(content: str) -> str | None:
    """Extract JSON from a markdown fenced code block.

    Looks for ```json or ``` fences and returns the content
    between them.

    Args:
        content: Text that may contain a fenced JSON block

    Returns:
        The extracted JSON string, or None if no fence found
    """
    lines = content.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("```") and "json" in line.lower():
            start_idx = idx + 1
            break
    if start_idx is None:
        return None
    for idx in range(start_idx, len(lines)):
        if lines[idx].strip().startswith("```"):
            snippet = "\n".join(lines[start_idx:idx]).strip()
            return snippet or None
    return None


def _fallback_from_text(content: str) -> dict[str, Any]:
    """Extract bullets/takeaway from plain text when JSON parsing fails.

    Attempts to extract bullet points from markdown-style lists,
    or falls back to splitting by sentences.

    Args:
        content: Plain text content from the LLM

    Returns:
        Dictionary with "bullets" (list of strings) and "takeaway"
    """
    text = (content or "").strip()
    bullets: list[str] = []
    if text:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("- ") or stripped.startswith("* "):
                bullet = stripped[2:].strip()
                if bullet:
                    bullets.append(bullet)
        if not bullets:
            parts = []
            for chunk in text.replace("\r", "").split("\n"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                pieces = [p.strip() for p in chunk.split(".") if p.strip()]
                parts.extend(pieces)
            bullets = parts[:3]
    takeaway = bullets[0] if bullets else ""
    return {"bullets": bullets, "takeaway": takeaway}


def _summary_prompt(article: Article, text: str, cfg: SummaryConfig) -> str:
    """Build the summarization prompt for the LLM.

    Creates a prompt that instructs the LLM to output JSON with
    bullet points and a takeaway, constrained by configuration.

    Args:
        article: The article to summarize (for metadata)
        text: The article content to summarize
        cfg: Summary configuration (bullet count, max chars)

    Returns:
        The complete prompt string to send to the LLM
    """
    trimmed = text[: cfg.max_chars]
    return (
        "Summarize the article in English. Output strict JSON with keys: "
        f"bullets (array of {cfg.bullets_min}-{cfg.bullets_max} strings), takeaway (string). "
        f"Title: {article.title}\n"
        f"Site: {article.site}\n"
        f"Author: {article.author or ''}\n"
        f"Content:\n{trimmed}"
    )


def _group_prompt(summaries: list[ArticleSummary]) -> str:
    """Build the topic grouping prompt for the LLM.

    Creates a prompt that instructs the LLM to assign concise
    topic labels to each article based on title and takeaway.

    Args:
        summaries: List of article summaries to group

    Returns:
        The complete prompt string to send to the LLM
    """
    lines = []
    for summary in summaries:
        lines.append(f"- {summary.article.title}: {summary.takeaway}")
    joined = "\n".join(lines)
    return (
        "Group the articles into topics. Output strict JSON with key 'items' "
        "which is an array of objects: {title, topic}. Use concise topic labels.\n"
        f"Articles:\n{joined}"
    )
