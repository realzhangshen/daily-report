"""Google Gemini provider for entry-level analysis workflow."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ...config import LoggingConfig, ProviderConfig, SummaryConfig
from ...core.types import Article
from ...utils.logging import log_event, redact_text, redact_value, truncate_text
from ..prompts import (
    build_analysis_prompt,
    build_deep_fetch_prompt,
    build_extraction_prompt,
    build_synthesis_prompt,
)
from ..tracing import record_span_error, set_span_output, start_span
from .base import AnalysisProvider


_DEEP_FETCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "need_deep_fetch": {"type": "BOOLEAN"},
        "urls": {"type": "ARRAY", "items": {"type": "STRING"}},
        "rationale": {"type": "STRING"},
    },
    "required": ["need_deep_fetch", "urls", "rationale"],
}


_EXTRACTION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "one_line_summary": {"type": "STRING"},
        "category": {"type": "STRING"},
        "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
        "importance": {"type": "INTEGER"},
        "content_type": {"type": "STRING"},
        "key_takeaway": {"type": "STRING"},
    },
    "required": [
        "one_line_summary",
        "category",
        "tags",
        "importance",
        "content_type",
        "key_takeaway",
    ],
}


class GeminiProvider(AnalysisProvider):
    """Gemini-backed provider for deep-fetch decision and article analysis."""

    def __init__(
        self,
        cfg: ProviderConfig,
        summary_cfg: SummaryConfig,
        api_key: str | None,
        log_cfg: LoggingConfig,
        llm_logger,
    ):
        if not api_key:
            raise ValueError("Missing Google API key")
        self.cfg = cfg
        self.summary_cfg = summary_cfg
        self.api_key = api_key
        self.log_cfg = log_cfg
        self.llm_logger = llm_logger

    def decide_deep_fetch(
        self,
        article: Article,
        text: str,
        candidate_links: list[str],
        entry_logger: logging.Logger | None = None,
    ) -> dict[str, Any]:
        prompt = build_deep_fetch_prompt(article, text, candidate_links, self.summary_cfg)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 512,
                "responseMimeType": "application/json",
                "responseSchema": _DEEP_FETCH_RESPONSE_SCHEMA,
            },
        }
        with start_span(
            "gemini.decide_deep_fetch",
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
                    event="llm_decide_deep_fetch",
                    status="ok",
                    content=content,
                    prompt=prompt,
                    logger=entry_logger,
                )
            except httpx.HTTPError as exc:
                record_span_error(span, exc)
                self._log_llm_response(
                    article=article,
                    event="llm_decide_deep_fetch",
                    status="provider_error",
                    content=str(exc),
                    prompt=prompt,
                    logger=entry_logger,
                )
                return {"need_deep_fetch": False, "urls": [], "rationale": "provider_error"}
            except json.JSONDecodeError as exc:
                record_span_error(span, exc)
                self._log_llm_response(
                    article=article,
                    event="llm_decide_deep_fetch",
                    status="parse_error",
                    content=content,
                    prompt=prompt,
                    logger=entry_logger,
                )
                return {"need_deep_fetch": False, "urls": [], "rationale": "parse_error"}

        need = bool(obj.get("need_deep_fetch"))
        urls = obj.get("urls") or []
        if not isinstance(urls, list):
            urls = []
        urls = [str(u).strip() for u in urls if str(u).strip()]
        rationale = str(obj.get("rationale") or "").strip()
        return {"need_deep_fetch": need, "urls": urls, "rationale": rationale}

    def analyze_entry(
        self,
        article: Article,
        base_text: str,
        deep_texts: list[str],
        decision: dict[str, Any],
        entry_logger: logging.Logger | None = None,
    ) -> str:
        prompt = build_analysis_prompt(article, base_text, deep_texts, decision, self.summary_cfg)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": self.summary_cfg.analysis_max_output_tokens,
            },
        }
        with start_span(
            "gemini.analyze_entry",
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
                self._log_llm_response(
                    article=article,
                    event="llm_analysis_response",
                    status="ok",
                    content=content,
                    prompt=prompt,
                    logger=entry_logger,
                )
                return content.strip()
            except httpx.HTTPError as exc:
                record_span_error(span, exc)
                self._log_llm_response(
                    article=article,
                    event="llm_analysis_response",
                    status="provider_error",
                    content=str(exc),
                    prompt=prompt,
                    logger=entry_logger,
                )
                return f"Provider error: {type(exc).__name__}"

    def extract_entry(
        self,
        article: Article,
        base_text: str,
        entry_logger: logging.Logger | None = None,
    ) -> dict[str, Any]:
        prompt = build_extraction_prompt(article, base_text, self.summary_cfg)
        payload = self._build_extraction_payload(
            prompt=prompt,
            temperature=0.1,
            max_output_tokens=max(self.summary_cfg.extraction_max_output_tokens, 2048),
        )
        with start_span(
            "gemini.extract_entry",
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
                data = self._post(payload, timeout=60.0)
                content = _extract_text(data)
                set_span_output(span, content)
                obj = _parse_json_response(content)
                self._log_llm_response(
                    article=article,
                    event="llm_extract_entry",
                    status="ok",
                    content=content,
                    prompt=prompt,
                    logger=entry_logger,
                )
                return obj
            except httpx.HTTPError as exc:
                record_span_error(span, exc)
                self._log_llm_response(
                    article=article,
                    event="llm_extract_entry",
                    status="provider_error",
                    content=str(exc),
                    prompt=prompt,
                    logger=entry_logger,
                )
                return {
                    "one_line_summary": article.title,
                    "category": "other",
                    "tags": [],
                    "importance": 3,
                    "content_type": "news",
                    "key_takeaway": "",
                    "status": "provider_error",
                    "error": "provider_error",
                }
            except json.JSONDecodeError as exc:
                retry_payload = self._build_extraction_payload(
                    prompt=prompt,
                    temperature=0.0,
                    max_output_tokens=max(self.summary_cfg.extraction_max_output_tokens, 3072),
                )
                retry_content = content
                try:
                    retry_data = self._post(retry_payload, timeout=90.0)
                    retry_content = _extract_text(retry_data)
                    set_span_output(span, retry_content)
                    obj = _parse_json_response(retry_content)
                    self._log_llm_response(
                        article=article,
                        event="llm_extract_entry",
                        status="ok_retry",
                        content=retry_content,
                        prompt=prompt,
                        logger=entry_logger,
                    )
                    return obj
                except httpx.HTTPError as retry_exc:
                    record_span_error(span, retry_exc)
                    self._log_llm_response(
                        article=article,
                        event="llm_extract_entry",
                        status="provider_error",
                        content=str(retry_exc),
                        prompt=prompt,
                        logger=entry_logger,
                    )
                    return {
                        "one_line_summary": article.title,
                        "category": "other",
                        "tags": [],
                        "importance": 3,
                        "content_type": "news",
                        "key_takeaway": "",
                        "status": "provider_error",
                        "error": "provider_error",
                    }
                except json.JSONDecodeError as retry_parse_exc:
                    record_span_error(span, retry_parse_exc)
                    self._log_llm_response(
                        article=article,
                        event="llm_extract_entry",
                        status="parse_error",
                        content=retry_content,
                        prompt=prompt,
                        logger=entry_logger,
                    )
                    return {
                        "one_line_summary": article.title,
                        "category": "other",
                        "tags": [],
                        "importance": 3,
                        "content_type": "news",
                        "key_takeaway": "",
                        "status": "parse_error",
                        "error": "parse_error",
                    }

    def synthesize(
        self,
        extractions: list[dict[str, Any]],
        logger: logging.Logger | None = None,
    ) -> str:
        prompt = build_synthesis_prompt(extractions, self.summary_cfg)
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": self.summary_cfg.synthesis_max_output_tokens,
            },
        }
        with start_span(
            "gemini.synthesize",
            kind="llm",
            input_value=prompt,
            attributes={
                "llm.model": self.cfg.model,
                "llm.provider": "gemini",
                "extraction.count": len(extractions),
            },
        ) as span:
            try:
                data = self._post(payload, timeout=120.0)
                content = _extract_text(data)
                set_span_output(span, content)
                if logger:
                    log_event(
                        logger,
                        "LLM response",
                        event="llm_synthesize",
                        status="ok",
                        model=self.cfg.model,
                        extraction_count=len(extractions),
                    )
                return content.strip()
            except httpx.HTTPError as exc:
                record_span_error(span, exc)
                if logger:
                    log_event(
                        logger,
                        "LLM response",
                        event="llm_synthesize",
                        status="provider_error",
                        model=self.cfg.model,
                        error=str(exc),
                    )
                return f"Provider error: {type(exc).__name__}"

    def _post(self, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        url = f"{self.cfg.base_url}/v1beta/models/{self.cfg.model}:generateContent"
        params = {"key": self.api_key}
        with httpx.Client(timeout=timeout, trust_env=self.cfg.trust_env) as client:
            resp = client.post(url, params=params, json=payload)
            resp.raise_for_status()
            return resp.json()

    def _build_extraction_payload(
        self,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": _EXTRACTION_RESPONSE_SCHEMA,
            },
        }

    def _log_llm_response(
        self,
        article: Article,
        event: str,
        status: str,
        content: str,
        prompt: str,
        logger: logging.Logger | None = None,
    ) -> None:
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
        if detail == "prompt_response":
            payload["raw_prompt"] = truncate_text(redact_text(prompt, redaction))
            payload["raw_response"] = truncate_text(redact_text(content, redaction))
        else:
            payload["raw_response"] = truncate_text(redact_text(content, redaction))
        log_event(active_logger, "LLM response", **payload)


def _extract_text(data: dict[str, Any]) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except Exception:  # noqa: BLE001
        return ""

    if not isinstance(parts, list):
        return ""

    non_thought_chunks: list[str] = []
    all_chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if text is None:
            continue
        chunk = str(text)
        if not chunk:
            continue
        all_chunks.append(chunk)
        if not bool(part.get("thought")):
            non_thought_chunks.append(chunk)

    if non_thought_chunks:
        return "".join(non_thought_chunks)
    return "".join(all_chunks)


def _parse_json_response(content: str) -> dict[str, Any]:
    if not content:
        raise json.JSONDecodeError("Empty content", content or "", 0)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        extracted = _extract_json_snippet(content)
        return json.loads(extracted)


def _extract_json_snippet(content: str) -> str:
    fence = _extract_fenced_json(content)
    if fence:
        return fence
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", content, 0)
    return content[start : end + 1]


def _extract_fenced_json(content: str) -> str | None:
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
