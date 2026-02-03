from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import LoggingConfig, ProviderConfig, SummaryConfig
from ..logging_utils import log_event, redact_text, redact_value, truncate_text
from ..types import Article, ArticleSummary
from .base import Provider


class GeminiProvider(Provider):
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

    def summarize_article(self, article: Article, text: str) -> ArticleSummary:
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
        try:
            data = self._post(payload)
            content = _extract_text(data)
            obj = _parse_json_response(content)
            self._log_llm_response(
                article=article,
                event="llm_response",
                status="ok",
                content=content,
                prompt=prompt,
            )
        except httpx.HTTPError as exc:
            self._log_llm_response(
                article=article,
                event="llm_response",
                status="provider_error",
                content=str(exc),
                prompt=prompt,
            )
            return ArticleSummary(
                article=article,
                bullets=[f"Provider error: {type(exc).__name__}"],
                takeaway=article.summary or "",
                status="provider_error",
            )
        except json.JSONDecodeError:
            self._log_llm_response(
                article=article,
                event="llm_response",
                status="parse_error",
                content=content,
                prompt=prompt,
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
        try:
            data = self._post(payload)
            content = _extract_text(data)
            obj = json.loads(content)
            self._log_llm_group_response(content, status="ok", prompt=prompt)
        except (httpx.HTTPError, json.JSONDecodeError):
            self._log_llm_group_response(
                content if "content" in locals() else "", status="error", prompt=prompt
            )
            return {}

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
    ) -> None:
        if self.llm_logger is None:
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
        log_event(self.llm_logger, "LLM response", **payload)

    def _log_llm_group_response(self, content: str, status: str, prompt: str) -> None:
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
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:  # noqa: BLE001
        return ""


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


def _fallback_from_text(content: str) -> dict[str, Any]:
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
    lines = []
    for summary in summaries:
        lines.append(f"- {summary.article.title}: {summary.takeaway}")
    joined = "\n".join(lines)
    return (
        "Group the articles into topics. Output strict JSON with key 'items' "
        "which is an array of objects: {title, topic}. Use concise topic labels.\n"
        f"Articles:\n{joined}"
    )
