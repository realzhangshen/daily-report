from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import ProviderConfig, SummaryConfig
from ..types import Article, ArticleSummary
from .base import Provider


class GeminiProvider(Provider):
    def __init__(self, cfg: ProviderConfig, summary_cfg: SummaryConfig, api_key: str | None):
        if not api_key:
            raise ValueError("Missing Google API key")
        self.cfg = cfg
        self.summary_cfg = summary_cfg
        self.api_key = api_key

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
            obj = json.loads(content)
        except httpx.HTTPError as exc:
            return ArticleSummary(
                article=article,
                bullets=[f"Provider error: {type(exc).__name__}"],
                takeaway=article.summary or "",
                status="provider_error",
            )
        except json.JSONDecodeError:
            return ArticleSummary(
                article=article,
                bullets=["Unable to parse summary JSON."],
                takeaway=content[:500] if content else "",
                status="parse_error",
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
        except (httpx.HTTPError, json.JSONDecodeError):
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


def _extract_text(data: dict[str, Any]) -> str:
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:  # noqa: BLE001
        return ""


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
