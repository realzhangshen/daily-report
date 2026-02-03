from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import yaml


@dataclass
class FetchConfig:
    timeout_seconds: float = 20.0
    retries: int = 2
    trust_env: bool = True
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )


@dataclass
class ExtractConfig:
    primary: str = "trafilatura"
    fallback: list[str] = field(default_factory=lambda: ["readability", "bs4"])


@dataclass
class DedupConfig:
    enabled: bool = True
    title_similarity_threshold: int = 92


@dataclass
class SummaryConfig:
    bullets_min: int = 3
    bullets_max: int = 6
    max_chars: int = 12000


@dataclass
class GroupingConfig:
    method: str = "provider"
    fallback: str = "site"


@dataclass
class OutputConfig:
    format: str = "html"
    include_markdown: bool = False
    run_folder_mode: str = "input"


@dataclass
class ProviderConfig:
    name: str = "gemini"
    model: str = "gemini-3-flash-preview"
    google_api_key_env: str = "GOOGLE_API_KEY"
    base_url: str = "https://generativelanguage.googleapis.com"
    api_key: str | None = None
    trust_env: bool = True


@dataclass
class AppConfig:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


DEFAULT_CONFIG = AppConfig()


def load_config(path: str | None) -> AppConfig:
    if not path:
        return DEFAULT_CONFIG

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return _merge_config(DEFAULT_CONFIG, raw)


def _merge_config(base: AppConfig, raw: dict[str, Any]) -> AppConfig:
    data = _asdict(base)
    for key, value in raw.items():
        if key not in data:
            continue
        if isinstance(value, dict) and isinstance(data[key], dict):
            data[key].update(value)
        else:
            data[key] = value
    return _fromdict(data)


def _asdict(cfg: AppConfig) -> dict[str, Any]:
    return {
        "provider": {
            "name": cfg.provider.name,
            "model": cfg.provider.model,
            "google_api_key_env": cfg.provider.google_api_key_env,
            "base_url": cfg.provider.base_url,
            "api_key": cfg.provider.api_key,
            "trust_env": cfg.provider.trust_env,
        },
        "fetch": {
            "timeout_seconds": cfg.fetch.timeout_seconds,
            "retries": cfg.fetch.retries,
            "trust_env": cfg.fetch.trust_env,
            "user_agent": cfg.fetch.user_agent,
        },
        "extract": {
            "primary": cfg.extract.primary,
            "fallback": cfg.extract.fallback,
        },
        "dedup": {
            "enabled": cfg.dedup.enabled,
            "title_similarity_threshold": cfg.dedup.title_similarity_threshold,
        },
        "summary": {
            "bullets_min": cfg.summary.bullets_min,
            "bullets_max": cfg.summary.bullets_max,
            "max_chars": cfg.summary.max_chars,
        },
        "grouping": {
            "method": cfg.grouping.method,
            "fallback": cfg.grouping.fallback,
        },
        "output": {
            "format": cfg.output.format,
            "include_markdown": cfg.output.include_markdown,
            "run_folder_mode": cfg.output.run_folder_mode,
        },
    }


def _fromdict(data: dict[str, Any]) -> AppConfig:
    return AppConfig(
        provider=ProviderConfig(**data["provider"]),
        fetch=FetchConfig(**data["fetch"]),
        extract=ExtractConfig(**data["extract"]),
        dedup=DedupConfig(**data["dedup"]),
        summary=SummaryConfig(**data["summary"]),
        grouping=GroupingConfig(**data["grouping"]),
        output=OutputConfig(**data["output"]),
    )


def get_api_key(cfg: ProviderConfig) -> str | None:
    if cfg.api_key:
        return cfg.api_key
    return os.getenv(cfg.google_api_key_env)
