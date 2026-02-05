"""
Configuration management using YAML files and dataclasses.

This module defines all configuration dataclasses and provides loading
from YAML files with defaults. Configuration sections:
- FetchConfig: HTTP fetching settings
- ExtractConfig: Content extraction settings
- DedupConfig: Deduplication settings
- SummaryConfig: LLM summarization settings
- GroupingConfig: Article grouping settings
- OutputConfig: Output format settings
- LoggingConfig: Logging behavior
- CacheConfig: Cache directory and TTL settings
- ProviderConfig: LLM provider settings
- LangfuseConfig: Langfuse tracing settings
- AppConfig: Root configuration container
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import yaml


@dataclass
class FetchConfig:
    """Configuration for HTTP content fetching.

    Attributes:
        backend: "httpx" for fast fetching, "crawl4ai" for JS rendering, "curl_cffi" for Cloudflare bypass
        fallback_to_httpx: If True, falls back to httpx when crawl4ai/curl_cffi fails
        crawl4ai_concurrency: Number of concurrent crawl4ai requests
        timeout_seconds: HTTP request timeout
        retries: Number of retry attempts for failed requests
        trust_env: Whether to respect system proxy settings
        user_agent: HTTP User-Agent header string
        crawl4ai_stealth: Enable stealth mode to bypass bot detection
        crawl4ai_delay: Delay before returning HTML (allows challenges to complete)
        crawl4ai_simulate_user: Simulate user behavior for anti-bot
        crawl4ai_magic: Enable anti-detection "magic" mode
    """

    backend: str = "httpx"
    fallback_to_httpx: bool = True
    crawl4ai_concurrency: int = 4
    timeout_seconds: float = 20.0
    retries: int = 2
    trust_env: bool = True
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    # Anti-bot detection options for Crawl4AI
    crawl4ai_stealth: bool = True
    crawl4ai_delay: float = 2.0
    crawl4ai_simulate_user: bool = True
    crawl4ai_magic: bool = True


@dataclass
class ExtractConfig:
    """Configuration for HTML content extraction.

    Attributes:
        primary: Primary extraction method ("trafilatura", "readability", or "bs4")
        fallback: List of fallback methods to try if primary fails
    """

    primary: str = "trafilatura"
    fallback: list[str] = field(default_factory=lambda: ["readability", "bs4"])


@dataclass
class DedupConfig:
    """Configuration for article deduplication.

    Attributes:
        enabled: Whether to perform deduplication
        title_similarity_threshold: Fuzzy match threshold (0-100) for title similarity
    """

    enabled: bool = True
    title_similarity_threshold: int = 92


@dataclass
class SummaryConfig:
    """Configuration for LLM summarization.

    Attributes:
        bullets_min: Minimum number of bullet points to generate
        bullets_max: Maximum number of bullet points to generate
        max_chars: Maximum characters of article text to send to LLM
    """

    bullets_min: int = 3
    bullets_max: int = 6
    max_chars: int = 12000


@dataclass
class GroupingConfig:
    """Configuration for topic grouping.

    Attributes:
        method: "provider" for LLM-based grouping, or "site" to use site names
        fallback: Fallback method if provider grouping fails
    """

    method: str = "provider"
    fallback: str = "site"


@dataclass
class OutputConfig:
    """Configuration for output generation.

    Attributes:
        format: "html" or "markdown"
        include_markdown: Whether to also generate markdown when format is "html"
        run_folder_mode: How to name output folders ("input", "timestamp", "input_timestamp")
    """

    format: str = "html"
    include_markdown: bool = False
    run_folder_mode: str = "input"


@dataclass
class LoggingConfig:
    """Configuration for logging behavior.

    Attributes:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        console: Whether to log to console
        file: Whether to log to file
        format: Log file format ("jsonl" or "plain")
        filename: Name of the main log file
        llm_log_enabled: Whether to enable separate LLM interaction logging
        llm_log_detail: LLM log detail level ("summary_only", "response_only", "prompt_response")
        llm_log_redaction: Redaction mode for LLM logs ("none", "redact_content", "redact_urls_authors")
        llm_log_file: Name of the LLM log file
    """

    level: str = "INFO"
    console: bool = True
    file: bool = True
    format: str = "jsonl"
    filename: str = "run.jsonl"
    llm_log_enabled: bool = True
    llm_log_detail: str = "response_only"
    llm_log_redaction: str = "redact_urls_authors"
    llm_log_file: str = "llm.jsonl"


@dataclass
class LangfuseConfig:
    """Configuration for Langfuse tracing.

    Attributes:
        enabled: Whether to enable Langfuse tracing
        public_key: Langfuse public key (optional)
        secret_key: Langfuse secret key (optional)
        host: Langfuse host URL (optional)
        environment: Langfuse environment label (optional)
        release: Langfuse release identifier (optional)
        redaction: Redaction mode for prompt/response payloads
        max_text_chars: Maximum characters for prompt/response payloads
    """

    enabled: bool = False
    public_key: str | None = None
    secret_key: str | None = None
    host: str | None = None
    environment: str | None = None
    release: str | None = None
    redaction: str = "redact_urls_authors"
    max_text_chars: int = 20000


@dataclass
class CacheConfig:
    """Configuration for caching.

    Attributes:
        mode: "run" for per-run cache, "shared" for persistent shared cache
        shared_dir: Custom path for shared cache directory
        ttl_days: Optional time-to-live for cache entries in days
        write_index: Whether to write cache index JSONL file
        index_filename: Name of the cache index file
    """

    mode: str = "run"
    shared_dir: str | None = None
    ttl_days: int | None = None
    write_index: bool = True
    index_filename: str = "index.jsonl"


@dataclass
class ProviderConfig:
    """Configuration for LLM provider.

    Attributes:
        name: Provider name ("gemini" currently supported)
        model: Model identifier (e.g., "gemini-3-flash-preview")
        google_api_key_env: Environment variable name containing the API key
        base_url: Base URL for the provider API
        api_key: Optional inline API key (overrides env var)
        trust_env: Whether to respect system proxy settings for API requests
    """

    name: str = "gemini"
    model: str = "gemini-3-flash-preview"
    google_api_key_env: str = "GOOGLE_API_KEY"
    base_url: str = "https://generativelanguage.googleapis.com"
    api_key: str | None = None
    trust_env: bool = True


@dataclass
class AppConfig:
    """Root configuration container aggregating all config sections."""

    provider: ProviderConfig = field(default_factory=ProviderConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


DEFAULT_CONFIG = AppConfig()


def load_config(path: str | None) -> AppConfig:
    """Load configuration from a YAML file with defaults."""
    if not path:
        return DEFAULT_CONFIG

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return _merge_config(DEFAULT_CONFIG, raw)


def _merge_config(base: AppConfig, raw: dict[str, Any]) -> AppConfig:
    """Merge raw YAML config into base AppConfig."""
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
    """Convert AppConfig to nested dictionary."""
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
            "backend": cfg.fetch.backend,
            "fallback_to_httpx": cfg.fetch.fallback_to_httpx,
            "crawl4ai_concurrency": cfg.fetch.crawl4ai_concurrency,
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
        "logging": {
            "level": cfg.logging.level,
            "console": cfg.logging.console,
            "file": cfg.logging.file,
            "format": cfg.logging.format,
            "filename": cfg.logging.filename,
            "llm_log_enabled": cfg.logging.llm_log_enabled,
            "llm_log_detail": cfg.logging.llm_log_detail,
            "llm_log_redaction": cfg.logging.llm_log_redaction,
            "llm_log_file": cfg.logging.llm_log_file,
        },
        "langfuse": {
            "enabled": cfg.langfuse.enabled,
            "public_key": cfg.langfuse.public_key,
            "secret_key": cfg.langfuse.secret_key,
            "host": cfg.langfuse.host,
            "environment": cfg.langfuse.environment,
            "release": cfg.langfuse.release,
            "redaction": cfg.langfuse.redaction,
            "max_text_chars": cfg.langfuse.max_text_chars,
        },
        "cache": {
            "mode": cfg.cache.mode,
            "shared_dir": cfg.cache.shared_dir,
            "ttl_days": cfg.cache.ttl_days,
            "write_index": cfg.cache.write_index,
            "index_filename": cfg.cache.index_filename,
        },
    }


def _fromdict(data: dict[str, Any]) -> AppConfig:
    """Reconstruct AppConfig from nested dictionary."""
    return AppConfig(
        provider=ProviderConfig(**data["provider"]),
        fetch=FetchConfig(**data["fetch"]),
        extract=ExtractConfig(**data["extract"]),
        dedup=DedupConfig(**data["dedup"]),
        summary=SummaryConfig(**data["summary"]),
        grouping=GroupingConfig(**data["grouping"]),
        output=OutputConfig(**data["output"]),
        logging=LoggingConfig(**data["logging"]),
        langfuse=LangfuseConfig(**data.get("langfuse", {})),
        cache=CacheConfig(**data["cache"]),
    )


def get_api_key(cfg: ProviderConfig) -> str | None:
    """Get API key from inline config or environment variable."""
    if cfg.api_key:
        return cfg.api_key
    return os.getenv(cfg.google_api_key_env)
