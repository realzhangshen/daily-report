"""
Configuration management using YAML files and dataclasses.

This module defines all configuration dataclasses and provides loading
from YAML files with defaults. Configuration sections:
- FetchConfig: Remote Crawl4AI API fetching settings
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
    """Configuration for HTTP content fetching via remote Crawl4AI API.

    Attributes:
        timeout_seconds: HTTP request timeout
        retries: Number of retry attempts for failed requests
        trust_env: Whether to respect system proxy settings
        user_agent: HTTP User-Agent header string
        crawl4ai_stealth: Enable stealth mode to bypass bot detection
        crawl4ai_delay: Delay before returning HTML (allows challenges to complete)
        crawl4ai_simulate_user: Simulate user behavior for anti-bot
        crawl4ai_magic: Enable anti-detection "magic" mode
        crawl4ai_api_url: Remote Crawl4AI API URL (required - uses environment variable if not set)
        crawl4ai_api_username: HTTP Basic Auth username (optional, for nginx auth)
        crawl4ai_api_password: HTTP Basic Auth password (optional, for nginx auth)
        deep_fetch_max_links: Maximum number of deep links to fetch per entry
    """

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
    # Remote API configuration
    crawl4ai_api_url: str | None = None
    crawl4ai_api_username: str | None = None
    crawl4ai_api_password: str | None = None
    deep_fetch_max_links: int = 5


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
        analysis_max_output_tokens: Max output tokens for per-entry analysis call
        analysis_concurrency: Number of parallel workers for per-entry analysis
        enable_deep_fetch_decision: Whether to run LLM deep-fetch decision step
    """

    bullets_min: int = 3
    bullets_max: int = 6
    max_chars: int = 10000
    analysis_max_output_tokens: int = 1200
    extraction_max_output_tokens: int = 300
    synthesis_max_output_tokens: int = 4096
    analysis_concurrency: int = 1
    enable_deep_fetch_decision: bool = True


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
        llm_log_detail: LLM log detail level ("response_only", "prompt_response")
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
        timeout_seconds: Timeout for Langfuse ingestion requests
        redaction: Redaction mode for prompt/response payloads
        max_text_chars: Maximum characters for prompt/response payloads
    """

    enabled: bool = False
    public_key: str | None = None
    secret_key: str | None = None
    host: str | None = None
    environment: str | None = None
    release: str | None = None
    timeout_seconds: int = 30
    redaction: str = "redact_urls_authors"
    max_text_chars: int = 20000


@dataclass
class CacheConfig:
    """Configuration for caching.

    Attributes:
        enabled: Whether to read from/write to entry cache
        ttl_days: Optional time-to-live for cache entries in days
    """

    enabled: bool = True
    ttl_days: int | None = None


@dataclass
class ProviderConfig:
    """Configuration for pluggable LLM providers."""

    name: str = "gemini"
    model: str = "gemini-3-flash-preview"
    api_key_env: str | None = None
    base_url: str = "https://generativelanguage.googleapis.com"
    api_key: str | None = None
    trust_env: bool = True


@dataclass
class AppConfig:
    """Root configuration container aggregating all config sections."""

    provider: ProviderConfig = field(default_factory=ProviderConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
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
        return _fromdict(_asdict(DEFAULT_CONFIG))

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
            "api_key_env": cfg.provider.api_key_env,
            "base_url": cfg.provider.base_url,
            "api_key": cfg.provider.api_key,
            "trust_env": cfg.provider.trust_env,
        },
        "fetch": {
            "timeout_seconds": cfg.fetch.timeout_seconds,
            "retries": cfg.fetch.retries,
            "trust_env": cfg.fetch.trust_env,
            "user_agent": cfg.fetch.user_agent,
            "deep_fetch_max_links": cfg.fetch.deep_fetch_max_links,
            "crawl4ai_stealth": cfg.fetch.crawl4ai_stealth,
            "crawl4ai_delay": cfg.fetch.crawl4ai_delay,
            "crawl4ai_simulate_user": cfg.fetch.crawl4ai_simulate_user,
            "crawl4ai_magic": cfg.fetch.crawl4ai_magic,
            "crawl4ai_api_url": cfg.fetch.crawl4ai_api_url,
            "crawl4ai_api_username": cfg.fetch.crawl4ai_api_username,
            "crawl4ai_api_password": cfg.fetch.crawl4ai_api_password,
        },
        "dedup": {
            "enabled": cfg.dedup.enabled,
            "title_similarity_threshold": cfg.dedup.title_similarity_threshold,
        },
        "summary": {
            "bullets_min": cfg.summary.bullets_min,
            "bullets_max": cfg.summary.bullets_max,
            "max_chars": cfg.summary.max_chars,
            "analysis_max_output_tokens": cfg.summary.analysis_max_output_tokens,
            "extraction_max_output_tokens": cfg.summary.extraction_max_output_tokens,
            "synthesis_max_output_tokens": cfg.summary.synthesis_max_output_tokens,
            "analysis_concurrency": cfg.summary.analysis_concurrency,
            "enable_deep_fetch_decision": cfg.summary.enable_deep_fetch_decision,
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
            "timeout_seconds": cfg.langfuse.timeout_seconds,
            "redaction": cfg.langfuse.redaction,
            "max_text_chars": cfg.langfuse.max_text_chars,
        },
        "cache": {
            "enabled": cfg.cache.enabled,
            "ttl_days": cfg.cache.ttl_days,
        },
    }


def _fromdict(data: dict[str, Any]) -> AppConfig:
    """Reconstruct AppConfig from nested dictionary."""
    # Handle old config files that may have deprecated fields
    fetch_data = data.get("fetch", {})
    # Ignore deprecated fields for backward compatibility
    fetch_data.pop("backend", None)
    fetch_data.pop("fallback_to_httpx", None)
    fetch_data.pop("crawl4ai_concurrency", None)
    cache_data = data.get("cache", {})
    cache_data.pop("mode", None)
    cache_data.pop("shared_dir", None)
    cache_data.pop("write_index", None)
    cache_data.pop("index_filename", None)

    # Handle old summary configs that may not have new fields
    summary_data = data.get("summary", {})

    return AppConfig(
        provider=ProviderConfig(**data["provider"]),
        fetch=FetchConfig(**fetch_data),
        dedup=DedupConfig(**data["dedup"]),
        summary=SummaryConfig(**summary_data),
        grouping=GroupingConfig(**data["grouping"]),
        output=OutputConfig(**data["output"]),
        logging=LoggingConfig(**data["logging"]),
        langfuse=LangfuseConfig(**data.get("langfuse", {})),
        cache=CacheConfig(**cache_data),
    )


def get_api_key(cfg: ProviderConfig) -> str | None:
    """Get API key from inline config or environment variable."""
    if cfg.api_key:
        return cfg.api_key
    if cfg.api_key_env:
        return os.getenv(cfg.api_key_env)
    defaults = {
        "gemini": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openai_compatible": "OPENAI_API_KEY",
        "openai-compatible": "OPENAI_API_KEY",
    }
    env_name = defaults.get(cfg.name.lower(), "OPENAI_API_KEY")
    return os.getenv(env_name)


def get_crawl4ai_api_url(cfg: FetchConfig) -> str | None:
    """Get Crawl4AI API URL from inline config or environment variable."""
    if cfg.crawl4ai_api_url:
        return cfg.crawl4ai_api_url
    return os.getenv("CRAWL4AI_API_URL")


def get_crawl4ai_api_auth(cfg: FetchConfig) -> tuple[str, str] | None:
    """Get Crawl4AI API Basic Auth credentials from config or environment variables.

    Returns (username, password) tuple if both are configured, None otherwise.
    """
    username = cfg.crawl4ai_api_username or os.getenv("CRAWL4AI_API_USERNAME")
    password = cfg.crawl4ai_api_password or os.getenv("CRAWL4AI_API_PASSWORD")
    if username and password:
        return (username, password)
    return None


def get_langfuse_host(cfg: LangfuseConfig) -> str | None:
    """Get Langfuse host from inline config or environment variable."""
    if cfg.host:
        return cfg.host
    return os.getenv("LANGFUSE_HOST")
