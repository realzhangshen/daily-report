"""Provider factory and registry for hot-swappable LLM backends."""

from __future__ import annotations

from ...config import LoggingConfig, ProviderConfig, SummaryConfig, get_api_key
from .base import AnalysisProvider
from .gemini import GeminiProvider
from .openai_compatible import OpenAICompatibleProvider


ProviderBuilder = type[AnalysisProvider]

_PROVIDER_REGISTRY: dict[str, ProviderBuilder] = {
    "gemini": GeminiProvider,
    "openai": OpenAICompatibleProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "openai-compatible": OpenAICompatibleProvider,
}


def available_providers() -> list[str]:
    """Return the set of registered provider names."""
    return sorted(_PROVIDER_REGISTRY.keys())


def create_provider(
    provider_cfg: ProviderConfig,
    summary_cfg: SummaryConfig,
    log_cfg: LoggingConfig,
    llm_logger,
) -> AnalysisProvider:
    """Build a provider instance from runtime config."""
    name = provider_cfg.name.lower().strip()
    builder = _PROVIDER_REGISTRY.get(name)
    if builder is None:
        supported = ", ".join(available_providers())
        raise ValueError(f"Unsupported provider: {provider_cfg.name}. Supported: {supported}")
    api_key = get_api_key(provider_cfg)
    return builder(provider_cfg, summary_cfg, api_key, log_cfg, llm_logger)
