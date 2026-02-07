"""Tests for hot-swappable LLM provider factory."""

import pytest

from daily_feed.config import LoggingConfig, ProviderConfig, SummaryConfig
from daily_feed.llm.providers.factory import available_providers, create_provider
from daily_feed.llm.providers.gemini import GeminiProvider
from daily_feed.llm.providers.openai_compatible import OpenAICompatibleProvider


def test_available_providers_contains_expected_backends():
    names = available_providers()
    assert "gemini" in names
    assert "openai" in names
    assert "openai_compatible" in names


def test_create_provider_gemini():
    provider = create_provider(
        ProviderConfig(
            name="gemini",
            model="gemini-3-flash-preview",
            api_key="test-key",
            base_url="https://generativelanguage.googleapis.com",
        ),
        SummaryConfig(),
        LoggingConfig(),
        llm_logger=None,
    )
    assert isinstance(provider, GeminiProvider)


def test_create_provider_openai_compatible():
    provider = create_provider(
        ProviderConfig(
            name="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        ),
        SummaryConfig(),
        LoggingConfig(),
        llm_logger=None,
    )
    assert isinstance(provider, OpenAICompatibleProvider)


def test_create_provider_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported provider"):
        create_provider(
            ProviderConfig(
                name="unknown-provider",
                model="x",
                api_key="test-key",
                base_url="https://example.com",
            ),
            SummaryConfig(),
            LoggingConfig(),
            llm_logger=None,
        )
