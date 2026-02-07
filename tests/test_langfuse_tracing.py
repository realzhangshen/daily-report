"""Tests for Langfuse tracing setup behavior."""

from __future__ import annotations

import sys
import types

from daily_feed.config import LangfuseConfig
from daily_feed.llm import tracing


def test_setup_langfuse_uses_only_langfuse_base_url_env(monkeypatch):
    captured: dict = {}

    class DummyLangfuse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.SimpleNamespace(Langfuse=DummyLangfuse)
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    monkeypatch.setenv("LANGFUSE_HOST", "https://should-be-ignored.example.com")

    tracing.setup_langfuse(LangfuseConfig(enabled=True, timeout_seconds=45))

    assert captured["public_key"] == "pk-test"
    assert captured["secret_key"] == "sk-test"
    assert captured["base_url"] == "https://us.cloud.langfuse.com"
    assert captured["timeout"] == 45


def test_setup_langfuse_disables_tracer_when_keys_missing(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    tracing.setup_langfuse(LangfuseConfig(enabled=True))

    assert tracing.get_tracer() is None
