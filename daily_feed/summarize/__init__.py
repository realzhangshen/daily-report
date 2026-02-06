"""
AI-powered summarization and observability.

This package contains LLM provider implementations and
observability/tracing integration.
"""

from .providers.base import Provider
from .providers.gemini import GeminiProvider
from .tracing import setup_langfuse, flush, start_span, set_span_output, record_span_error

__all__ = [
    "Provider",
    "GeminiProvider",
    "setup_langfuse",
    "flush",
    "start_span",
    "set_span_output",
    "record_span_error",
]
