"""
LLM provider implementations.

This package contains abstract base class and concrete implementations
 for different LLM providers (currently Gemini).

To add a new provider:
1. Inherit from Provider base class
2. Implement summarize_article() and group_topics()
3. Export the class from __init__.py
4. Add provider name to ProviderConfig in config.py
5. Add provider instantiation logic in runner.py:_build_provider()
"""

from .base import Provider
from .gemini import GeminiProvider

__all__ = ["Provider", "GeminiProvider"]
