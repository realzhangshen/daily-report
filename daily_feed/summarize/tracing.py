"""
Langfuse tracing helpers for prompt and workflow iteration.

This module wraps the Langfuse SDK so the pipeline can emit traces/spans
without hard dependencies when tracing is disabled.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from typing import Any, Iterator

from ..config import LangfuseConfig
from ..logging_utils import redact_text, truncate_text

_TRACER = None
_CFG: LangfuseConfig | None = None
_TRACE_STACK: ContextVar[list[Any]] = ContextVar("langfuse_trace_stack", default=[])


def setup_langfuse(cfg: LangfuseConfig) -> None:
    """Initialize Langfuse tracing if enabled."""
    global _TRACER, _CFG  # noqa: PLW0603
    _CFG = cfg
    if not cfg.enabled:
        _TRACER = None
        return
    try:
        from langfuse import Langfuse  # type: ignore
    except Exception:  # noqa: BLE001
        _TRACER = None
        return

    _TRACER = Langfuse(
        public_key=_coalesce(cfg.public_key, "LANGFUSE_PUBLIC_KEY"),
        secret_key=_coalesce(cfg.secret_key, "LANGFUSE_SECRET_KEY"),
        host=_coalesce(cfg.host, "LANGFUSE_HOST"),
        environment=_coalesce(cfg.environment, "LANGFUSE_ENVIRONMENT"),
        release=_coalesce(cfg.release, "LANGFUSE_RELEASE"),
    )


def get_tracer():
    return _TRACER


@contextmanager
def start_span(
    name: str,
    kind: str,
    input_value: Any | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Start a Langfuse span if tracing is enabled."""
    tracer = _TRACER
    if tracer is None:
        yield None
        return

    attrs = _clean_attributes(attributes or {})
    if kind:
        attrs.setdefault("span.kind", kind)
    input_payload = _normalize_text(input_value)

    span = None
    try:
        # Use start_as_current_span for proper context tracking
        cm = tracer.start_as_current_span(
            name=name,
            input=input_payload,
            metadata=attrs,
        )
        span = cm.__enter__()
    except Exception:  # noqa: BLE001
        yield None
        return

    try:
        yield span
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass


def set_span_output(span: Any | None, output_value: Any) -> None:
    if span is None:
        return
    payload = _normalize_text(output_value)
    if payload is None:
        return
    _safe_update(span, output=payload)


def record_span_error(span: Any | None, exc: Exception) -> None:
    if span is None:
        return
    _safe_update(span, status="error", metadata={"error": str(exc)})


def _coalesce(value: str | None, env_key: str) -> str | None:
    if value:
        return value
    return _getenv(env_key)


def _getenv(key: str) -> str | None:
    import os

    return os.getenv(key)


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=True, default=str)
    cfg = _CFG
    if cfg is None:
        return text
    text = redact_text(text, cfg.redaction)
    return truncate_text(text, cfg.max_text_chars)


def _clean_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def _safe_update(span: Any, **kwargs: Any) -> None:
    try:
        if hasattr(span, "update"):
            span.update(**kwargs)
            return
        if "output" in kwargs and hasattr(span, "end"):
            span.end(output=kwargs["output"])
            return
        if hasattr(span, "end"):
            span.end()
    except Exception:  # noqa: BLE001
        return


def _safe_end(span: Any) -> None:
    try:
        if hasattr(span, "end"):
            span.end()
    except Exception:  # noqa: BLE001
        return


def flush() -> None:
    """Flush any pending traces to Langfuse.

    Langfuse uses async ingestion by default. Call this before program
    exit to ensure all traces are sent.
    """
    tracer = _TRACER
    if tracer is None:
        return
    try:
        if hasattr(tracer, "flush"):
            tracer.flush()
    except Exception:  # noqa: BLE001
        return
