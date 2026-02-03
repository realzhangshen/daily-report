"""
Phoenix tracing helpers for prompt and workflow iteration.

This module provides a small wrapper around arize-phoenix-otel so the
pipeline can emit OpenInference-compatible spans without hard
dependencies when Phoenix is disabled.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from typing import Any, Iterator

from .config import PhoenixConfig
from .logging_utils import redact_text, truncate_text

_TRACER = None
_CFG: PhoenixConfig | None = None


def setup_phoenix(cfg: PhoenixConfig) -> None:
    """Initialize Phoenix tracing if enabled."""
    global _TRACER, _CFG  # noqa: PLW0603
    _CFG = cfg
    if not cfg.enabled:
        _TRACER = None
        return
    try:
        from phoenix.otel import register  # type: ignore
    except Exception:  # noqa: BLE001
        _TRACER = None
        return

    kwargs: dict[str, Any] = {
        "auto_instrument": cfg.auto_instrument,
        "batch": cfg.batch,
    }
    if cfg.project_name:
        kwargs["project_name"] = cfg.project_name
    if cfg.collector_endpoint:
        kwargs["endpoint"] = cfg.collector_endpoint
    if cfg.protocol:
        kwargs["protocol"] = cfg.protocol
    if cfg.headers:
        kwargs["headers"] = cfg.headers
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key

    tracer_provider = register(**kwargs)
    _TRACER = tracer_provider.get_tracer("daily_feed")


def get_tracer():
    return _TRACER


@contextmanager
def start_span(
    name: str,
    kind: str,
    input_value: Any | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Start a Phoenix span if tracing is enabled."""
    tracer = _TRACER
    if tracer is None:
        yield None
        return

    attrs = _clean_attributes(attributes or {})
    ctx = _start_span_context(tracer, name, kind, attrs)
    if ctx is None:
        yield None
        return

    with ctx as span:
        if input_value is not None:
            _set_span_input(span, input_value)
        yield span


def set_span_output(span: Any | None, output_value: Any) -> None:
    if span is None:
        return
    _set_span_output(span, output_value)


def record_span_error(span: Any | None, exc: Exception) -> None:
    if span is None:
        return
    if hasattr(span, "record_exception"):
        span.record_exception(exc)
    if hasattr(span, "set_status"):
        try:
            from opentelemetry.trace import Status, StatusCode  # type: ignore

            span.set_status(Status(StatusCode.ERROR))
        except Exception:  # noqa: BLE001
            return


def _start_span_context(tracer, name: str, kind: str, attrs: dict[str, Any]):
    try:
        return tracer.start_as_current_span(
            name, openinference_span_kind=kind, attributes=attrs
        )
    except TypeError:
        if kind:
            attrs["openinference.span.kind"] = kind
        return tracer.start_as_current_span(name, attributes=attrs)
    except Exception:  # noqa: BLE001
        return None


def _set_span_input(span: Any, value: Any) -> None:
    payload = _normalize_text(value)
    if payload is None:
        return
    if hasattr(span, "set_input"):
        span.set_input(payload)
    else:
        span.set_attribute("input.value", payload)


def _set_span_output(span: Any, value: Any) -> None:
    payload = _normalize_text(value)
    if payload is None:
        return
    if hasattr(span, "set_output"):
        span.set_output(payload)
    else:
        span.set_attribute("output.value", payload)


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
