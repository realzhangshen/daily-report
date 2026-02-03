from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from rich.logging import RichHandler

from .config import LoggingConfig


_URL_RE = re.compile(r"https?://\\S+")


def setup_logging(cfg: LoggingConfig, run_output_dir: Path | None) -> logging.Logger:
    logger = logging.getLogger("daily_feed")
    logger.setLevel(_level_from_string(cfg.level))
    logger.handlers = []
    logger.propagate = False

    if cfg.console:
        console_handler = RichHandler(rich_tracebacks=True, show_time=False, show_level=True)
        console_handler.setLevel(_level_from_string(cfg.level))
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

    if cfg.file and run_output_dir is not None:
        run_output_dir.mkdir(parents=True, exist_ok=True)
        file_path = run_output_dir / cfg.filename
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(_level_from_string(cfg.level))
        file_handler.setFormatter(_build_file_formatter(cfg.format))
        logger.addHandler(file_handler)

    return logger


def setup_llm_logger(cfg: LoggingConfig, run_output_dir: Path | None) -> logging.Logger | None:
    if not cfg.llm_log_enabled:
        return None
    if run_output_dir is None:
        return None

    logger = logging.getLogger("daily_feed.llm")
    logger.setLevel(_level_from_string(cfg.level))
    logger.handlers = []
    logger.propagate = False

    run_output_dir.mkdir(parents=True, exist_ok=True)
    file_path = run_output_dir / cfg.llm_log_file
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(_level_from_string(cfg.level))
    file_handler.setFormatter(JsonlFormatter())
    logger.addHandler(file_handler)
    return logger


def log_event(logger: logging.Logger | None, message: str, **fields: Any) -> None:
    if logger is None:
        return
    logger.info(message, extra=fields)


def redact_text(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "redact_content":
        return ""
    if mode == "redact_urls_authors":
        return _URL_RE.sub("[REDACTED_URL]", text)
    return text


def redact_value(value: str | None, mode: str) -> str | None:
    if value is None:
        return None
    if mode == "redact_urls_authors":
        return "[REDACTED]"
    if mode == "redact_content":
        return None
    return value


def truncate_text(text: str, max_chars: int = 20000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extract_extras(record))
        return json.dumps(payload, ensure_ascii=True)


def _extract_extras(record: logging.LogRecord) -> dict[str, Any]:
    reserved = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }
    extras = {}
    for key, value in record.__dict__.items():
        if key in reserved:
            continue
        extras[key] = value
    return extras


def _build_file_formatter(fmt: str) -> logging.Formatter:
    if fmt == "jsonl":
        return JsonlFormatter()
    return logging.Formatter("%(asctime)s %(levelname)s %(message)s")


def _level_from_string(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)
