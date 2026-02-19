"""
Best-effort sync of generated reports into web_dailyreport index.

This module is intentionally defensive:
- If web_dailyreport workspace cannot be located, it silently skips sync.
- If report date cannot be determined, it skips sync.
- If index file is malformed, it rewrites a clean normalized index.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


DATE_PATTERN = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def sync_report_to_web_index(*, html_path: Path, title: str, input_path: Path) -> Path | None:
    """Sync a generated report into ``web_dailyreport/data/reports.json``.

    Returns:
        Path to updated index file on success, or ``None`` when skipped.
    """
    web_root = _resolve_web_root(html_path=html_path, input_path=input_path)
    if web_root is None:
        return None

    report_date = _detect_report_date(input_path=input_path, html_path=html_path)
    if report_date is None:
        return None

    index_path = web_root / "data" / "reports.json"
    entry = {
        "date": report_date,
        "title": title or f"Folo 每日简报 - {report_date}",
        "path": f"reports/{report_date}/report.html",
    }

    reports = _load_reports(index_path)
    updated = False
    for idx, item in enumerate(reports):
        if item.get("date") == report_date:
            reports[idx] = entry
            updated = True
            break

    if not updated:
        reports.append(entry)

    reports.sort(key=lambda item: item["date"], reverse=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(f"{json.dumps(reports, ensure_ascii=False, indent=2)}\n", encoding="utf-8")
    return index_path


def _resolve_web_root(*, html_path: Path, input_path: Path) -> Path | None:
    env_root = os.getenv("WEB_DAILYREPORT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if candidate.exists():
            return candidate
        return None

    probes = [html_path.resolve(), input_path.resolve(), Path.cwd().resolve()]
    for probe in probes:
        for parent in [probe, *probe.parents]:
            candidate = parent / "web_dailyreport"
            if candidate.exists():
                return candidate
    return None


def _detect_report_date(*, input_path: Path, html_path: Path) -> str | None:
    for text in (input_path.stem, input_path.name, html_path.parent.name, html_path.name):
        match = DATE_PATTERN.search(text)
        if match:
            return match.group(1)
    return None


def _load_reports(index_path: Path) -> list[dict[str, str]]:
    if not index_path.exists():
        return []

    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if not isinstance(raw, list):
        return []

    reports: list[dict[str, str]] = []
    for item in raw:
        normalized = _normalize_report_item(item)
        if normalized is not None:
            reports.append(normalized)
    return reports


def _normalize_report_item(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None

    date = item.get("date")
    title = item.get("title")
    path_value = item.get("path")
    if not isinstance(date, str) or not isinstance(title, str) or not isinstance(path_value, str):
        return None

    match = DATE_PATTERN.fullmatch(date)
    if match is None:
        return None

    return {
        "date": date,
        "title": title,
        "path": path_value,
    }
