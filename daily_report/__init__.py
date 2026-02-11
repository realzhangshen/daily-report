"""
Daily Report Agent - AI-powered RSS feed analyzer.

This package processes RSS feed exports (Folo JSON format) to generate
HTML/Markdown reports with AI-powered analysis and optional deep fetch.

Main entry point is the CLI via `daily-report run` command.

Example:
    $ daily-report run -i feeds.json -o output/
"""

__all__ = ["__version__", "parse_folo_json", "EntryManager", "slugify", "short_hash"]
__version__ = "0.1.0"

from .core.entry import EntryManager, short_hash, slugify
from .input.json_parser import parse_folo_json
