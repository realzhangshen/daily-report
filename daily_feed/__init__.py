"""
Daily Feed Agent - AI-powered RSS feed summarizer.

This package processes RSS feed exports (Folo JSON format) to generate
HTML/Markdown reports with AI-powered summaries and topic grouping.

Main entry point is the CLI via `daily-feed run` command.

Example:
    $ daily-feed run -i feeds.json -o output/
"""

__all__ = ["__version__", "parse_folo_json", "EntryManager", "slugify", "short_hash"]
__version__ = "0.1.0"

from .core.entry import EntryManager, short_hash, slugify
from .input.json_parser import parse_folo_json
