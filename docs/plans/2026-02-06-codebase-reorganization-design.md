# Codebase Reorganization Design

**Date:** 2026-02-06
**Status:** Approved

## Overview

Reorganize the `daily_feed/` package from a flat structure to logical sub-packages that mirror the data processing pipeline.

## Current Structure

```
daily_feed/
├── __init__.py
├── cache.py
├── cli.py
├── config.py
├── dedup.py
├── entry_manager.py
├── extractor.py
├── fetcher.py
├── json_parser.py
├── langfuse_utils.py
├── logging_utils.py
├── providers/
├── renderer.py
├── runner.py
├── templates/
└── types.py
```

**Problems:**
- Too many files in one directory (cognitive load)
- Unclear which modules handle related functionality
- Doesn't scale well for future features
- Difficult for new contributors to understand architecture

## Target Structure

```
daily_feed/
├── __init__.py          # Package initialization, exports public API
├── cli.py               # CLI entry point (Click commands)
├── config.py            # Configuration loading/validation
├── runner.py            # Main orchestration
│
├── core/                # Domain models & business logic
│   ├── __init__.py
│   ├── types.py         # All type definitions
│   ├── entry.py         # EntryManager (from entry_manager.py)
│   └── dedup.py         # Deduplication logic
│
├── fetch/               # Article fetching & extraction
│   ├── __init__.py
│   ├── fetcher.py       # HTTP fetching, Crawl4AI
│   ├── extractor.py     # Text extraction
│   └── cache.py         # Caching logic
│
├── summarize/           # AI summarization
│   ├── __init__.py
│   ├── providers/       # AI provider implementations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gemini.py
│   │   └── ...
│   └── tracing.py       # Langfuse observability
│
├── input/               # Input parsing
│   ├── __init__.py
│   └── json_parser.py   # RSS export JSON parsing
│
├── output/              # Rendering & output
│   ├── __init__.py
│   ├── renderer.py      # HTML/Markdown rendering
│   └── templates/       # Jinja2 templates
│       └── ...
│
└── utils/               # Shared utilities
    ├── __init__.py
    └── logging.py       # Logging setup
```

## Package Responsibilities

### Top-Level Files
- **`cli.py`**: CLI entry point, handles user input parsing
- **`config.py`**: YAML → config object loading/validation
- **`runner.py`**: Main orchestration, wires together pipeline stages

### `core/` Package
Domain models and business logic independent of pipeline stages.

### `fetch/` Package
All article retrieval and extraction logic.

### `summarize/` Package
AI-powered summarization and observability.

### `input/` Package
Parsing input from various sources (JSON, future: RSS, OPML).

### `output/` Package
Rendering and writing final output (HTML, Markdown, future: PDF).

### `utils/` Package
Shared utility code (logging, future: text helpers, date formatting).

## Migration Strategy

1. Create new package structure alongside existing files
2. Update imports incrementally - one package at a time
3. Run tests after each package migration
4. Delete old files once all imports updated and tests pass

## Benefits

- **Clearer navigation**: Find code by pipeline stage
- **Better onboarding**: New contributors understand architecture quickly
- **Scalability**: Easy to add new features without clutter
- **Maintainability**: Clear boundaries between concerns
