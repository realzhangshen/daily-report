# Codebase Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize the `daily_feed/` package from a flat structure to logical sub-packages that mirror the data processing pipeline (INPUT → FETCH → EXTRACT → DEDUP → SUMMARIZE → OUTPUT).

**Architecture:** Create new sub-package directories (`core/`, `fetch/`, `summarize/`, `input/`, `output/`, `utils/`), move modules to appropriate packages, update all import statements, and verify tests pass after each package migration.

**Tech Stack:** Python 3.13, pytest, uv, git worktrees

---

## Migration Strategy

1. Create new package structure (one package at a time)
2. Move files to new locations
3. Update imports in affected files
4. Run tests to verify
5. Commit after each package

**Important:** Do NOT delete old files until all imports are updated and tests pass. This allows incremental migration with rollback capability.

---

## Task 1: Create `core/` Package

**Files:**
- Create: `daily_feed/core/__init__.py`
- Move: `daily_feed/types.py` → `daily_feed/core/types.py`
- Move: `daily_feed/entry_manager.py` → `daily_feed/core/entry.py`
- Move: `daily_feed/dedup.py` → `daily_feed/core/dedup.py`
- Modify: `daily_feed/__init__.py`
- Modify: `daily_feed/runner.py`
- Modify: `daily_feed/providers/base.py`
- Modify: `daily_feed/providers/gemini.py`
- Modify: `tests/test_entry_manager.py`
- Modify: `tests/test_integration_per_entry.py`

**Step 1: Create core package and __init__.py**

```bash
mkdir -p daily_feed/core
cat > daily_feed/core/__init__.py << 'EOF'
"""
Core domain models and business logic.

This package contains data types and business logic that is
independent of any specific pipeline stage.
"""

from .types import Article, ExtractedArticle, ArticleSummary
from .entry import EntryManager, slugify, short_hash
from .dedup import dedup_articles

__all__ = [
    "Article",
    "ExtractedArticle",
    "ArticleSummary",
    "EntryManager",
    "slugify",
    "short_hash",
    "dedup_articles",
]
EOF
```

**Step 2: Move types.py to core/types.py**

```bash
git mv daily_feed/types.py daily_feed/core/types.py
```

**Step 3: Move entry_manager.py to core/entry.py**

```bash
git mv daily_feed/entry_manager.py daily_feed/core/entry.py
```

**Step 4: Move dedup.py to core/dedup.py**

```bash
git mv daily_feed/dedup.py daily_feed/core/dedup.py
```

**Step 5: Update daily_feed/__init__.py imports**

Edit `daily_feed/__init__.py`, change:
```python
# Old:
from .entry_manager import EntryManager, short_hash, slugify
from .json_parser import parse_folo_json

# New:
from .core.entry import EntryManager, short_hash, slugify
from .input.json_parser import parse_folo_json
```

**Step 6: Update daily_feed/runner.py imports**

Edit `daily_feed/runner.py`, change:
```python
# Old:
from .dedup import dedup_articles
from .entry_manager import EntryManager
from .types import ArticleSummary, ExtractedArticle

# New:
from .core.dedup import dedup_articles
from .core.entry import EntryManager
from .core.types import ArticleSummary, ExtractedArticle
```

**Step 7: Update daily_feed/providers/base.py imports**

Edit `daily_feed/providers/base.py`, change:
```python
# Old:
from ..types import Article, ArticleSummary

# New:
from ..core.types import Article, ArticleSummary
```

**Step 8: Update daily_feed/providers/gemini.py imports**

Edit `daily_feed/providers/gemini.py`, change:
```python
# Old:
from ..types import Article, ArticleSummary

# New:
from ..core.types import Article, ArticleSummary
```

**Step 9: Update test imports in tests/test_entry_manager.py**

Edit `tests/test_entry_manager.py`, change:
```python
# Old:
from daily_feed.entry_manager import EntryManager
from daily_feed.types import Article, ArticleSummary

# New:
from daily_feed.core.entry import EntryManager
from daily_feed.core.types import Article, ArticleSummary
```

**Step 10: Update test imports in tests/test_integration_per_entry.py**

Edit `tests/test_integration_per_entry.py`, change:
```python
# Old:
from daily_feed.entry_manager import EntryManager
from daily_feed.types import Article, ArticleSummary

# New:
from daily_feed.core.entry import EntryManager
from daily_feed.core.types import Article, ArticleSummary
```

**Step 11: Run tests to verify**

```bash
pytest tests/ -v
```

Expected: All 18 tests PASS

**Step 12: Commit**

```bash
git add -A
git commit -m "refactor: create core package and move domain models

- Create daily_feed/core/ package for domain models
- Move types.py → core/types.py
- Move entry_manager.py → core/entry.py
- Move dedup.py → core/dedup.py
- Update all imports across codebase and tests

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Create `fetch/` Package

**Files:**
- Create: `daily_feed/fetch/__init__.py`
- Move: `daily_feed/fetcher.py` → `daily_feed/fetch/fetcher.py`
- Move: `daily_feed/extractor.py` → `daily_feed/fetch/extractor.py`
- Move: `daily_feed/cache.py` → `daily_feed/fetch/cache.py`
- Modify: `daily_feed/runner.py`
- Modify: `daily_feed/core/entry.py` (may import from cache)

**Step 1: Create fetch package and __init__.py**

```bash
mkdir -p daily_feed/fetch
cat > daily_feed/fetch/__init__.py << 'EOF'
"""
Article fetching and extraction.

This package handles HTTP fetching, content extraction,
and caching for article processing.
"""

from .fetcher import fetch_url_crawl4ai_api, Fetcher
from .extractor import extract_text
from .cache import PerEntryCache

__all__ = [
    "fetch_url_crawl4ai_api",
    "Fetcher",
    "extract_text",
    "PerEntryCache",
]
EOF
```

**Step 2: Move fetcher.py to fetch/fetcher.py**

```bash
git mv daily_feed/fetcher.py daily_feed/fetch/fetcher.py
```

**Step 3: Move extractor.py to fetch/extractor.py**

```bash
git mv daily_feed/extractor.py daily_feed/fetch/extractor.py
```

**Step 4: Move cache.py to fetch/cache.py**

```bash
git mv daily_feed/cache.py daily_feed/fetch/cache.py
```

**Step 5: Update daily_feed/runner.py imports**

Edit `daily_feed/runner.py`, change:
```python
# Old:
from .extractor import extract_text
from .fetcher import fetch_url_crawl4ai_api

# New:
from .fetch.extractor import extract_text
from .fetch.fetcher import fetch_url_crawl4ai_api
```

**Step 6: Check and update core/entry.py if it imports cache**

```bash
grep -n "from.*cache import\|from.*\.cache" daily_feed/core/entry.py
```

If imports exist, update them (likely importing PerEntryCache or similar).

**Step 7: Run tests to verify**

```bash
pytest tests/ -v
```

Expected: All 18 tests PASS

**Step 8: Commit**

```bash
git add -A
git commit -m "refactor: create fetch package for article retrieval

- Create daily_feed/fetch/ package
- Move fetcher.py → fetch/fetcher.py
- Move extractor.py → fetch/extractor.py
- Move cache.py → fetch/cache.py
- Update imports in runner.py and core/entry.py

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Create `summarize/` Package

**Files:**
- Create: `daily_feed/summarize/__init__.py`
- Move: `daily_feed/providers/` → `daily_feed/summarize/providers/`
- Move: `daily_feed/langfuse_utils.py` → `daily_feed/summarize/tracing.py`
- Modify: `daily_feed/runner.py`
- Modify: `daily_feed/cli.py`
- Modify: `daily_feed/fetch/cache.py` (may have langfuse imports)
- Modify: `daily_feed/summarize/providers/gemini.py`

**Step 1: Create summarize package**

```bash
mkdir -p daily_feed/summarize
```

**Step 2: Create summarize/__init__.py**

```bash
cat > daily_feed/summarize/__init__.py << 'EOF'
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
EOF
```

**Step 3: Move providers directory to summarize/providers**

```bash
git mv daily_feed/providers daily_feed/summarize/providers
```

**Step 4: Move langfuse_utils.py to summarize/tracing.py**

```bash
git mv daily_feed/langfuse_utils.py daily_feed/summarize/tracing.py
```

**Step 5: Update daily_feed/runner.py imports**

Edit `daily_feed/runner.py`, change:
```python
# Old:
from .langfuse_utils import set_span_output, setup_langfuse, start_span
from .providers.gemini import GeminiProvider

# New:
from .summarize.tracing import set_span_output, setup_langfuse, start_span
from .summarize.providers.gemini import GeminiProvider
```

**Step 6: Update daily_feed/cli.py imports**

Edit `daily_feed/cli.py`, change:
```python
# Old:
from .langfuse_utils import flush

# New:
from .summarize.tracing import flush
```

**Step 7: Update summarize/providers/gemini.py imports**

Edit `daily_feed/summarize/providers/gemini.py`, change:
```python
# Old:
from ..langfuse_utils import record_span_error, set_span_output, start_span

# New:
from ..tracing import record_span_error, set_span_output, start_span
```

**Step 8: Check for any other files importing langfuse_utils**

```bash
grep -r "from.*langfuse_utils\|import.*langfuse_utils" daily_feed/ tests/ --include="*.py"
```

Update any found imports to use `daily_feed.summarize.tracing`.

**Step 9: Run tests to verify**

```bash
pytest tests/ -v
```

Expected: All 18 tests PASS

**Step 10: Commit**

```bash
git add -A
git commit -m "refactor: create summarize package for AI features

- Create daily_feed/summarize/ package
- Move providers/ → summarize/providers/
- Move langfuse_utils.py → summarize/tracing.py
- Update imports across runner.py, cli.py, and providers

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Create `input/` Package

**Files:**
- Create: `daily_feed/input/__init__.py`
- Move: `daily_feed/json_parser.py` → `daily_feed/input/json_parser.py`
- Modify: `daily_feed/runner.py`
- Modify: `daily_feed/__init__.py` (already done in Task 1)

**Step 1: Create input package and __init__.py**

```bash
mkdir -p daily_feed/input
cat > daily_feed/input/__init__.py << 'EOF'
"""
Input parsing for various data sources.

This package handles parsing RSS exports and other input formats.
"""

from .json_parser import parse_folo_json

__all__ = ["parse_folo_json"]
EOF
```

**Step 2: Move json_parser.py to input/json_parser.py**

```bash
git mv daily_feed/json_parser.py daily_feed/input/json_parser.py
```

**Step 3: Update daily_feed/runner.py imports**

Edit `daily_feed/runner.py`, change:
```python
# Old:
from .json_parser import parse_folo_json

# New:
from .input.json_parser import parse_folo_json
```

**Step 4: Run tests to verify**

```bash
pytest tests/ -v
```

Expected: All 18 tests PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: create input package for parsing

- Create daily_feed/input/ package
- Move json_parser.py → input/json_parser.py
- Update imports in runner.py

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Create `output/` Package

**Files:**
- Create: `daily_feed/output/__init__.py`
- Move: `daily_feed/renderer.py` → `daily_feed/output/renderer.py`
- Move: `daily_feed/templates/` → `daily_feed/output/templates/`
- Modify: `daily_feed/runner.py`

**Step 1: Create output package and __init__.py**

```bash
mkdir -p daily_feed/output
cat > daily_feed/output/__init__.py << 'EOF'
"""
Output rendering and file generation.

This package handles HTML/Markdown rendering and
template management.
"""

from .renderer import render_html, render_markdown

__all__ = ["render_html", "render_markdown"]
EOF
```

**Step 2: Move renderer.py to output/renderer.py**

```bash
git mv daily_feed/renderer.py daily_feed/output/renderer.py
```

**Step 3: Move templates directory to output/templates**

```bash
git mv daily_feed/templates daily_feed/output/templates
```

**Step 4: Update daily_feed/runner.py imports**

Edit `daily_feed/runner.py`, change:
```python
# Old:
from .renderer import render_html, render_markdown

# New:
from .output.renderer import render_html, render_markdown
```

**Step 5: Check renderer.py for template path references**

```bash
grep -n "templates" daily_feed/output/renderer.py
```

If renderer.py references templates path (e.g., `Path(__file__).parent / "templates"`), update it to account for the new location.

**Step 6: Run tests to verify**

```bash
pytest tests/ -v
```

Expected: All 18 tests PASS

**Step 7: Commit**

```bash
git add -A
git commit -m "refactor: create output package for rendering

- Create daily_feed/output/ package
- Move renderer.py → output/renderer.py
- Move templates/ → output/templates/
- Update imports and template paths in renderer.py

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Create `utils/` Package

**Files:**
- Create: `daily_feed/utils/__init__.py`
- Move: `daily_feed/logging_utils.py` → `daily_feed/utils/logging.py`
- Modify: `daily_feed/runner.py`
- Modify: `daily_feed/config.py` (if imports logging_utils)
- Modify: `daily_feed/summarize/providers/gemini.py`
- Modify: `daily_feed/summarize/tracing.py` (formerly langfuse_utils.py)
- Modify: `daily_feed/fetch/cache.py`

**Step 1: Create utils package and __init__.py**

```bash
mkdir -p daily_feed/utils
cat > daily_feed/utils/__init__.py << 'EOF'
"""
Shared utility functions.

This package contains utility code used across multiple
pipeline stages.
"""

from .logging import setup_logging, setup_llm_logger, log_event, redact_text, truncate_text

__all__ = [
    "setup_logging",
    "setup_llm_logger",
    "log_event",
    "redact_text",
    "truncate_text",
]
EOF
```

**Step 2: Move logging_utils.py to utils/logging.py**

```bash
git mv daily_feed/logging_utils.py daily_feed/utils/logging.py
```

**Step 3: Find all files importing logging_utils**

```bash
grep -rn "from.*logging_utils\|import.*logging_utils" daily_feed/ --include="*.py"
```

**Step 4: Update imports in all found files**

For each file found, update imports:
```python
# Old:
from .logging_utils import ...

# New:
from .utils.logging import ...
```

Or for relative imports from subpackages:
```python
# Old:
from ..logging_utils import ...

# New:
from ...utils.logging import ...
```

**Step 5: Run tests to verify**

```bash
pytest tests/ -v
```

Expected: All 18 tests PASS

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: create utils package for shared utilities

- Create daily_feed/utils/ package
- Move logging_utils.py → utils/logging.py
- Update all logging_utils imports across codebase

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Final Verification and Cleanup

**Step 1: Run all tests with verbose output**

```bash
pytest tests/ -v --tb=short
```

Expected: All 18 tests PASS

**Step 2: Verify no old import statements remain**

```bash
grep -rn "from \.(cache|dedup|entry_manager|extractor|fetcher|json_parser|langfuse_utils|logging_utils|renderer|types|providers)" daily_feed/ --include="*.py" | grep -v "summarize/providers"
```

Expected: No results (all imports updated)

**Step 3: Verify final directory structure**

```bash
ls -la daily_feed/
ls -la daily_feed/core/
ls -la daily_feed/fetch/
ls -la daily_feed/summarize/
ls -la daily_feed/input/
ls -la daily_feed/output/
ls -la daily_feed/utils/
```

Expected structure:
```
daily_feed/
├── __init__.py
├── cli.py
├── config.py
├── runner.py
├── core/
├── fetch/
├── summarize/
├── input/
├── output/
└── utils/
```

**Step 4: Run the CLI to verify it works**

```bash
# Test help command
daily-feed run --help
```

Expected: Help output displays correctly

**Step 5: Final commit**

```bash
git add -A
git commit -m "refactor: complete codebase reorganization

- Verify all tests pass after reorganization
- Verify no old import statements remain
- Confirm CLI works correctly with new structure

All modules now organized into logical sub-packages:
- core/: Domain models and business logic
- fetch/: Article fetching and extraction
- summarize/: AI summarization and tracing
- input/: Input parsing (JSON, future: RSS, OPML)
- output/: Rendering and templates
- utils/: Shared utilities

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 8: Update Documentation

**Step 1: Update README.md with new import examples**

If README.md shows any import examples, update them to reflect the new structure.

**Step 2: Update CLAUDE.md if it exists**

Check for and update any code structure references.

**Step 3: Commit documentation updates**

```bash
git add README.md CLAUDE.md docs/
git commit -m "docs: update import examples after reorganization

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Verification Checklist

After completing all tasks, verify:

- [ ] All 18 tests pass
- [ ] `daily-feed run --help` works
- [ ] No old import patterns remain in codebase
- [ ] Directory structure matches design
- [ ] Documentation updated
- [ ] Can run full pipeline on test data

---

## Rollback Instructions

If you need to rollback at any point:

```bash
# To rollback specific task commits
git revert HEAD~N..HEAD  # where N is number of tasks to rollback

# Or reset to design commit
git reset --hard 69bcfd8  # The design document commit
```
