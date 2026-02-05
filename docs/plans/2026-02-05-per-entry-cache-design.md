# Per-Entry Cache and Logging Design

**Date**: 2026-02-05
**Status**: Design Approved

## Overview

Restructure cache and logging to be organized per-entry (per article) instead of a flat cache directory with global logs.

## Folder Structure

```
out/
└── folo-export-2026-02-03/                    # run folder
    ├── report.html                            # final report
    ├── run.jsonl                              # global pipeline events
    └── articles/                              # NEW: articles directory
        ├── openai-codex-app-a1b2c/           # entry folder (slug-shortHash)
        │   ├── fetched.html                   # raw HTML from fetch
        │   ├── extracted.txt                  # extracted text content
        │   ├── llm_summary.json               # LLM summary result
        │   └── llm_debug.jsonl                # LLM interaction logs
        ├── vm0-ai-vm0-d3e4f/
        │   ├── fetched.html
        │   ├── extracted.txt
        │   ├── llm_summary.json
        │   └── llm_debug.jsonl
        └── ...
```

## Entry Folder Naming

Each entry folder is named `{slug}-{shortHash}`:
- **slug**: URL-safe slug from article title (lowercase, hyphens)
- **shortHash**: First 5 characters of MD5 hash of the URL

Example: `openai-codex-app-a1b2c`

## File Formats

### `fetched.html`
Raw HTML content from the fetch stage.

### `extracted.txt`
Extracted plain text content from the extractor stage.

### `llm_summary.json`
LLM-generated summary result:

```json
{
  "bullets": [
    "Bullet point 1",
    "Bullet point 2"
  ],
  "takeaway": "One-sentence takeaway.",
  "topic": "AI Coding Tools",
  "status": "ok",
  "model": "gemini-3-flash-preview",
  "generated_at": "2026-02-05T12:00:00Z",
  "article": {
    "title": "Article Title",
    "url": "https://example.com/article",
    "site": "Example Site"
  }
}
```

### `llm_debug.jsonl`
LLM interaction debug logs (JSONL format):

```jsonl
{"timestamp": "...", "event": "llm_request", "url": "...", "prompt": "..."}
{"timestamp": "...", "event": "llm_response", "status": "ok", "model": "...", "raw_response": "..."}
```

### `run.jsonl` (Global)
Pipeline-level events:

```jsonl
{"timestamp": "...", "event": "pipeline_start", "input": "...", "output": "..."}
{"timestamp": "...", "event": "fetch_start", "url": "...", "title": "..."}
{"timestamp": "...", "event": "pipeline_complete", "total": 42}
```

## Implementation Changes

### Modules to Modify

| Module | Changes |
|--------|---------|
| `cache.py` | Rename/refactor to `entry_manager.py` |
| `runner.py` | Update cache calls to use EntryManager |
| `fetcher.py` | Update cache_path to use entry folders |
| `logging_utils.py` | Add per-entry logger setup |
| `providers/gemini.py` | Write LLM logs to per-entry |

### New EntryManager Class

```python
class EntryManager:
    """Manages entry folder, cache files, and logs for a single article"""

    def __init__(self, articles_dir: Path, article: Article):
        self.articles_dir = articles_dir
        self.article = article
        self.folder = self._entry_folder()

    def _entry_folder(self) -> Path:
        """Generate entry folder path: {slug}-{shortHash}/"""
        slug = slugify(self.article.title)
        short_hash = hashlib.md5(self.article.url.encode()).hexdigest()[:5]
        return self.articles_dir / f"{slug}-{short_hash}"

    @property
    def fetched_html(self) -> Path:
        return self.folder / "fetched.html"

    @property
    def extracted_txt(self) -> Path:
        return self.folder / "extracted.txt"

    @property
    def llm_summary(self) -> Path:
        return self.folder / "llm_summary.json"

    @property
    def llm_debug(self) -> Path:
        return self.folder / "llm_debug.jsonl"

    def ensure_folder(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
```

## Cache Strategy

### Cache Validation
TTL validation applies to the entire entry folder:

```python
def is_entry_valid(folder: Path, ttl_days: int | None) -> bool:
    """Check if entry cache is still valid"""
    if not folder.exists():
        return False
    if ttl_days is None:
        return True
    age_seconds = (datetime.now() - datetime.fromtimestamp(folder.stat().st_mtime)).total_seconds()
    return age_seconds <= ttl_days * 86400
```

### Cache Read Priority
1. `llm_summary.json` exists and valid → use directly
2. `extracted.txt` exists and valid → skip fetch, do LLM
3. `fetched.html` exists and valid → skip fetch, do extract
4. Otherwise → fetch from scratch

## Logging Separation

### Global Logs (`run.jsonl`)
- Pipeline-level events
- Aggregated statistics

### Per-Entry Logs (`llm_debug.jsonl`)
- LLM interactions specific to this entry
- Detailed fetch/extract events

## Backward Compatibility

- Config option to enable/disable per-entry mode (default: new mode)
- Existing `cache/` directory structure still supported via config
