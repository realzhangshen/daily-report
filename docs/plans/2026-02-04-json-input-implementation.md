# JSON Input Format Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the markdown-based input parser with JSON format to match the Folo export format, requiring changes to the Article type, parser implementation, and CLI interface.

**Architecture:** Extend the existing Article dataclass with new JSON-specific fields (id, published_at, inserted_at), create a new JSON parser that converts Folo JSON exports to Article objects, delete the markdown parser, and update the CLI to reference JSON input. The downstream pipeline (fetcher, extractor, dedup, summary, renderer) remains unchanged as it consumes the Article type.

**Tech Stack:** Python 3.13+, dataclasses, JSON, click CLI

---

### Task 1: Extend Article Type

**Files:**
- Modify: `daily_feed/types.py`

**Step 1: Read current Article type**

Read the file to understand the current structure:

```bash
cat daily_feed/types.py
```

**Step 2: Add new JSON fields to Article dataclass**

Add `id`, `published_at`, and `inserted_at` fields to the Article dataclass. These fields should be optional (default to None) for backward compatibility with any code that creates Article objects directly.

Edit `daily_feed/types.py`:

```python
@dataclass
class Article:
    title: str
    site: str
    url: str
    category: str | None = None
    author: str | None = None
    time: str | None = None
    summary: str | None = None
    # New fields from JSON format
    id: str | None = None
    published_at: str | None = None  # ISO 8601 format
    inserted_at: str | None = None  # ISO 8601 format
```

**Step 3: Verify the code is valid**

Run Python syntax check:

```bash
python -m py_compile daily_feed/types.py
```

Expected: No output (success)

**Step 4: Test CLI still works**

Verify the extension doesn't break existing code:

```bash
daily-feed --help
```

Expected: Help text displays successfully

**Step 5: Commit**

```bash
git add daily_feed/types.py
git commit -m "feat: extend Article type with JSON fields

Add id, published_at, inserted_at fields to support JSON input format.
These fields are optional to maintain backward compatibility.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Create JSON Parser

**Files:**
- Create: `daily_feed/json_parser.py`

**Step 1: Create the JSON parser module**

Create `daily_feed/json_parser.py` with the following implementation:

```python
"""JSON parser for Folo RSS export format.

This module parses RSS feed exports in Folo's JSON format into structured
Article objects. The format uses:
- Top-level metadata (exportTime, total)
- Articles array with id, title, url, publishedAt, insertedAt, summary, feedTitle, category
"""

from __future__ import annotations

import logging
from typing import Any

from urllib.parse import urlparse

from .types import Article

logger = logging.getLogger(__name__)


def parse_folo_json(data: dict[str, Any]) -> list[Article]:
    """Parse Folo JSON format into a list of Article objects.

    The Folo JSON format structure:
        {
            "exportTime": "2026-02-03T13:22:16.502Z",
            "total": 41,
            "articles": [
                {
                    "id": "241476308963169281",
                    "title": "Article Title",
                    "url": "https://example.com/article",
                    "publishedAt": "2026-02-03T11:44:10.702Z",
                    "insertedAt": "2026-02-03T11:44:11.423Z",
                    "summary": "Brief summary or empty string",
                    "feedTitle": "Twitter @OpenAI",
                    "category": "Twitter"
                }
            ]
        }

    Args:
        data: The parsed JSON content as a dictionary

    Returns:
        A list of Article objects parsed from the JSON. Articles with
        missing required fields (title, url) are skipped with a warning.

    Raises:
        ValueError: If the JSON is missing the 'articles' key
    """
    if "articles" not in data:
        raise ValueError("Invalid JSON format: missing 'articles' key")

    articles: list[Article] = []

    for item in data["articles"]:
        # Required fields validation
        title = item.get("title")
        url = item.get("url")

        if not title or not url:
            article_id = item.get("id", "unknown")
            logger.warning(f"Skipping article {article_id}: missing required fields (title or url)")
            continue

        # Parse feedTitle into site and author
        feed_title = item.get("feedTitle", "")
        if not feed_title:
            # Fallback: extract domain from URL
            domain = urlparse(url).netloc
            feed_title = domain

        site, author = _parse_feed_title(feed_title)

        # Convert empty summary to None
        summary = item.get("summary")
        if summary == "":
            summary = None

        article = Article(
            id=item.get("id"),
            title=title,
            url=url,
            site=site,
            author=author,
            category=item.get("category"),
            summary=summary,
            published_at=item.get("publishedAt"),
            inserted_at=item.get("insertedAt"),
        )
        articles.append(article)

    return articles


def _parse_feed_title(feed_title: str) -> tuple[str, str | None]:
    """Parse the feedTitle field into site and optional author.

    The feedTitle field format is "Site Name @Author". The @author part
    is optional.

    Args:
        feed_title: The raw feedTitle string from the JSON

    Returns:
        A tuple of (site, author) where author may be None

    Examples:
        >>> _parse_feed_title("Twitter @OpenAI")
        ("Twitter", "OpenAI")
        >>> _parse_feed_title("Product Hunt")
        ("Product Hunt", None)
    """
    if "@" in feed_title:
        parts = feed_title.split("@", 1)
        site = parts[0].strip()
        author = parts[1].strip() if parts[1].strip() else None
        return site, author
    return feed_title.strip(), None
```

**Step 2: Verify the code is valid**

Run Python syntax check:

```bash
python -m py_compile daily_feed/json_parser.py
```

Expected: No output (success)

**Step 3: Commit**

```bash
git add daily_feed/json_parser.py
git commit -m "feat: add JSON parser for Folo export format

Implement parse_folo_json() to convert Folo JSON exports to Article objects.
Handles feedTitle parsing (site@author), empty summaries, and missing fields
with appropriate warnings and fallbacks.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Update Module Imports

**Files:**
- Modify: `daily_feed/__init__.py`

**Step 1: Read current __init__.py**

```bash
cat daily_feed/__init__.py
```

**Step 2: Update imports to use JSON parser**

Replace the markdown parser import with the JSON parser import.

Edit `daily_feed/__init__.py`:

Find the line importing `parse_folo_markdown` and replace it with:
```python
from .json_parser import parse_folo_json
```

**Step 3: Verify the code is valid**

```bash
python -c "from daily_feed import parse_folo_json; print('Import successful')"
```

Expected: Output "Import successful"

**Step 4: Commit**

```bash
git add daily_feed/__init__.py
git commit -m "refactor: import JSON parser instead of markdown parser

Update module imports to use parse_folo_json from json_parser module.
The markdown parser will be removed in a subsequent commit.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Update Runner to Use JSON

**Files:**
- Modify: `daily_feed/runner.py`

**Step 1: Read current runner.py**

```bash
cat daily_feed/runner.py
```

**Step 2: Locate the input file reading section**

Find where the markdown file is read and parsed. Look for the section that:
1. Opens the input file
2. Calls `parse_folo_markdown()` or reads the file as text

**Step 3: Update to use JSON loading**

Replace the markdown file reading with JSON loading. The change should:
1. Use `json.load()` to parse the file
2. Call `parse_folo_json()` instead of `parse_folo_markdown()`
3. Add `import json` at the top of the file if not already present

Edit `daily_feed/runner.py`:

Add import at top if needed:
```python
import json
```

Replace the file reading section with:
```python
# Load and parse JSON input file
with open(input_path) as f:
    data = json.load(f)

articles = parse_folo_json(data)
```

**Step 4: Verify the code is valid**

```bash
python -m py_compile daily_feed/runner.py
```

Expected: No output (success)

**Step 5: Test with JSON file**

Test with the reference JSON file:

```bash
daily-feed run --input data/folo-export-2026-02-03-21-22.json --output out/test
```

Expected: Pipeline runs without errors, generates HTML report

**Step 6: Commit**

```bash
git add daily_feed/runner.py
git commit -m "refactor: use JSON input loading in runner

Replace markdown file reading with JSON loading using json.load().
Call parse_folo_json() instead of the markdown parser.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 5: Update CLI Help Text

**Files:**
- Modify: `daily_feed/cli.py`

**Step 1: Read current cli.py**

```bash
cat daily_feed/cli.py
```

**Step 2: Find and update help text references**

Locate any references to "markdown" in help text and update to reference "JSON":

1. The `--input` option help text
2. Any command description text
3. Any docstring references

Edit `daily_feed/cli.py`:

Find and replace references like:
- "Path to input markdown file" → "Path to input JSON file"
- "Folo markdown format" → "Folo JSON export format"
- "Processes a markdown RSS export" → "Processes a JSON RSS export"

**Step 3: Verify CLI help displays correctly**

```bash
daily-feed --help
```

Expected: Help text shows "JSON" references instead of "markdown"

**Step 4: Commit**

```bash
git add daily_feed/cli.py
git commit -m "docs: update CLI help text to reference JSON input

Replace markdown references with JSON in CLI help text and descriptions.
Reflects the new JSON-based input format.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 6: Delete Markdown Parser

**Files:**
- Delete: `daily_feed/parser.py`

**Step 1: Verify nothing imports parser module**

Check for any remaining imports of the old parser:

```bash
grep -r "parse_folo_markdown\|from .parser import\|from daily_feed.parser" daily_feed/
```

Expected: No results (if there are results, update those files first)

**Step 2: Delete the markdown parser file**

```bash
rm daily_feed/parser.py
```

**Step 3: Verify the codebase still works**

```bash
python -c "from daily_feed import parse_folo_json; print('OK')"
daily-feed --help
```

Expected: Both commands succeed

**Step 4: Test full pipeline with JSON input**

```bash
daily-feed run --input data/folo-export-2026-02-03-21-22.json --output out/test-json
```

Expected: Complete pipeline runs successfully

**Step 5: Commit**

```bash
git add daily_feed/parser.py
git commit -m "refactor: remove markdown parser

Delete the markdown parser module as it is no longer needed.
All input parsing now uses the JSON format.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 7: Update README

**Files:**
- Modify: `README.md`

**Step 1: Read current README.md**

```bash
cat README.md
```

**Step 2: Update references from markdown to JSON**

Find and update:
1. Input file format references (markdown → JSON)
2. Example commands (use .json extension)
3. Any format descriptions

Edit `README.md`:

Update relevant sections to reference JSON input format and update example commands like:
```bash
daily-feed run --input data/folo-export-2026-02-03.json --output out --config config.example.yaml
```

**Step 3: Verify documentation is consistent**

Ensure all references consistently point to JSON format, not markdown.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README for JSON input format

Update documentation to reference JSON input format instead of markdown.
Update example commands to use .json file extensions.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 8: Final Verification and Cleanup

**Files:**
- No specific files (verification task)

**Step 1: Run full end-to-end test**

```bash
daily-feed run --input data/folo-export-2026-02-03-21-22.json --output out/final-test
```

Expected: Complete pipeline runs, generates HTML report

**Step 2: Verify output contains expected articles**

Check that articles were parsed correctly:

```bash
# Count articles in JSON vs output
python -c "
import json
with open('data/folo-export-2026-02-03-21-22.json') as f:
    data = json.load(f)
    print(f'JSON input: {len(data[\"articles\"])} articles')
"
```

Expected: Shows "41 articles"

**Step 3: Check for any remaining markdown references**

```bash
grep -r "markdown\|\.md" daily_feed/ --include="*.py" | grep -v ".egg-info" | head -20
```

Expected: No results related to input parsing (only comments/ok)

**Step 4: Verify git status**

```bash
git status
```

Expected: Only the design document in docs/plans/ should be untracked (if anything)

**Step 5: Final commit if needed**

If any final tweaks were needed:

```bash
git add -A
git commit -m "chore: final cleanup after JSON migration

Complete the migration from markdown to JSON input format.
All components updated and tested.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

After completing all tasks:
- Article type extended with JSON fields (id, published_at, inserted_at)
- New JSON parser handles Folo export format with proper error handling
- Markdown parser removed
- CLI and documentation updated
- Full pipeline tested with JSON input

Total estimated time: 45-60 minutes
