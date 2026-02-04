# JSON Input Format Migration Design

**Date:** 2026-02-04
**Status:** Approved
**Author:** Daily Feed Agent

## Overview

Replace the current markdown-based input parser with JSON format to match the Folo export format. This is a complete replacement—no backward compatibility with markdown.

## JSON Format Reference

```json
{
  "exportTime": "2026-02-03T13:22:16.502Z",
  "exportTimeFormatted": "2/3/2026, 9:22:16 PM",
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
```

## Changes Required

### 1. Article Type Extension (`daily_feed/types.py`)

Add new fields to the `Article` dataclass:

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
    # New fields from JSON
    id: str | None = None
    published_at: str | None = None  # ISO 8601
    inserted_at: str | None = None  # ISO 8601
```

### 2. New JSON Parser (`daily_feed/json_parser.py`)

Create new parser to replace `parser.py`:

```python
def parse_folo_json(data: dict) -> list[Article]:
    """Parse Folo JSON export format into Article objects."""
    if "articles" not in data:
        raise ValueError("Invalid JSON format: missing 'articles' key")

    articles = []
    for item in data["articles"]:
        # Required fields validation
        if not item.get("title") or not item.get("url"):
            logger.warning(f"Skipping article missing required fields: {item.get('id')}")
            continue

        # feedTitle fallback to domain extraction
        feed_title = item.get("feedTitle", "")
        if not feed_title:
            from urllib.parse import urlparse
            domain = urlparse(item["url"]).netloc
            feed_title = domain

        site, author = _parse_feed_title(feed_title)

        article = Article(
            id=item.get("id"),
            title=item.get("title"),
            url=item.get("url"),
            site=site,
            author=author,
            category=item.get("category"),
            summary=item.get("summary") or None,
            published_at=item.get("publishedAt"),
            inserted_at=item.get("insertedAt"),
        )
        articles.append(article)
    return articles

def _parse_feed_title(feed_title: str) -> tuple[str, str | None]:
    """Parse feedTitle (e.g., 'Twitter @OpenAI') into site and author."""
    if "@" in feed_title:
        parts = feed_title.split("@", 1)
        return parts[0].strip(), parts[1].strip()
    return feed_title.strip(), None
```

### 3. Module Updates

| File | Change |
|------|--------|
| `daily_feed/__init__.py` | Import `parse_folo_json` instead of `parse_folo_markdown` |
| `daily_feed/cli.py` | Update help text to reference JSON input |
| `daily_feed/runner.py` | Use `json.load()` and call `parse_folo_json()` |
| `daily_feed/parser.py` | **DELETE** - markdown parser no longer needed |

### 4. CLI Update

```python
@click.option("--input", "-i", type=click.Path(exists=True), required=True,
    help="Input JSON file path (Folo export format)")
```

## Field Mapping

| JSON Field | Article Field | Notes |
|------------|---------------|-------|
| id | id | New field |
| title | title | Same |
| url | url | Same |
| publishedAt | published_at | ISO 8601 format |
| insertedAt | inserted_at | ISO 8601 format |
| summary | summary | Empty string → None |
| feedTitle | site + author | Parsed: "Twitter @OpenAI" → site="Twitter", author="OpenAI" |
| category | category | Same |

## Error Handling

- Missing `articles` key → explicit `ValueError`
- Missing required fields (title/url) → skip with warning
- Missing `feedTitle` → fallback to URL domain
- Empty `summary` → convert to `None`
- Missing optional fields → use defaults

## Testing

Test with reference file `data/folo-export-2026-02-03-21-22.json`:

```bash
daily-feed run --input data/folo-export-2026-02-03-21-22.json --output out/test
```

Verify:
- All 41 articles are parsed
- `feedTitle` "Twitter @OpenAI" → site="Twitter", author="OpenAI"
- `feedTitle` "Product Hunt — The best new products" → site="Product Hunt...", author=None
- Empty summaries handled correctly
- ISO timestamps preserved

## Impact

- **No changes needed** to: fetcher, extractor, dedup, summary, renderer, config
- These modules consume `Article` type, which is backward compatible
- Only input parsing changes
