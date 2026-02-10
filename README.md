# Daily Feed Agent

Analyze daily RSS exports into a clean HTML report using a pluggable AI provider (Gemini by default).

## Install (uv)

```bash
uv venv
uv pip install -e .
```

## Run

```bash
cp .env.example .env
# edit .env to add provider API key and CRAWL4AI_API_URL

daily-feed --input data/folo-export-2026-02-03.json --output out --config config.example.yaml
# disable cache for a full fresh run
daily-feed --input data/folo-export-2026-02-03.json --output out --config config.example.yaml --no-use-cache
```

## Rebucket Exports By Day (22:00 Cutoff)

When you have multiple raw exports under `data/`, you can merge/deduplicate
and split them into per-day feeds using a local 22:00 boundary:

```bash
/usr/bin/python3 tools/rebucket_data.py \
  --input-dir data \
  --output-dir data/daily_feeds \
  --cutoff 22:00
```

Generated files:

```text
data/daily_feeds/feed-YYYY-MM-DD.json
data/daily_feeds/manifest.json
```

Output will be written to a subfolder under `out/` named after the input file stem, for example:

```text
out/folo-export-2026-02-03/report.html
out/folo-export-2026-02-03/articles/
```

You can change the subfolder strategy with `--run-folder-mode` or config `output.run_folder_mode`:

```text
input            -> out/<input-stem>/
timestamp        -> out/YYYYMMDD-HHMMSS-<input-stem>/
input_timestamp  -> out/<input-stem>-YYYYMMDD-HHMMSS/
```

> **Note:** This project requires a remote Crawl4AI API service for fetching articles. Set the `CRAWL4AI_API_URL` environment variable to point to your Crawl4AI API server.

## Cache Structure

Articles are cached in per-entry folders under `articles/`:

```
out/run-20260205/
├── report.html
├── run.jsonl              # Pipeline-level events
└── articles/              # Per-article cache
    ├── article-one-a1b2c/
    │   ├── extracted.txt      # Extracted text/markdown
    │   ├── analysis.txt       # LLM analysis text
    │   ├── analysis.json      # Analysis metadata
    │   ├── llm_events.jsonl   # LLM call logs
    │   └── entry_events.jsonl # Entry-level workflow logs
    └── article-two-c3d4e/
        └── ...
```

Each entry folder is named `{slug}-{shortHash}` where:
- `slug`: URL-safe version of article title
- `shortHash`: First 5 chars of MD5 hash of URL

## Config

```yaml
provider:
  name: gemini
  model: gemini-3-flash-preview
  api_key_env: GOOGLE_API_KEY
  base_url: https://generativelanguage.googleapis.com
  # api_key: ""  # optional inline override
  trust_env: true
fetch:
  timeout_seconds: 20
  retries: 2
  trust_env: true
  user_agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36
  # Anti-bot detection options
  crawl4ai_stealth: true  # Enable stealth mode to bypass bot detection
  crawl4ai_delay: 2.0  # Delay before returning HTML (allows challenges to complete)
  crawl4ai_simulate_user: true  # Simulate user behavior for anti-bot
  crawl4ai_magic: true  # Enable anti-detection "magic" mode
  # Remote Crawl4AI API (required - use environment variable for sensitive URLs)
  crawl4ai_api_url: null  # Uses CRAWL4AI_API_URL from environment
dedup:
  enabled: true
  title_similarity_threshold: 92
summary:
  bullets_min: 3
  bullets_max: 6
  max_chars: 12000
  analysis_max_output_tokens: 1200
  analysis_concurrency: 1
  enable_deep_fetch_decision: true
grouping:
  method: provider
  fallback: site
output:
  format: html
  include_markdown: false
logging:
  level: INFO
  console: true
  file: true
  format: jsonl
  filename: run.jsonl
  llm_log_enabled: true
  llm_log_detail: response_only
  llm_log_redaction: redact_urls_authors
  llm_log_file: llm.jsonl
cache:
  enabled: true
  ttl_days: null
langfuse:
  enabled: false
  public_key: null
  secret_key: null
  host: null
  environment: null
  release: null
  timeout_seconds: 30
  redaction: redact_urls_authors
  max_text_chars: 20000
```

Provider is hot-swappable:

```yaml
# Gemini
provider:
  name: gemini
  model: gemini-3-flash-preview
  api_key_env: GOOGLE_API_KEY
  base_url: https://generativelanguage.googleapis.com

# OpenAI-compatible (OpenAI, OpenRouter, self-hosted gateways, etc.)
provider:
  name: openai
  model: gpt-4.1-mini
  api_key_env: OPENAI_API_KEY
  base_url: https://api.openai.com/v1
```

## Langfuse Tracing (Prompt/Workflow Iteration)

This project can emit traces/spans to Langfuse to help iterate on prompts
and workflow steps. Enable tracing in config and set the appropriate Langfuse
environment variables (or inline config).

**IMPORTANT:** For self-hosted Langfuse, use environment variables instead of hardcoding sensitive URLs in config.yaml.

Example environment variables:

```bash
export LANGFUSE_PUBLIC_KEY="pk_..."
export LANGFUSE_SECRET_KEY="sk_..."
export LANGFUSE_BASE_URL="https://langfuse.your-domain.com"
# export LANGFUSE_ENVIRONMENT="local"
# export LANGFUSE_RELEASE="daily-feed"
```

Enable in config:

```yaml
langfuse:
  enabled: true
  host: null  # Uses LANGFUSE_BASE_URL from environment
  environment: daily-feed
```
