# Daily Feed Agent

Summarize daily RSS exports into a clean HTML report using a pluggable AI provider (Gemini by default).

## Install (uv)

```bash
uv venv
uv pip install -e .
```

## Run

```bash
cp .env.example .env
# edit .env to add GOOGLE_API_KEY

daily-feed run --input data/folo-export-2026-02-03.md --output out --config config.example.yaml
```

## Config

```yaml
provider:
  name: gemini
  model: gemini-3-flash-preview
  google_api_key_env: GOOGLE_API_KEY
  base_url: https://generativelanguage.googleapis.com
  # api_key: ""  # optional inline override
  trust_env: true
fetch:
  timeout_seconds: 20
  retries: 2
  trust_env: true
  user_agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36
extract:
  primary: trafilatura
  fallback: [readability, bs4]
dedup:
  enabled: true
  title_similarity_threshold: 92
summary:
  bullets_min: 3
  bullets_max: 6
  max_chars: 12000
grouping:
  method: provider
  fallback: site
output:
  format: html
  include_markdown: false
```
