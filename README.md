# Daily Feed Agent

Summarize daily RSS exports into a clean HTML report using a pluggable AI provider (Gemini by default).

## Install (uv)

```bash
uv venv
uv pip install -e .
```

### Optional: Crawl4AI Web Fetch

If you want JS-rendered fetching, install Crawl4AI:

```bash
pip install -U crawl4ai
playwright install
```

## Run

```bash
cp .env.example .env
# edit .env to add GOOGLE_API_KEY

daily-feed run --input data/folo-export-2026-02-03.md --output out --config config.example.yaml
```

Output will be written to a subfolder under `out/` named after the input file stem, for example:

```text
out/folo-export-2026-02-03/report.html
out/folo-export-2026-02-03/cache/
```

You can change the subfolder strategy with `--run-folder-mode` or config `output.run_folder_mode`:

```text
input            -> out/<input-stem>/
timestamp        -> out/YYYYMMDD-HHMMSS-<input-stem>/
input_timestamp  -> out/<input-stem>-YYYYMMDD-HHMMSS/
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
  backend: httpx  # httpx or crawl4ai
  fallback_to_httpx: true
  crawl4ai_concurrency: 4
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
