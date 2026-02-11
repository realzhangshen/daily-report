"""
Command-line interface for the Daily Report Agent.

Uses Typer to provide a CLI with options for all major configuration
settings. Supports loading .env files for API key configuration.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .config import load_config
from .llm.tracing import flush
from .runner import run_pipeline

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def run(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True),
    output: Path = typer.Option(Path("out"), "--output", "-o"),
    config: Path | None = typer.Option(Path("config.yaml"), "--config", "-c", exists=True),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
    run_folder_mode: str | None = typer.Option(
        None,
        "--run-folder-mode",
        help="Output subfolder mode: input, timestamp, or input_timestamp.",
    ),
    log_level: str | None = typer.Option(None, "--log-level", help="Logging level."),
    log_format: str | None = typer.Option(
        None, "--log-format", help="Log file format: jsonl or plain."
    ),
    log_file: bool | None = typer.Option(
        None, "--log-file/--no-log-file", help="Enable or disable file logging."
    ),
    cache_ttl_days: int | None = typer.Option(
        None, "--cache-ttl-days", help="Cache TTL in days."
    ),
    use_cache: bool | None = typer.Option(
        None,
        "--use-cache/--no-use-cache",
        help="Enable or disable cache usage for fetch/analyze stages.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="GOOGLE_API_KEY",
        help="Override provider API key (or set GOOGLE_API_KEY / .env).",
    ),
):
    """Run the daily report pipeline.

    Processes a JSON RSS export file, fetches article content,
    generates AI summaries, groups by topic, and outputs an HTML report.

    Args:
        input: Path to input JSON file (Folo format)
        output: Directory for output reports
        config: Optional path to YAML config file
        progress: Whether to show progress bar
        run_folder_mode: Output folder naming strategy
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Log file format (jsonl, plain)
        log_file: Enable/disable file logging
        cache_ttl_days: Cache entry time-to-live
        use_cache: Whether to use cache for fetch/analyze
        api_key: Override LLM provider API key
    """
    # Load environment variables from .env if available
    if load_dotenv is not None:
        load_dotenv()

    # Load base configuration
    cfg = load_config(str(config) if config else None)

    # Override with CLI options
    if api_key:
        cfg.provider.api_key = api_key
    if run_folder_mode:
        cfg.output.run_folder_mode = run_folder_mode
    if log_level:
        cfg.logging.level = log_level
    if log_format:
        cfg.logging.format = log_format
    if log_file is not None:
        cfg.logging.file = log_file
    if cache_ttl_days is not None:
        cfg.cache.ttl_days = cache_ttl_days
    if use_cache is not None:
        cfg.cache.enabled = use_cache

    # Run the pipeline
    output_path = run_pipeline(input, output, cfg, show_progress=progress, console=console)
    console.print(f"Report generated: {output_path}")

    # Flush Langfuse traces before exit
    flush()


if __name__ == "__main__":
    app()
