from __future__ import annotations

from pathlib import Path
import os

import typer
from rich.console import Console

from .config import load_config
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
    output: Path = typer.Option(..., "--output", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
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
    cache_mode: str | None = typer.Option(
        None, "--cache-mode", help="Cache mode: run or shared."
    ),
    cache_ttl_days: int | None = typer.Option(
        None, "--cache-ttl-days", help="Cache TTL in days."
    ),
    cache_shared_dir: Path | None = typer.Option(
        None, "--cache-shared-dir", help="Shared cache directory."
    ),
    cache_index: bool | None = typer.Option(
        None, "--cache-index/--no-cache-index", help="Enable or disable cache index."
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="GOOGLE_API_KEY",
        help="Override provider API key (or set GOOGLE_API_KEY / .env).",
    ),
):
    """Run the daily feed pipeline."""
    if load_dotenv is not None:
        load_dotenv()

    cfg = load_config(str(config) if config else None)
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
    if cache_mode:
        cfg.cache.mode = cache_mode
    if cache_ttl_days is not None:
        cfg.cache.ttl_days = cache_ttl_days
    if cache_shared_dir is not None:
        cfg.cache.shared_dir = str(cache_shared_dir)
    if cache_index is not None:
        cfg.cache.write_index = cache_index

    output_path = run_pipeline(input, output, cfg, show_progress=progress, console=console)
    console.print(f"Report generated: {output_path}")


if __name__ == "__main__":
    app()
