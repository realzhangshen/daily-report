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

    output_path = run_pipeline(input, output, cfg, show_progress=progress, console=console)
    console.print(f"Report generated: {output_path}")


if __name__ == "__main__":
    app()
