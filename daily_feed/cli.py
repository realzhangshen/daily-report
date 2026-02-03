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

    output_path = run_pipeline(input, output, cfg)
    console.print(f"Report generated: {output_path}")


if __name__ == "__main__":
    app()
