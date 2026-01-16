from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from clipgen.config import load_config
from clipgen.pipeline import analyze, render, render_from_plan

app = typer.Typer(add_completion=False)
console = Console()


def _resolve_out(out: Optional[Path]) -> Path:
    return out or Path.cwd() / "clipgen_output"


@app.command()
def analyze_command(
    source: str = typer.Argument(..., help="Local file path or direct-download URL"),
    config_path: Path = typer.Option(..., "--config", exists=True, help="Config YAML"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output directory"),
    header: List[str] = typer.Option(None, "--header", help="Request header: 'Key: Value'"),
) -> None:
    config = load_config(config_path)
    out_dir = _resolve_out(out)
    headers = header or []
    console.print(f"[bold]Analyzing[/bold] {source}")
    paths = analyze(source, config, out_dir, headers)
    console.print("Analysis complete.")
    for name, path in paths.items():
        console.print(f" - {name}: {path}")


@app.command()
def render_command(
    source: str = typer.Argument(..., help="Local file path or direct-download URL"),
    config_path: Path = typer.Option(..., "--config", exists=True, help="Config YAML"),
    bundle: Path = typer.Option(..., "--bundle", exists=True, help="analysis/export_plan.json"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output directory"),
    header: List[str] = typer.Option(None, "--header", help="Request header: 'Key: Value'"),
) -> None:
    config = load_config(config_path)
    out_dir = _resolve_out(out)
    headers = header or []
    console.print(f"[bold]Rendering[/bold] using {bundle}")
    render_from_plan(source, config, out_dir, headers, bundle)
    console.print("Render complete.")


@app.command()
def all(
    source: str = typer.Argument(..., help="Local file path or direct-download URL"),
    config_path: Path = typer.Option(..., "--config", exists=True, help="Config YAML"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output directory"),
    header: List[str] = typer.Option(None, "--header", help="Request header: 'Key: Value'"),
    device: str = typer.Option("cpu", "--device", help="cpu or cuda"),
) -> None:
    _ = device
    config = load_config(config_path)
    out_dir = _resolve_out(out)
    headers = header or []
    console.print(f"[bold]Running full pipeline[/bold] for {source}")
    render(source, config, out_dir, headers)
    console.print("All clips rendered.")


if __name__ == "__main__":
    app()
