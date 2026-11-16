"""Command-line interface: ``uv run nflsim --help``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from nflsim.config import load_settings
from nflsim.data import DATASETS, DataCache, parse_seasons

app = typer.Typer(help="Monte Carlo NFL game simulator.", no_args_is_help=True)

ConfigOption = Annotated[
    Path | None, typer.Option("--config", help="TOML config file (defaults to packaged config).")
]


@app.command("datasets")
def list_datasets() -> None:
    """List the datasets that can be fetched."""
    for spec in DATASETS.values():
        since = f"{spec.first_season}+" if spec.seasonal else "all"
        typer.echo(f"{spec.name:<16} {since:<6} {spec.description}")


@app.command()
def fetch(
    seasons: Annotated[
        str | None,
        typer.Option(
            help="Seasons, e.g. '2016-2025' or '2023,2024'. Default: config start to now."
        ),
    ] = None,
    datasets: Annotated[
        str | None, typer.Option(help="Comma-separated dataset names. Default: all.")
    ] = None,
    refresh: Annotated[bool, typer.Option(help="Re-download even if cached.")] = False,
    config: ConfigOption = None,
) -> None:
    """Download nflverse data and cache it as parquet under the cache directory."""
    settings = load_settings(config)
    cache = DataCache(settings.data)

    names = [n.strip() for n in datasets.split(",")] if datasets else list(cache.datasets)
    wanted = (
        parse_seasons(seasons)
        if seasons
        else list(range(settings.data.default_first_season, cache.current_season + 1))
    )

    for name in names:
        spec = cache.spec(name)
        if spec.seasonal:
            usable = [s for s in wanted if s >= (spec.first_season or 0)]
            skipped = sorted(set(wanted) - set(usable))
            if skipped:
                typer.echo(f"{name}: skipping {skipped} (data starts {spec.first_season})")
            if not usable:
                continue
            frame = cache.load(name, usable, refresh=refresh)
        else:
            frame = cache.load(name, refresh=refresh)
        typer.echo(f"{name}: {frame.height:,} rows cached")


if __name__ == "__main__":
    app()
