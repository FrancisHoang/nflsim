"""Command-line interface: ``uv run nflsim --help``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from nflsim.backtest import calibration, format_table, summarize, walk_forward
from nflsim.config import load_settings
from nflsim.data import (
    DATASETS,
    PBP_PLAYER_COLUMNS,
    DataCache,
    id_coverage,
    parse_seasons,
    player_snaps,
    roster_weeks,
    unmatched_snaps,
)
from nflsim.pipeline import load_games, matchup_row, predict_matchup
from nflsim.tuning import current_params, evaluate, evaluate_holdout, grid

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


@app.command()
def backtest(
    seasons: Annotated[str, typer.Option(help="Test seasons, e.g. '2021-2024'.")] = "2021-2024",
    n: Annotated[int, typer.Option(help="Simulations per game.", min=1)] = 10_000,
    seed: Annotated[int, typer.Option(help="RNG seed (season s uses seed + s).")] = 42,
    features: Annotated[
        str | None,
        typer.Option(help="Comma-separated points-model features (default: from config)."),
    ] = None,
    margin_fit: Annotated[
        str | None,
        typer.Option(help="How the margin is fit: 'points' or 'direct' (default: config)."),
    ] = None,
    output: Annotated[
        Path | None, typer.Option(help="Write per-game predictions to this parquet file.")
    ] = None,
    config: ConfigOption = None,
) -> None:
    """Walk-forward backtest: fit on earlier seasons, predict each test season."""
    settings = load_settings(config)
    overrides: dict[str, object] = {}
    if features is not None:
        overrides["features"] = tuple(f.strip() for f in features.split(",") if f.strip())
    if margin_fit is not None:
        overrides["margin_fit"] = margin_fit
    if overrides:
        model_cfg = settings.model.model_validate({**settings.model.model_dump(), **overrides})
        settings = settings.model_copy(update={"model": model_cfg})
    test_seasons = parse_seasons(seasons)
    games = load_games(DataCache(settings.data), settings, max(test_seasons))
    predictions, fits = walk_forward(games, settings, test_seasons, n, seed)

    typer.echo(f"features: {', '.join(settings.model.features) or '(none)'}")
    typer.echo(f"margin fit: {settings.model.margin_fit}")
    typer.echo(f"simulations per game: {n:,}, seed: {seed}\n")
    for fit in fits:
        m = fit.model
        coefs = ", ".join(f"{f}={c:+.2f}" for f, c in zip(m.features, m.coefs, strict=True))
        skipped = f", {fit.skipped_games} skipped (no ratings)" if fit.skipped_games else ""
        typer.echo(
            f"{fit.season}: trained on {fit.train_seasons[0]}-{fit.train_seasons[-1]} "
            f"({m.train_games} games), predicted {fit.test_games}{skipped}; "
            f"home={m.home:+.2f}/team, dispersion={m.dispersion:.1f}, {coefs}"
        )
        if m.margin_coefs is not None:
            mc = ", ".join(f"{f}={c:+.2f}" for f, c in zip(m.features, m.margin_coefs, strict=True))
            typer.echo(f"      margin fit: home={m.margin_home:+.2f}, {mc}")
    fallback = predictions.filter(pl.col("vegas_p_source") == "spread").height
    if fallback:
        typer.echo(f"note: {fallback} game(s) had no moneyline; Vegas win prob from spread")

    digits = [("margin_mae", 2), ("total_mae", 2), ("brier", 4), ("log_loss", 4), ("pick_rate", 3)]
    typer.echo("\nMetrics (lower is better, except pick_rate = share of winners picked):")
    typer.echo(format_table(summarize(predictions), digits))
    bins = settings.backtest.calibration_bins
    for method in ("model", "vegas"):
        typer.echo(f"\nCalibration ({method}): predicted vs. actual home win rate")
        typer.echo(format_table(calibration(predictions, method, bins), [("lo", 1), ("hi", 1)]))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        predictions.write_parquet(output)
        typer.echo(f"\npredictions written to {output}")


@app.command()
def simulate(
    home: Annotated[str, typer.Option(help="Home team abbreviation, e.g. KC.")],
    away: Annotated[str, typer.Option(help="Away team abbreviation, e.g. BUF.")],
    season: Annotated[int, typer.Option(help="Season of the game.")],
    week: Annotated[int, typer.Option(help="Week of the game (ratings use weeks before it).")],
    n: Annotated[int, typer.Option(help="Number of simulations.", min=1)] = 10_000,
    seed: Annotated[int, typer.Option(help="RNG seed.")] = 42,
    neutral: Annotated[bool, typer.Option(help="Neutral site (no home-field value).")] = False,
    config: ConfigOption = None,
) -> None:
    """Simulate one game from ratings as of the given week."""
    settings = load_settings(config)
    extra = matchup_row(season, week, home, away, neutral)
    games = load_games(DataCache(settings.data), settings, season, extra=extra)
    try:
        r = predict_matchup(games, settings, n, seed)
    except ValueError as err:
        raise typer.BadParameter(str(err)) from err

    site = "neutral site" if neutral else f"at {r.home}"
    typer.echo(f"{r.away} {site}, {r.season} week {r.week} ({r.n:,} simulations, seed {r.seed})")
    typer.echo(f"  {r.home} win probability: {r.home_win_prob:.1%}")
    typer.echo(f"  {r.away} win probability: {1 - r.home_win_prob:.1%}")
    typer.echo(f"  expected points: {r.home} {r.expected_home:.1f}, {r.away} {r.expected_away:.1f}")
    typer.echo(f"  mean margin ({r.home} minus {r.away}): {r.mean_margin:+.1f}")
    typer.echo(f"  mean total: {r.mean_total:.1f}")
    qs = list(r.margin_quantiles)
    typer.echo("  quantiles:   " + "  ".join(f"{int(q * 100):>4}%" for q in qs))
    for label, values in (
        (f"{r.home} pts", r.home_quantiles),
        (f"{r.away} pts", r.away_quantiles),
        ("margin", r.margin_quantiles),
    ):
        typer.echo(f"  {label:<11}" + "  ".join(f"{values[q]:>5.0f}" for q in qs))
    for flag in r.flags:
        typer.echo(f"  flag: {flag}")


@app.command()
def tune(
    seasons: Annotated[
        str, typer.Option(help="Seasons the settings are chosen on, e.g. '2021-2023'.")
    ] = "2021-2023",
    holdout: Annotated[
        str | None,
        typer.Option(help="Held-out season(s) to confirm the winner on, e.g. '2024'."),
    ] = "2024",
    n: Annotated[int, typer.Option(help="Simulations per game.", min=1)] = 4_000,
    seed: Annotated[int, typer.Option(help="RNG seed shared by every combination.")] = 42,
    top: Annotated[int, typer.Option(help="Rows of the ranking to print.", min=1)] = 10,
    output: Annotated[
        Path | None, typer.Option(help="Write the full ranking to this parquet file.")
    ] = None,
    config: ConfigOption = None,
) -> None:
    """Grid-search the [ratings] shrinkage settings with the walk-forward backtest."""
    settings = load_settings(config)
    tune_seasons = parse_seasons(seasons)
    holdout_seasons = parse_seasons(holdout) if holdout else []
    overlap = sorted(set(tune_seasons) & set(holdout_seasons))
    if overlap:
        raise typer.BadParameter(f"holdout season(s) {overlap} are also tuning seasons")

    cache = DataCache(settings.data)
    last = max(tune_seasons + holdout_seasons)
    all_seasons = list(range(settings.data.default_first_season, last + 1))
    pbp = cache.scan("pbp", all_seasons)
    schedules = cache.load("schedules", all_seasons)

    combos = grid(settings)
    baseline = current_params(settings.ratings)
    typer.echo(f"{len(combos)} combinations on {seasons}, {n:,} simulations per game, seed {seed}")

    def progress(i: int, total: int) -> None:
        if i % 50 == 0 or i == total:
            typer.echo(f"  {i}/{total}")

    ranking = evaluate(pbp, schedules, settings, combos, tune_seasons, n, seed, progress)
    ranking = ranking.with_row_index("rank", offset=1)
    is_base = pl.all_horizontal([pl.col(k) == v for k, v in baseline.items()])
    base_row = ranking.filter(is_base)
    digits = [("brier", 4), ("log_loss", 4), ("margin_mae", 2), ("total_mae", 2)]
    digits += [(f, 2) for f in ("carryover", "min_wp", "max_wp")]
    digits += [(f, 0) for f in ("prior_plays_epa", "prior_plays_success", "prior_games_pace")]
    typer.echo(f"\nTop {top} by Brier on {seasons}:")
    typer.echo(format_table(ranking.head(top), digits))
    typer.echo("\nCurrent settings:")
    typer.echo(format_table(base_row, digits))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        ranking.write_parquet(output)
        typer.echo(f"\nranking written to {output}")

    if holdout_seasons:
        best = {k: ranking.row(0, named=True)[k] for k in baseline}
        result = evaluate_holdout(
            pbp, schedules, settings, {"current": baseline, "best": best}, holdout_seasons, n, seed
        )
        typer.echo(f"\nHoldout {holdout} (not used to choose settings):")
        typer.echo(format_table(result, [*digits, ("pick_rate", 3)]))


if __name__ == "__main__":
    app()


@app.command("player-coverage")
def player_coverage(
    seasons: Annotated[str, typer.Option(help="Seasons to check, e.g. '2016-2025'.")] = (
        "2016-2025"
    ),
    top: Annotated[int, typer.Option(help="Unmatched players to list.", min=0)] = 10,
    config: ConfigOption = None,
) -> None:
    """Report how well player IDs link across play-by-play, rosters and snap counts."""
    settings = load_settings(config)
    wanted = parse_seasons(seasons)
    cache = DataCache(settings.data)
    players = cache.load("players")
    snaps = player_snaps(cache.load("snap_counts", wanted), players)
    rosters = roster_weeks(cache.load("rosters_weekly", wanted), players)
    pbp_ids = (
        cache.scan("pbp", wanted)
        .filter(pl.col("play_type").is_in(["pass", "run"]))
        .select("season", *PBP_PLAYER_COLUMNS)
        .collect()
    )
    report = id_coverage(snaps, pbp_ids, rosters)
    shares = [c for c in report.columns if c not in ("season", "snap_rows")]
    typer.echo("Share of snaps / play-by-play player IDs linked to a known player:")
    typer.echo(format_table(report, [(c, 4) for c in shares]))
    fallback = rosters.filter(pl.col("position_source") == "players").height
    missing = rosters.filter(pl.col("position").is_null()).height
    typer.echo(
        f"\nroster player-weeks: {rosters.height:,}; position from crosswalk: {fallback:,}; "
        f"no position: {missing:,}"
    )
    if top:
        worst = unmatched_snaps(snaps, top)
        if worst.is_empty():
            typer.echo("every snap-count player matched")
        else:
            typer.echo(f"\nMost-used unmatched snap-count players (top {top}):")
            typer.echo(format_table(worst, [("offense_snaps", 0), ("defense_snaps", 0)]))
