"""Command-line interface: ``uv run nflsim --help``."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from nflsim.backtest import calibration, format_table, summarize, walk_forward
from nflsim.config import PLAYER_FEATURES, Settings, load_settings
from nflsim.data import (
    DATASETS,
    PBP_PLAYER_COLUMNS,
    DataCache,
    depth_chart_weeks,
    dropbacks,
    id_coverage,
    injury_reports,
    parse_seasons,
    pfr_team_pressure,
    player_snaps,
    roster_weeks,
    unmatched_snaps,
)
from nflsim.data.drives import drive_coverage, drives
from nflsim.distribution import distribution_summary, key_number_table, line_calibration
from nflsim.injury_eval import (
    brier_scores,
    coverage,
    depth_coverage,
    position_table,
    rate_table,
    report_outcomes,
)
from nflsim.live import Matchup, cached_at, find_matchup, ratings_cutoff, report_summary
from nflsim.matchup_eval import coverage as matchup_coverage
from nflsim.matchup_eval import pass_epa_gate, pressure_gate, rating_skill
from nflsim.model.matchups import dropback_matchups, matchup_game_stats, matchup_ratings_as_of
from nflsim.model.players import player_game_stats, usage_history
from nflsim.model.stat_lines import Ranges, box_scores
from nflsim.overrides import Override, OverrideKind, apply_overrides, resolve, week_roster
from nflsim.pipeline import (
    MATCHUP_ID,
    MatchupResult,
    availability_report,
    expected_stat_lines,
    load_games_with_lineups,
    load_lineups,
    predict_matchup,
)
from nflsim.player_eval import evaluate_injury_usage, evaluate_ratings, evaluate_usage
from nflsim.stat_eval import baseline_per_game, evaluate_stat_lines
from nflsim.summary import (
    LineResult,
    format_spread,
    game_lines,
    line_dict,
    quantile_dict,
    records,
)
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
    injuries: Annotated[
        bool | None,
        typer.Option(
            "--injuries/--no-injuries",
            help="Injury-aware player features (default: config model.injury_aware).",
        ),
    ] = None,
    sample_availability: Annotated[
        bool | None,
        typer.Option(
            "--sample-availability/--expected-availability",
            help="With --injuries: draw who plays in each run, or use expected features "
            "(default: config sim.sample_availability).",
        ),
    ] = None,
    engine: Annotated[
        str | None,
        typer.Option(help="Simulation engine: 'score' or 'drive' (default: config sim.engine)."),
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
    if injuries is not None:
        overrides["injury_aware"] = injuries
    if overrides:
        model_cfg = settings.model.model_validate({**settings.model.model_dump(), **overrides})
        settings = settings.model_copy(update={"model": model_cfg})
    sim_updates: dict[str, object] = {}
    if sample_availability is not None:
        sim_updates["sample_availability"] = sample_availability
    if engine is not None:
        sim_updates["engine"] = engine
    if sim_updates:
        sim_cfg = settings.sim.model_validate({**settings.sim.model_dump(), **sim_updates})
        settings = settings.model_copy(update={"sim": sim_cfg})
    test_seasons = parse_seasons(seasons)
    cache = DataCache(settings.data)
    games, lineup = load_games_with_lineups(cache, settings, max(test_seasons))
    drive_table = None
    if settings.sim.engine == "drive":
        first = settings.data.default_first_season
        drive_table = drives(cache.scan("pbp", list(range(first, max(test_seasons) + 1))))
    predictions, fits = walk_forward(games, settings, test_seasons, n, seed, lineup, drive_table)

    typer.echo(f"features: {', '.join(settings.model.features) or '(none)'}")
    typer.echo(f"margin fit: {settings.model.margin_fit}")
    typer.echo(f"engine: {settings.sim.engine}")
    if set(settings.model.features) & set(PLAYER_FEATURES):
        typer.echo(f"injury-aware player features: {settings.model.injury_aware}")
        if settings.model.injury_aware:
            typer.echo(f"availability sampled per run: {settings.sim.sample_availability}")
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

    typer.echo(
        "\nWhole-distribution scores: CRPS of margin and total (lower is better; Vegas as a"
        "\nnormal around the closing line with the training seasons' spread)"
    )
    typer.echo(
        format_table(distribution_summary(predictions), [("crps_margin", 3), ("crps_total", 3)])
    )
    for kind, label in (("cover", "home covers the spread"), ("over", "game goes over the total")):
        typer.echo(f"\nModel chance the {label} vs. actual (pushes left out)")
        typer.echo(format_table(line_calibration(predictions, kind, bins), [("lo", 1), ("hi", 1)]))
    typer.echo("\nKey numbers: share of games with each absolute margin, actual vs. model")
    typer.echo(
        format_table(
            key_number_table(predictions, settings.backtest.key_margins),
            [("actual", 3), ("model", 3)],
        )
    )

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
    injuries: Annotated[
        bool | None,
        typer.Option(
            "--injuries/--no-injuries",
            help="Use the injury report in the prediction (adds injury-aware player features "
            "for this run). Default: config.",
        ),
    ] = None,
    offline: Annotated[
        bool,
        typer.Option(help="Use cached data as it is; never download (stale data is used)."),
    ] = False,
    out: Annotated[
        list[str] | None,
        typer.Option("--out", help="Player who will not play (name or GSIS ID; repeatable)."),
    ] = None,
    plays: Annotated[
        list[str] | None,
        typer.Option("--plays", help="Player who will play regardless of the report."),
    ] = None,
    doubtful: Annotated[
        list[str] | None,
        typer.Option("--doubtful", help="Player treated as doubtful (config doubtful rate)."),
    ] = None,
    compare: Annotated[
        bool,
        typer.Option(help="With overrides: also run without them (same seed) and show both."),
    ] = False,
    players: Annotated[
        bool,
        typer.Option(
            "--players", help="Per-player stat lines with 10-90% ranges (uses the injury report)."
        ),
    ] = False,
    spread: Annotated[
        float | None,
        typer.Option(
            help="Home team's spread, sportsbook style (-3.5 = home favored by 3.5). "
            "Default: the scheduled game's nflverse line, if any.",
        ),
    ] = None,
    total: Annotated[
        float | None,
        typer.Option(help="Game total line. Default: the scheduled game's nflverse line."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the full result as JSON instead of text.")
    ] = False,
    config: ConfigOption = None,
) -> None:
    """Simulate one game from ratings as of the given week.

    Uses the real scheduled game when the teams meet that week, refreshes current-season
    data that is out of date (unless --offline), and prints what data it used. --out,
    --plays and --doubtful override the injury report for either team.
    """
    settings = load_settings(config)
    by_kind: list[tuple[OverrideKind, list[str] | None]] = [
        ("out", out),
        ("plays", plays),
        ("doubtful", doubtful),
    ]
    requested = [Override(kind, name) for kind, names in by_kind for name in names or []]
    if (requested or players) and injuries is False:
        raise typer.BadParameter(
            "overrides and --players need injury-aware features; drop --no-injuries"
        )
    if requested or players:
        injuries = True  # both act through the injury-aware lineup
    if injuries is not None:
        features = settings.model.features
        if injuries:
            features = (*features, *[f for f in PLAYER_FEATURES if f not in features])
        model_cfg = settings.model.model_copy(
            update={"injury_aware": injuries, "features": features}
        )
        settings = settings.model_copy(update={"model": model_cfg})
    cache = DataCache(settings.data, offline=offline)
    effects = None
    base = None
    try:
        schedules = cache.load("schedules", [season])
        matchup = find_matchup(schedules, season, week, home, away, neutral)
        games, lineup = load_games_with_lineups(cache, settings, season, extra=matchup.row)
        if requested and lineup is not None:
            info = games.filter(pl.col("game_id") == MATCHUP_ID).row(0, named=True)
            rosters = roster_weeks(cache.load("rosters_weekly", [season]), cache.load("players"))
            roster = week_roster(rosters, season, week, (info["home_team"], info["away_team"]))
            resolved = resolve(requested, roster)
            if compare:
                base = predict_matchup(games, settings, n, seed, lineup)
            games, lineup, effects = apply_overrides(
                games, MATCHUP_ID, lineup, resolved, settings.injuries
            )
        r = predict_matchup(games, settings, n, seed, lineup)
    except (ValueError, FileNotFoundError) as err:
        raise typer.BadParameter(str(err)) from err

    neutral_site = matchup.row["location"][0] == "Neutral"
    lines = game_lines(
        r.home_scores - r.away_scores,
        r.home_scores + r.away_scores,
        spread,
        total,
        matchup.spread_line,
        matchup.total_line,
    )
    report = None
    if lineup is not None:
        report = availability_report(cache, lineup, r, settings.sim.report_min_share)
    stat_frame = None
    if players and lineup is not None:
        stat_frame = _stat_line_frame(cache, settings, lineup, r)
    used = _used(cache, schedules, matchup, r, lineup is not None)

    if json_output:
        payload = {
            "game": {
                "home": r.home,
                "away": r.away,
                "season": r.season,
                "week": r.week,
                "neutral": neutral_site,
                "scheduled_id": matchup.scheduled_id,
                "final": dict(zip(("home", "away"), matchup.final, strict=True))
                if matchup.final
                else None,
            },
            "simulation": {
                "n": r.n,
                "seed": r.seed,
                "sampled_availability": r.sampled_availability,
            },
            "result": _result_dict(r),
            "lines": [line_dict(x) for x in lines],
            "flags": r.flags,
            "may_not_play": records(report) if report is not None else None,
            "overrides": records(effects) if effects is not None else None,
            "without_overrides": _result_dict(base) if base is not None else None,
            "players": records(stat_frame) if stat_frame is not None else None,
            "used": _used_dict(used),
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    site = "neutral site" if neutral_site else f"at {r.home}"
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
    if lines:
        _print_lines(lines, r)
    if report is not None:
        min_share = settings.sim.report_min_share
        how = "drawn per run" if r.sampled_availability else "expected value in every run"
        typer.echo(f"\n  May not play (healthy share >= {min_share:.0%}; availability {how}):")
        if report.is_empty():
            typer.echo("  (none)")
        else:
            typer.echo(
                "  "
                + format_table(report, [("slot_share", 2), ("play_prob", 2)]).replace("\n", "\n  ")
            )
    for flag in r.flags:
        typer.echo(f"  flag: {flag}")
    if effects is not None:
        _scenario(effects, r, base)
    if stat_frame is not None:
        _print_stat_lines(stat_frame, settings)
    _print_used(used, r)


def _result_dict(r: MatchupResult) -> dict[str, object]:
    return {
        "home_win_prob": r.home_win_prob,
        "away_win_prob": 1 - r.home_win_prob,
        "expected_points": {"home": r.expected_home, "away": r.expected_away},
        "mean_margin": r.mean_margin,
        "mean_total": r.mean_total,
        "quantiles": {
            "home_points": quantile_dict(r.home_quantiles),
            "away_points": quantile_dict(r.away_quantiles),
            "margin": quantile_dict(r.margin_quantiles),
        },
    }


def _print_lines(lines: list[LineResult], r: MatchupResult) -> None:
    """Cover and over chances against each line (pushes counted separately)."""
    typer.echo("\n  Lines (chances from the simulated scores; the lines are not a model input):")
    for x in lines:
        where = "given" if x.source == "given" else "nflverse line"
        push = f", push {x.push:.1%}" if x.push > 0 else ""
        if x.kind == "spread":
            label = format_spread(r.home, x.line)
            typer.echo(
                f"  spread {label} ({where}): {r.home} covers {x.win:.1%}, "
                f"{r.away} covers {x.lose:.1%}{push}"
            )
        else:
            typer.echo(f"  total {x.line:g} ({where}): over {x.win:.1%}, under {x.lose:.1%}{push}")


def _stat_line_frame(
    cache: DataCache, settings: Settings, lineup: pl.DataFrame, r: MatchupResult
) -> pl.DataFrame:
    """Expected stat lines with ranges (``exp_``, ``low_``, ``high_`` per stat) for both
    teams' main players, home team first, then QB, RB, WR, TE."""
    import numpy as np

    cfg = settings.stat_lines
    seasons = [s for s in (r.season - 1, r.season) if s >= settings.data.default_first_season]
    box = box_scores(cache.load("player_stats", seasons))
    rosters = roster_weeks(cache.load("rosters_weekly", seasons), cache.load("players"))
    positions = rosters.select("season", "week", "gsis_id", "position")
    targets = pl.DataFrame(
        {"season": [r.season] * 2, "week": [r.week] * 2, "team": [r.home, r.away]},
        schema={"season": pl.Int32, "week": pl.Int32, "team": pl.String},
    )
    expected = expected_stat_lines(box, positions, lineup, targets, settings)
    ranges = Ranges(cfg.low_quantile, cfg.high_quantile, cfg.range_low, cfg.range_high)
    yards = ["passing_yards", "rushing_yards", "receiving_yards"]
    shown = expected.filter(
        pl.sum_horizontal([pl.col(f"exp_{s}") for s in yards]) >= cfg.display_min_yards
    )
    names = cache.load("players").select("gsis_id", pl.col("display_name").alias("player"))
    pos = positions.filter((pl.col("season") == r.season) & (pl.col("week") <= r.week))
    pos = pos.sort("week").unique("gsis_id", keep="last").select("gsis_id", "position")
    shown = (
        shown.join(names, on="gsis_id", how="left")
        .join(pos, on="gsis_id", how="left")
        .with_columns(pl.col("player").fill_null(pl.col("gsis_id")))
    )
    order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
    shown = shown.with_columns(
        pl.col("position").replace_strict(order, default=9, return_dtype=pl.Int32).alias("_o"),
        (pl.col("team") == r.away).alias("_away"),
    ).sort("_away", "_o", -pl.col("exp_receiving_yards"), -pl.col("exp_rushing_yards"))
    columns: dict[str, pl.Series] = {}
    for stat in _STAT_COLUMNS.values():
        e = shown[f"exp_{stat[0]}"].to_numpy().astype(np.float64)
        lo, hi = ranges.interval(stat[0], e)
        columns[f"exp_{stat[0]}"] = pl.Series(e)
        columns[f"low_{stat[0]}"] = pl.Series(np.asarray(lo, dtype=np.float64))
        columns[f"high_{stat[0]}"] = pl.Series(np.asarray(hi, dtype=np.float64))
    return pl.DataFrame(
        {
            "team": shown["team"],
            "player": shown["player"],
            "gsis_id": shown["gsis_id"],
            "position": shown["position"],
            **columns,
        }
    )


_STAT_COLUMNS = {
    "pass yds": ("passing_yards", 0),
    "pass td": ("passing_tds", 1),
    "rush yds": ("rushing_yards", 0),
    "rush td": ("rushing_tds", 1),
    "rec": ("receptions", 1),
    "rec yds": ("receiving_yards", 0),
    "rec td": ("receiving_tds", 1),
}


def _print_stat_lines(frame: pl.DataFrame, settings: Settings) -> None:
    """Print ``_stat_line_frame`` as ``expected (low-high)`` cells."""

    def cell(stat: str, digits: int) -> list[str]:
        rows = zip(frame[f"exp_{stat}"], frame[f"low_{stat}"], frame[f"high_{stat}"], strict=True)
        return [
            "" if v < 0.5 * 10.0**-digits else f"{v:.{digits}f} ({a:.{digits}f}-{b:.{digits}f})"
            for v, a, b in rows
        ]

    table = pl.DataFrame(
        {
            "team": frame["team"],
            "player": frame["player"],
            "pos": frame["position"].fill_null(""),
            **{label: cell(stat, digits) for label, (stat, digits) in _STAT_COLUMNS.items()},
        }
    )
    lo_q, hi_q = settings.stat_lines.low_quantile, settings.stat_lines.high_quantile
    typer.echo(
        f"\nPlayer stat lines: expected ({lo_q:.0%}-{hi_q:.0%} range). From team volume x share x "
        "rate, games\nbefore this week only. Rushing yards are no better than a player's "
        "per-game average\n(2023-2025 check); the other stats beat it."
    )
    typer.echo("  " + format_table(table).replace("\n", "\n  "))


def _scenario(effects: pl.DataFrame, r: MatchupResult, base: MatchupResult | None) -> None:
    """Print the overrides, what each changed, and (with --compare) the effect."""
    typer.echo("\nScenario (overrides; chance to play before -> after)")
    typer.echo("  " + format_table(effects, [("before", 2), ("after", 2)]).replace("\n", "\n  "))
    for row in effects.filter(pl.col("before").is_null()).iter_rows(named=True):
        typer.echo(
            f"  note: {row['player']} ({row['position']}) has no projected role in the model "
            "(only QB, RB, WR and TE with recent usage count), so this override does not "
            "change the prediction"
        )
    skill = effects.filter(pl.col("before").is_not_null() & (pl.col("position") != "QB"))
    if not skill.is_empty():
        typer.echo(
            "  note: the model's running back and receiver ratings barely tell players apart "
            "(Phase 2: about 0.5% skill), so RB/WR/TE overrides move predictions very little; "
            "QB overrides matter most"
        )
    if base is None:
        return
    rows = [
        (f"{r.home} win probability", base.home_win_prob, r.home_win_prob, True),
        (f"expected points {r.home}", base.expected_home, r.expected_home, False),
        (f"expected points {r.away}", base.expected_away, r.expected_away, False),
        (f"mean margin ({r.home} minus {r.away})", base.mean_margin, r.mean_margin, False),
        ("mean total", base.mean_total, r.mean_total, False),
    ]
    table = pl.DataFrame(
        {
            "": [x[0] for x in rows],
            "without": [x[1] for x in rows],
            "with": [x[2] for x in rows],
            "change": [x[2] - x[1] for x in rows],
        }
    )
    typer.echo(f"\n  Compared with no overrides (same seed {r.seed}):")
    typer.echo(
        "  "
        + format_table(table, [("without", 3), ("with", 3), ("change", 3)]).replace("\n", "\n  ")
    )


@dataclass(frozen=True)
class _Used:
    """What a simulation used: the game, how current the data is and the injury report."""

    scheduled_id: str | None
    last_week: int | None
    completed: int
    pbp_cached: dt.datetime | None
    offline: bool
    injury_aware: bool
    final: tuple[int, int] | None
    report_error: str | None = None
    report_counts: dict[str, int] | None = None
    listed: pl.DataFrame | None = None


def _used(
    cache: DataCache,
    schedules: pl.DataFrame,
    matchup: Matchup,
    r: MatchupResult,
    injury_aware: bool,
) -> _Used:
    last, done = ratings_cutoff(schedules, r.season, r.week)
    base = _Used(
        scheduled_id=matchup.scheduled_id,
        last_week=last,
        completed=done,
        pbp_cached=cached_at(cache, "pbp", r.season),
        offline=cache.offline,
        injury_aware=injury_aware,
        final=matchup.final,
    )
    try:
        reports = injury_reports(cache.load("injuries", [r.season]))
    except (ValueError, FileNotFoundError) as err:
        return replace(base, report_error=str(err))
    counts, listed = report_summary(reports, r.season, r.week, (r.home, r.away))
    return replace(base, report_counts=counts, listed=listed)


def _used_dict(u: _Used) -> dict[str, object]:
    return {
        "scheduled_id": u.scheduled_id,
        "ratings_through_week": u.last_week,
        "completed_games_before_week": u.completed,
        "pbp_cached_at": u.pbp_cached.isoformat(timespec="minutes") if u.pbp_cached else None,
        "offline": u.offline,
        "injury_aware": u.injury_aware,
        "injury_report_error": u.report_error,
        "injury_report_players": u.report_counts,
        "injury_report": records(u.listed),
    }


def _print_used(u: _Used, r: MatchupResult) -> None:
    """Print where the game came from, how current the data is and the injury report."""
    typer.echo("\nWhat I used")
    if u.scheduled_id is None:
        typer.echo("  game: not on the schedule that week; simulated as a hypothetical game")
    else:
        typer.echo(f"  game: {u.scheduled_id} from the schedule (site, roof, rest)")
    through = (
        f"through week {u.last_week}"
        if u.last_week is not None
        else ("none yet (prior season only)")
    )
    typer.echo(
        f"  ratings: {r.season} games {through}; {u.completed} completed before week {r.week}"
    )
    if u.pbp_cached is not None:
        mode = " (offline: not refreshed)" if u.offline else ""
        typer.echo(f"  play-by-play cached {u.pbp_cached:%Y-%m-%d %H:%M}{mode}")
    if u.report_error is not None or u.report_counts is None:
        typer.echo(f"  injury report: unavailable ({u.report_error})")
        return
    sizes = ", ".join(f"{t} {k} players" for t, k in u.report_counts.items())
    typer.echo(f"  injury report, week {r.week}: {sizes}")
    if not any(u.report_counts.values()):
        typer.echo(
            "    none published yet (reports come out Wednesday-Friday); "
            "everyone treated as available"
        )
    if u.injury_aware:
        typer.echo(
            "  injury-aware: on. Only QB, RB, WR and TE availability changes the prediction "
            "(see 'May not play'); other positions are listed for information."
        )
    else:
        typer.echo(
            "  injury-aware: off. The injury report does not change this prediction "
            "(use --injuries)."
        )
    if u.listed is not None and not u.listed.is_empty():
        typer.echo("  " + format_table(u.listed).replace("\n", "\n  "))
    if u.final is not None:
        h, a = u.final
        typer.echo(
            f"  final score: {r.home} {h}, {r.away} {a} (already played; the ratings use "
            "only earlier weeks)"
        )


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


@app.command("matchup-eval")
def matchup_eval(
    seasons: Annotated[str, typer.Option(help="Seasons to measure, e.g. '2016-2025'.")] = (
        "2016-2025"
    ),
    fit_seasons: Annotated[
        str, typer.Option(help="Seasons the pressure models are fit on (step 3 gate).")
    ] = "2018-2022",
    test_seasons: Annotated[
        str, typer.Option(help="Seasons the pressure models are scored on (step 3 gate).")
    ] = "2023-2025",
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples of games.", min=100)] = 2000,
    seed: Annotated[int, typer.Option(help="RNG seed for the bootstrap.")] = 42,
    config: ConfigOption = None,
) -> None:
    """Check the participation data and pass rush matchups (Phase 4, steps 1-3)."""
    settings = load_settings(config)
    fit, test = parse_seasons(fit_seasons), parse_seasons(test_seasons)
    if set(fit) & set(test):
        raise typer.BadParameter("fit and test seasons must not overlap")
    wanted = sorted(set(parse_seasons(seasons)) | set(fit) | set(test))
    cache = DataCache(settings.data)
    # One season earlier too, so ratings in the first season have last season's prior.
    first = max(min(wanted) - 1, cache.spec("participation").first_season or 0)
    loaded = list(range(first, max(wanted) + 1))
    all_plays = dropbacks(cache.scan("pbp", loaded), cache.load("participation", loaded))
    plays = all_plays.filter(pl.col("season").is_in(wanted))
    pfr_seasons = [s for s in wanted if s >= (cache.spec("pfr_pass").first_season or 0)]
    pfr = pfr_team_pressure(cache.load("pfr_pass", pfr_seasons)) if pfr_seasons else None
    typer.echo(f"Dropbacks with participation data ({seasons}). pfr_*: check against PFR")
    typer.echo("pressures allowed per team-game (from 2018).")
    table = matchup_coverage(plays, pfr)
    digits = [(c, 3) for c in table.columns if c not in ("season", "dropbacks")]
    typer.echo(format_table(table, digits))

    stats = matchup_game_stats(all_plays)
    ratings = matchup_ratings_as_of(
        stats, stats.select("season", "week", "team").unique(), settings.matchups
    )
    typer.echo(
        "\nPass rush ratings: skill at predicting each team-game's deviation from the "
        "league\n(1 - MSE / MSE of the league average; higher is better; config [matchups])"
    )
    skill = rating_skill(stats, ratings, wanted)
    typer.echo(format_table(skill, [("mse", 5), ("mse_league", 5), ("skill", 4)]))

    frame = dropback_matchups(all_plays, ratings)
    gate, models = pressure_gate(frame, fit, test, n_boot, seed)
    typer.echo(
        f"\nStep 3 gate, part 1: pressure on each dropback (fit {fit_seasons}, scored on "
        f"{test_seasons}).\nlog loss: lower is better; vs_additive: difference from the "
        "fitted additive model,\nwith a 95% interval from resampling games. The matchup "
        "term must beat additive."
    )
    typer.echo(format_table(gate, [(c, 5) for c in gate.columns if c != "model"]))
    epa = pass_epa_gate(frame, models, test, n_boot, seed)
    typer.echo(
        "\nStep 3 gate, part 2: each team-game's EPA per dropback (squared error weighted "
        "by dropbacks).\nadditive: league + offense + defense EPA; mixture/<model>: "
        "pressure chance x EPA pressured\n+ the rest x EPA clean."
    )
    typer.echo(format_table(epa, [(c, 5) for c in epa.columns[2:]]))


@app.command("drive-eval")
def drive_eval(
    seasons: Annotated[str, typer.Option(help="Seasons to measure, e.g. '2016-2025'.")] = (
        "2016-2025"
    ),
    config: ConfigOption = None,
) -> None:
    """Check the drive table built from play-by-play (Phase 5): results, field position,
    clock, and whether drive points add up to every final score."""
    settings = load_settings(config)
    wanted = parse_seasons(seasons)
    cache = DataCache(settings.data)
    table = drive_coverage(drives(cache.scan("pbp", wanted)), cache.load("schedules", wanted))
    typer.echo(f"Drives by season ({seasons}). score_match: share of games where drive")
    typer.echo("points equal both final scores. Result columns are shares of drives.")
    typer.echo(format_table(table, [(c, 3) for c in table.columns if c != "season"]))


@app.command("stat-eval")
def stat_eval(
    fit_seasons: Annotated[
        str, typer.Option(help="Seasons the ranges are fit on, e.g. '2018-2022'.")
    ] = "2018-2022",
    test_seasons: Annotated[
        str, typer.Option(help="Seasons to score, e.g. '2023-2025'.")
    ] = "2023-2025",
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples of games.", min=100)] = 2000,
    seed: Annotated[int, typer.Option(help="RNG seed for the bootstrap.")] = 42,
    config: ConfigOption = None,
) -> None:
    """Per-player stat lines vs. each player's per-game average, out of sample (Phase 7)."""
    settings = load_settings(config)
    fit, test = parse_seasons(fit_seasons), parse_seasons(test_seasons)
    if set(fit) & set(test):
        raise typer.BadParameter("fit and test seasons must not overlap")
    cache = DataCache(settings.data)
    evals = sorted(set(fit) | set(test))
    seasons = list(range(settings.data.default_first_season, max(evals) + 1))
    rosters = roster_weeks(cache.load("rosters_weekly", seasons), cache.load("players"))
    player_games, role_totals = player_game_stats(
        cache.scan("pbp", seasons), rosters, settings.ratings
    )
    box = box_scores(cache.load("player_stats", seasons))
    targets = box.filter(pl.col("season").is_in(evals)).select("season", "week", "team").unique()
    _, lineup = load_lineups(cache, settings, evals, targets, player_games, role_totals, rosters)
    positions = rosters.select("season", "week", "gsis_id", "position")
    expected = expected_stat_lines(box, positions, lineup, targets, settings)
    pairs = expected.join(box, on=["season", "week", "team", "gsis_id"], how="inner")
    pairs = pairs.join(baseline_per_game(box, pairs), on=["season", "week", "gsis_id"], how="left")
    cfg = settings.stat_lines
    table, ranges = evaluate_stat_lines(
        pairs, fit, test, cfg.low_quantile, cfg.high_quantile, cfg.min_expected, n_boot, seed
    )
    typer.echo(
        f"Stat lines on {test_seasons} (player-games he played, with a baseline): model vs. his "
        "per-game average\nso far (last season's before he plays). diff = MAE model - MAE "
        "baseline (negative is better),\n95% CI from resampling games. in_range: share inside "
        f"the {cfg.low_quantile:.0%}-{cfg.high_quantile:.0%} range fit on {fit_seasons}."
    )
    typer.echo(format_table(table, [(c, 3) for c in table.columns[2:]]))
    typer.echo("\nScaled-error quantiles (config [stat_lines] range_low / range_high):")
    for s in ranges.low:
        typer.echo(f"  {s}: {ranges.low[s]:+.3f} / {ranges.high[s]:+.3f}")


@app.command("injury-eval")
def injury_eval(
    seasons: Annotated[str, typer.Option(help="Seasons to measure, e.g. '2016-2025'.")] = (
        "2016-2025"
    ),
    config: ConfigOption = None,
) -> None:
    """Measure how often players on the injury report play, vs. the configured rates."""
    settings = load_settings(config)
    wanted = parse_seasons(seasons)
    cache = DataCache(settings.data)
    cfg = settings.injuries
    # One season earlier, so week 1 knows who the regulars were.
    first = max(min(wanted) - 1, cache.spec("snap_counts").first_season or 0)
    players = cache.load("players")
    snaps = player_snaps(cache.load("snap_counts", range(first, max(wanted) + 1)), players)
    rosters = roster_weeks(cache.load("rosters_weekly", wanted), players)
    reports = injury_reports(cache.load("injuries", wanted))
    in_range = pl.col("season").is_in(wanted)
    ruled_out = settings.player_ratings.unavailable_statuses
    outcomes = report_outcomes(reports, snaps, cfg.regular_min_share, rosters, ruled_out).filter(
        in_range
    )
    snaps = snaps.filter(in_range)

    typer.echo(f"Injury report coverage ({seasons}):")
    typer.echo(format_table(coverage(reports, snaps), []))
    depth = depth_chart_weeks(cache.load("depth_charts", wanted), cache.load("schedules", wanted))
    typer.echo("\nDepth charts (starters = depth 1 on offense/defense):")
    typer.echo(
        format_table(
            depth_coverage(depth, snaps),
            [("charted", 3), ("starters_played", 3), ("qb_starter_led", 3)],
        )
    )

    typer.echo(
        f"\nPlay rate by final report (regular: >= {cfg.regular_min_share:.0%} of offensive "
        "or defensive snaps in the previous game; players with weekly roster status "
        f"{', '.join(ruled_out)} left out)"
    )
    rate_digits = [(c, 3) for c in ("observed", "config", "diff")]
    typer.echo(format_table(rate_table(outcomes, cfg), rate_digits))
    typer.echo("\nQuestionable regulars by position:")
    typer.echo(format_table(position_table(outcomes, cfg), rate_digits))
    typer.echo("\nBrier score of play predictions (lower is better):")
    briers = brier_scores(outcomes, cfg)
    typer.echo(format_table(briers, [("played", 3), *[(c, 4) for c in briers.columns[3:]]]))


@app.command("player-eval")
def player_eval(
    seasons: Annotated[str, typer.Option(help="Test seasons, e.g. '2018-2024'.")] = "2018-2024",
    injuries: Annotated[
        bool,
        typer.Option(help="Also compare injury-aware usage (needs injuries, depth charts)."),
    ] = False,
    affected_share: Annotated[
        float,
        typer.Option(
            help="With --injuries: a team-week counts as affected when a player with at "
            "least this healthy share may not play.",
            min=0.0,
            max=1.0,
        ),
    ] = 0.10,
    config: ConfigOption = None,
) -> None:
    """Check player ratings and projected usage against what happened, out of sample."""
    settings = load_settings(config)
    test_seasons = parse_seasons(seasons)
    cache = DataCache(settings.data)
    all_seasons = list(range(settings.data.default_first_season, max(test_seasons) + 1))
    rosters = roster_weeks(cache.load("rosters_weekly", all_seasons), cache.load("players"))
    player_games, team_games = player_game_stats(
        cache.scan("pbp", all_seasons), rosters, settings.ratings
    )
    cfg = settings.player_ratings

    ratings = evaluate_ratings(player_games, test_seasons, cfg)
    typer.echo(f"Player ratings on {seasons} (predicting each game's EPA/play, weighted by plays)")
    typer.echo("skill = 1 - MSE / MSE of the position average; higher is better")
    mse_cols = [c for c in ratings.columns if c.startswith(("mse_", "skill_"))]
    typer.echo(
        format_table(ratings, [("plays", 0), ("unknown_share", 3), *[(c, 4) for c in mse_cols]])
    )

    history = usage_history(player_games, team_games, rosters, cfg)
    usage = evaluate_usage(history, team_games, rosters, test_seasons, cfg)
    typer.echo("\nProjected usage vs. actual shares (total-variation distance; 0 = exact)")
    typer.echo(format_table(usage, [("no_projection", 4), ("tv_distance", 4)]))

    if injuries:
        targets = (
            team_games.filter(pl.col("season").is_in(test_seasons))
            .select("season", "week", "team")
            .unique()
        )
        _, lineup = load_lineups(
            cache, settings, test_seasons, targets, player_games, team_games, rosters
        )
        result = evaluate_injury_usage(history, lineup, test_seasons, affected_share)
        typer.echo(
            "\nUsage with injury reports vs. Phase 2 (tv: total-variation distance, lower is "
            "better; top: projected top player had the most plays, higher is better)"
        )
        typer.echo(format_table(result, [(c, 4) for c in result.columns[3:]]))
