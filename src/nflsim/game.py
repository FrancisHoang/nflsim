"""One game with everything shown next to it (``nflsim simulate``, dashboard game page).

``run_game`` does the work behind ``nflsim simulate``: find the scheduled game (or build a
hypothetical one), apply availability overrides, simulate it, and collect the lines,
the players who may not play, per-player stat lines and what the prediction used. The
CLI prints it; the dashboard shows it, so both show the same numbers for the same seed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from nflsim.config import MARKET_FEATURES, Settings
from nflsim.data import DataCache, injury_reports, roster_weeks
from nflsim.live import Matchup, cached_at, find_matchup, ratings_cutoff, report_summary
from nflsim.model.stat_lines import Ranges, box_scores
from nflsim.overrides import Override, apply_overrides, resolve, week_roster
from nflsim.pipeline import (
    MATCHUP_ID,
    MatchupResult,
    availability_report,
    expected_stat_lines,
    load_games_with_lineups,
    predict_matchup,
)
from nflsim.summary import LineResult, game_lines


@dataclass(frozen=True)
class GameRequest:
    """Which game to simulate and how."""

    season: int
    week: int
    home: str
    away: str
    neutral: bool = False
    n: int = 10_000
    seed: int = 42


@dataclass(frozen=True)
class GameUsed:
    """What a simulation used: the game, how current the data is and the injury report."""

    scheduled_id: str | None
    last_week: int | None
    completed: int
    pbp_cached: dt.datetime | None
    offline: bool
    injury_aware: bool
    final: tuple[int, int] | None
    market: bool = False  # market strength (earlier weeks' closing spreads) used
    # Quarterback change: team -> listed starter's name (None: not listed, projection used).
    starters: dict[str, str | None] | None = None
    report_error: str | None = None
    report_counts: dict[str, int] | None = None
    listed: pl.DataFrame | None = None


def game_used(
    cache: DataCache,
    schedules: pl.DataFrame,
    matchup: Matchup,
    r: MatchupResult,
    injury_aware: bool,
    market: bool = False,
) -> GameUsed:
    last, done = ratings_cutoff(schedules, r.season, r.week)
    base = GameUsed(
        scheduled_id=matchup.scheduled_id,
        last_week=last,
        completed=done,
        pbp_cached=cached_at(cache, "pbp", r.season),
        offline=cache.offline,
        injury_aware=injury_aware,
        final=matchup.final,
        market=market,
    )
    try:
        reports = injury_reports(cache.load("injuries", [r.season]))
    except (ValueError, FileNotFoundError) as err:
        return replace(base, report_error=str(err))
    counts, listed = report_summary(reports, r.season, r.week, (r.home, r.away))
    return replace(base, report_counts=counts, listed=listed)


def listed_starters(
    cache: DataCache, games: pl.DataFrame, r: MatchupResult, ruled_out: set[str]
) -> dict[str, str | None]:
    """Each team's listed starting QB in the matchup row (None when not listed); one in
    ``ruled_out`` (overridden out or doubtful) is marked as replaced by the projection."""
    info = games.filter(pl.col("game_id") == MATCHUP_ID).row(0, named=True)
    names = dict(cache.load("players").select("gsis_id", "display_name").iter_rows())
    out: dict[str, str | None] = {}
    for team, side in ((r.home, "h_"), (r.away, "a_")):
        gsis = info.get(f"{side}qb_starter_id")
        if not gsis:
            out[team] = None
            continue
        name = names.get(gsis) or gsis
        out[team] = f"{name} (overridden; his backup is used)" if gsis in ruled_out else name
    return out


STAT_COLUMNS = {
    "pass yds": ("passing_yards", 0),
    "pass td": ("passing_tds", 1),
    "rush yds": ("rushing_yards", 0),
    "rush td": ("rushing_tds", 1),
    "rec": ("receptions", 1),
    "rec yds": ("receiving_yards", 0),
    "rec td": ("receiving_tds", 1),
}


def stat_line_frame(
    cache: DataCache, settings: Settings, lineup: pl.DataFrame, r: MatchupResult
) -> pl.DataFrame:
    """Expected stat lines with ranges (``exp_``, ``low_``, ``high_`` per stat) for both
    teams' main players, home team first, then QB, RB, WR, TE."""
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
    for stat in STAT_COLUMNS.values():
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


def stat_line_cells(frame: pl.DataFrame) -> pl.DataFrame:
    """``stat_line_frame`` as display text: ``expected (low-high)`` per stat, blank when
    the expected value rounds to 0."""

    def cell(stat: str, digits: int) -> list[str]:
        rows = zip(frame[f"exp_{stat}"], frame[f"low_{stat}"], frame[f"high_{stat}"], strict=True)
        return [
            "" if v < 0.5 * 10.0**-digits else f"{v:.{digits}f} ({a:.{digits}f}-{b:.{digits}f})"
            for v, a, b in rows
        ]

    return pl.DataFrame(
        {
            "team": frame["team"],
            "player": frame["player"],
            "pos": frame["position"].fill_null(""),
            **{label: cell(stat, digits) for label, (stat, digits) in STAT_COLUMNS.items()},
        }
    )


@dataclass(frozen=True)
class GameRun:
    """A simulated game and everything shown next to it."""

    matchup: Matchup
    result: MatchupResult
    lines: list[LineResult]
    used: GameUsed
    neutral_site: bool
    # Players who may not play (injury-aware only; ``availability_report``).
    report: pl.DataFrame | None = None
    # Overrides and their effect on each player's chance to play.
    effects: pl.DataFrame | None = None
    base: MatchupResult | None = None  # same seed, no overrides (``compare``)
    stat_frame: pl.DataFrame | None = None  # per-player stat lines (``players``)


def run_game(
    cache: DataCache,
    settings: Settings,
    game: GameRequest,
    *,
    overrides: list[Override] | None = None,
    compare: bool = False,
    players: bool = False,
    spread: float | None = None,
    total: float | None = None,
    loaded: tuple[pl.DataFrame, pl.DataFrame | None] | None = None,
) -> GameRun:
    """Simulate one game as ``nflsim simulate`` does.

    ``settings`` must already be injury-aware when ``overrides`` or ``players`` are used
    (``nflsim.weekly.injury_settings``); otherwise they have no effect, as in the CLI.
    ``spread``/``total`` default to the scheduled game's nflverse lines. ``loaded`` is
    ``load_games_with_lineups`` with this game's row when the caller already has it.
    Raises ``ValueError``/``FileNotFoundError`` for unknown teams or missing data.
    """
    requested = overrides or []
    season, week, n, seed = game.season, game.week, game.n, game.seed
    effects = None
    base = None
    ruled_out: set[str] = set()
    schedules = cache.load("schedules", [season])
    matchup = find_matchup(schedules, season, week, game.home, game.away, game.neutral)
    games, lineup = loaded or load_games_with_lineups(cache, settings, season, extra=matchup.row)
    if requested and lineup is not None:
        info = games.filter(pl.col("game_id") == MATCHUP_ID).row(0, named=True)
        rosters = roster_weeks(cache.load("rosters_weekly", [season]), cache.load("players"))
        roster = week_roster(rosters, season, week, (info["home_team"], info["away_team"]))
        resolved = resolve(requested, roster)
        ruled_out = {x.gsis_id for x in resolved if x.kind in ("out", "doubtful")}
        if compare:
            base = predict_matchup(games, settings, n, seed, lineup)
        games, lineup, effects = apply_overrides(
            games, MATCHUP_ID, lineup, resolved, settings.injuries
        )
    r = predict_matchup(games, settings, n, seed, lineup)
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
        stat_frame = stat_line_frame(cache, settings, lineup, r)
    market = bool(set(settings.model.features) & set(MARKET_FEATURES))
    used = game_used(cache, schedules, matchup, r, lineup is not None, market)
    if "qb_change" in settings.model.features:
        used = replace(used, starters=listed_starters(cache, games, r, ruled_out))
    return GameRun(
        matchup=matchup,
        result=r,
        lines=lines,
        used=used,
        neutral_site=matchup.row["location"][0] == "Neutral",
        report=report,
        effects=effects,
        base=base,
        stat_frame=stat_frame,
    )


__all__ = [
    "STAT_COLUMNS",
    "GameRequest",
    "GameRun",
    "GameUsed",
    "game_used",
    "listed_starters",
    "run_game",
    "stat_line_cells",
    "stat_line_frame",
]
