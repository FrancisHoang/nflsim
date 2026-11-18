"""Live matchups: the real scheduled game and what data a simulation used (Phase 7, step 1)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from nflsim.data import DataCache, normalize_teams
from nflsim.data.teams import TEAM_ALIASES
from nflsim.pipeline import MATCHUP_ID, matchup_row

# Schedule columns carried into the simulated row when the game is on the schedule.
_CONTEXT = ["game_type", "location", "roof", "home_rest", "away_rest", "temp", "wind"]
# Report statuses shown in the "what I used" block (players merely on the practice
# report with no game status are counted but not listed).
LISTED_STATUSES = ("out", "doubtful", "questionable")


@dataclass(frozen=True)
class Matchup:
    """The row to simulate and where it came from."""

    row: pl.DataFrame  # schedule-shaped, game_id = MATCHUP_ID
    scheduled_id: str | None  # nflverse game_id when the game is on the schedule
    final: tuple[int, int] | None  # (home, away) score when it has been played
    # nflverse lines for the scheduled game (spread_line > 0 when home is favored).
    spread_line: float | None = None
    total_line: float | None = None


def _code(team: str) -> str:
    team = team.upper()
    return TEAM_ALIASES.get(team, team)


def find_matchup(
    schedules: pl.DataFrame, season: int, week: int, home: str, away: str, neutral: bool
) -> Matchup:
    """The scheduled game between ``home`` and ``away`` in that week, else a hypothetical one.

    A scheduled game keeps its site, game type, roof, rest and weather. Raises
    ``ValueError`` when the teams meet that week with home and away the other way round,
    or when ``neutral`` contradicts the schedule.
    """
    home, away = _code(home), _code(away)
    sched = normalize_teams(
        schedules.filter((pl.col("season") == season) & (pl.col("week") == week)),
        ["home_team", "away_team"],
    )
    game = sched.filter((pl.col("home_team") == home) & (pl.col("away_team") == away))
    if game.is_empty():
        swapped = sched.filter((pl.col("home_team") == away) & (pl.col("away_team") == home))
        if not swapped.is_empty():
            raise ValueError(
                f"{away} is the home team in {season} week {week}; use --home {away} --away {home}"
            )
        return Matchup(matchup_row(season, week, home, away, neutral), None, None)

    info = game.row(0, named=True)
    if neutral and info.get("location") != "Neutral":
        raise ValueError(f"{info['game_id']} is not at a neutral site; drop --neutral")
    row = matchup_row(season, week, home, away, info.get("location") == "Neutral").drop(
        "game_type", "location"
    )
    context = game.select([c for c in _CONTEXT if c in game.columns])
    final = None
    if info.get("home_score") is not None and info.get("away_score") is not None:
        final = (int(info["home_score"]), int(info["away_score"]))
    spread, total = info.get("spread_line"), info.get("total_line")
    return Matchup(
        row.hstack(context),
        str(info["game_id"]),
        final,
        float(spread) if spread is not None else None,
        float(total) if total is not None else None,
    )


def ratings_cutoff(schedules: pl.DataFrame, season: int, week: int) -> tuple[int | None, int]:
    """Last week of ``season`` before ``week`` with completed games, and how many games
    of the season before ``week`` are complete (what the ratings can use)."""
    done = schedules.filter(
        (pl.col("season") == season) & (pl.col("week") < week) & pl.col("home_score").is_not_null()
    )
    last = done.select(pl.col("week").max()).item()
    return (int(last) if last is not None else None), done.height


def report_summary(
    reports: pl.DataFrame, season: int, week: int, teams: tuple[str, str]
) -> tuple[dict[str, int], pl.DataFrame]:
    """Players on each team's injury report for that week.

    Returns ``(counts, listed)``: ``counts`` maps team -> players on the report (0 when
    there is none), and ``listed`` has those with a game status of out, doubtful or
    questionable (all positions, defense included): ``team, name, position, status,
    practice, injury``.
    """
    week_reports = reports.filter(
        (pl.col("season") == season) & (pl.col("week") == week) & pl.col("team").is_in(teams)
    )
    counts = {t: week_reports.filter(pl.col("team") == t).height for t in teams}
    order = {s: i for i, s in enumerate(LISTED_STATUSES)}
    listed = (
        week_reports.filter(pl.col("status").is_in(LISTED_STATUSES))
        .with_columns(pl.col("status").replace_strict(order, return_dtype=pl.Int32).alias("_o"))
        .sort("team", "_o", "position", "name")
        .select("team", "name", "position", "status", "practice", "injury")
    )
    return counts, listed


def cached_at(cache: DataCache, name: str, season: int) -> dt.datetime | None:
    """When a season's cached file was last written (None when not cached)."""
    path = cache.path(name, season)
    if not path.exists():
        return None
    return dt.datetime.fromtimestamp(path.stat().st_mtime)


__all__ = [
    "LISTED_STATUSES",
    "MATCHUP_ID",
    "Matchup",
    "cached_at",
    "find_matchup",
    "ratings_cutoff",
    "report_summary",
]
