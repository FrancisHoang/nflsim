"""Cached data access for the dashboard pages.

Settings come from the file in ``NFLSIM_CONFIG`` (set by ``nflsim dashboard --config``)
or the packaged defaults. The cache is always offline: the dashboard never downloads.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import streamlit as st

from nflsim.config import Settings, load_settings
from nflsim.data import DataCache, normalize_teams, roster_weeks
from nflsim.game import GameRequest, GameRun, run_game
from nflsim.live import find_matchup
from nflsim.overrides import Override, week_roster
from nflsim.pipeline import load_games_with_lineups
from nflsim.weekly import WeekRun, cached_seasons, current_week, injury_settings, run_week

CONFIG_ENV = "NFLSIM_CONFIG"


def settings() -> Settings:
    path = os.environ.get(CONFIG_ENV)
    return load_settings(Path(path) if path else None)


def cache() -> DataCache:
    return DataCache(settings().data, offline=True)


@st.cache_data(ttl=600, show_spinner=False)
def seasons() -> list[int]:
    """Seasons with a cached schedule and play-by-play, newest first."""
    c = cache()
    both = set(cached_seasons(c, "schedules")) & set(cached_seasons(c, "pbp"))
    return sorted(both, reverse=True)


@st.cache_data(ttl=600, show_spinner=False)
def weeks(season: int) -> tuple[list[int], int | None]:
    """The season's scheduled weeks and the first one not yet fully played."""
    c = cache()
    sched = c.load("schedules", [season])
    return sorted(int(w) for w in sched["week"].unique().to_list()), current_week(c, season)


@st.cache_resource(ttl=3600, show_spinner=False, max_entries=4)
def games(season: int, injuries: bool) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Rated games (and lineups) through ``season``: the slow part, kept across weeks."""
    return load_games_with_lineups(cache(), injury_settings(settings(), injuries), season)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=32)
def week_run(season: int, week: int, n: int, seed: int, injuries: bool) -> WeekRun:
    """``run_week`` for the dashboard's settings (cached per argument set)."""
    s = injury_settings(settings(), injuries)
    return run_week(cache(), s, season, week, n, seed, loaded=games(season, injuries))


@st.cache_data(ttl=600, show_spinner=False)
def week_games(season: int, week: int) -> list[tuple[str, str]]:
    """``(away, home)`` of every scheduled game that week, in kickoff order."""
    sched = cache().load("schedules", [season])
    games = normalize_teams(sched.filter(pl.col("week") == week), ["home_team", "away_team"]).sort(
        "gameday", "gametime", "game_id", nulls_last=True
    )
    return list(zip(games["away_team"].to_list(), games["home_team"].to_list(), strict=True))


@st.cache_data(ttl=600, show_spinner=False)
def teams(season: int) -> list[str]:
    """Every team on the ``season`` schedule (current codes), sorted."""
    sched = normalize_teams(cache().load("schedules", [season]), ["home_team", "away_team"])
    return sorted(set(sched["home_team"].to_list()) | set(sched["away_team"].to_list()))


@st.cache_data(ttl=600, show_spinner=False)
def roster(season: int, week: int, home: str, away: str) -> pl.DataFrame | None:
    """Both teams' rosters as of the week (``week_roster``), None when not cached."""
    c = cache()
    try:
        rosters = roster_weeks(c.load("rosters_weekly", [season]), c.load("players"))
        return week_roster(rosters, season, week, (home, away))
    except (ValueError, FileNotFoundError):
        return None


@st.cache_resource(ttl=3600, show_spinner=False, max_entries=8)
def game_games(
    season: int, week: int, home: str, away: str, neutral: bool, injuries: bool
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Rated games with this matchup's row (``run_game`` input), kept across reruns."""
    c = cache()
    matchup = find_matchup(c.load("schedules", [season]), season, week, home, away, neutral)
    s = injury_settings(settings(), injuries)
    return load_games_with_lineups(c, s, season, extra=matchup.row)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=32)
def game_run(
    game: GameRequest,
    injuries: bool,
    out: tuple[str, ...] = (),
    doubtful: tuple[str, ...] = (),
    players: bool = False,
    spread: float | None = None,
    total: float | None = None,
) -> GameRun:
    """``run_game`` for the dashboard's settings. ``out``/``doubtful`` are GSIS IDs.

    Overrides and stat lines act through the injury report, so they switch it on.
    """
    injuries = injuries or bool(out or doubtful or players)
    overrides = [Override("out", x) for x in out] + [Override("doubtful", x) for x in doubtful]
    return run_game(
        cache(),
        injury_settings(settings(), injuries),
        game,
        overrides=overrides,
        compare=bool(overrides),
        players=players,
        spread=spread,
        total=total,
        loaded=game_games(game.season, game.week, game.home, game.away, game.neutral, injuries),
    )


__all__ = [
    "CONFIG_ENV",
    "cache",
    "game_games",
    "game_run",
    "games",
    "roster",
    "seasons",
    "settings",
    "teams",
    "week_games",
    "week_run",
    "weeks",
]
