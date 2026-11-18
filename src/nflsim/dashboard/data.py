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
from nflsim.data import DataCache
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


__all__ = ["CONFIG_ENV", "cache", "games", "seasons", "settings", "week_run", "weeks"]
