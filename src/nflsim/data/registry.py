"""Catalogue of nflverse datasets nflsim knows how to fetch.

Each entry wraps one ``nflreadpy`` loader. Seasonal datasets are fetched and cached
one season at a time; the rest (``players``, ``teams``) are a single table.
``first_season`` is the earliest season nflverse publishes for that dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import nflreadpy
import polars as pl

SeasonFetcher = Callable[[int], pl.DataFrame]
StaticFetcher = Callable[[], pl.DataFrame]


@dataclass(frozen=True)
class DatasetSpec:
    """How to fetch one dataset.

    Exactly one of ``fetch_season`` (seasonal data) or ``fetch_all`` (static data)
    is set.
    """

    name: str
    description: str
    first_season: int | None = None
    fetch_season: SeasonFetcher | None = None
    fetch_all: StaticFetcher | None = None

    @property
    def seasonal(self) -> bool:
        """True when the dataset is split by season."""
        return self.fetch_season is not None


def _season_loader(func: str, **kwargs: Any) -> SeasonFetcher:
    """Build a fetcher that calls ``nflreadpy.<func>(season, **kwargs)``.

    The function is looked up at call time, not import time, so tests can replace it.
    """

    def fetch(season: int) -> pl.DataFrame:
        result: pl.DataFrame = getattr(nflreadpy, func)(season, **kwargs)
        return result

    return fetch


def _static_loader(func: str) -> StaticFetcher:
    """Build a fetcher that calls ``nflreadpy.<func>()`` (looked up at call time)."""

    def fetch() -> pl.DataFrame:
        result: pl.DataFrame = getattr(nflreadpy, func)()
        return result

    return fetch


def _seasonal(
    name: str, description: str, first: int, func: str, **kwargs: Any
) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        description=description,
        first_season=first,
        fetch_season=_season_loader(func, **kwargs),
    )


def _static(name: str, description: str, func: str) -> DatasetSpec:
    return DatasetSpec(name=name, description=description, fetch_all=_static_loader(func))


_WEEK = {"summary_level": "week"}

DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in [
        _seasonal("pbp", "Play-by-play with EPA/WP", 1999, "load_pbp"),
        _seasonal(
            "schedules", "Games, results, and closing spread/total lines", 1999, "load_schedules"
        ),
        _seasonal("rosters", "Season rosters", 1999, "load_rosters"),
        _seasonal("rosters_weekly", "Week-by-week rosters", 2002, "load_rosters_weekly"),
        _seasonal(
            "player_stats", "Weekly player box-score stats", 1999, "load_player_stats", **_WEEK
        ),
        _seasonal("team_stats", "Weekly team stats", 1999, "load_team_stats", **_WEEK),
        _seasonal(
            "injuries", "Weekly injury reports and practice participation", 2009, "load_injuries"
        ),
        _seasonal("depth_charts", "Weekly depth charts", 2001, "load_depth_charts"),
        _seasonal("snap_counts", "Offensive/defensive/ST snaps", 2012, "load_snap_counts"),
        _seasonal("participation", "Players on the field per play", 2016, "load_participation"),
        *[
            _seasonal(
                f"ngs_{kind}",
                f"Next Gen Stats ({kind})",
                2016,
                "load_nextgen_stats",
                stat_type=kind,
            )
            for kind in ("passing", "receiving", "rushing")
        ],
        *[
            _seasonal(
                f"pfr_{kind}",
                f"PFR advanced weekly stats ({kind})",
                2018,
                "load_pfr_advstats",
                stat_type=kind,
                **_WEEK,
            )
            for kind in ("pass", "rush", "rec", "def")
        ],
        _static("players", "Player ID crosswalk and bio", "load_players"),
        _static("teams", "Team abbreviations, colours, divisions", "load_teams"),
    ]
}
