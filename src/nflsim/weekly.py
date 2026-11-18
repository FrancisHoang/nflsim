"""A week of predictions with everything shown next to them (``nflsim week``, dashboard).

``run_week`` does the work behind ``nflsim week``: simulate every scheduled game of a week
(``predict_week``), lay them out one row per game (``week_frame``) and collect what the
predictions used (ratings cutoff, flags, data freshness). The CLI prints it; the
dashboard shows it. Both therefore show the same numbers for the same seed.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

import polars as pl

from nflsim.config import MARKET_FEATURES, PLAYER_FEATURES, Settings
from nflsim.data import DataCache, normalize_teams
from nflsim.live import cached_at, ratings_cutoff
from nflsim.pipeline import WeekPrediction, load_games_with_lineups, predict_week
from nflsim.summary import week_frame


@dataclass(frozen=True)
class WeekRun:
    """Every game of one week, plus what the predictions were built from."""

    season: int
    week: int
    n: int
    seed: int
    frame: pl.DataFrame  # week_frame: one row per game, kickoff order
    predictions: list[WeekPrediction]
    # Flag text -> the games ("AWAY@HOME") it applies to.
    flags: dict[str, list[str]] = field(default_factory=dict)
    ratings_through: int | None = None  # last week of this season in the ratings
    completed_before: int = 0  # completed games of this season before the week
    market: bool = False  # market strength (earlier weeks' spreads) is an input
    injury_aware: bool = False
    pbp_cached_at: dt.datetime | None = None
    offline: bool = False

    @property
    def simulated(self) -> int:
        return sum(p.result is not None for p in self.predictions)


def injury_settings(settings: Settings, injuries: bool | None) -> Settings:
    """``settings`` with injury-aware player features switched on or off (None: as is)."""
    if injuries is None:
        return settings
    features = settings.model.features
    if injuries:
        features = (*features, *[f for f in PLAYER_FEATURES if f not in features])
    model_cfg = settings.model.model_copy(update={"injury_aware": injuries, "features": features})
    return settings.model_copy(update={"model": model_cfg})


def group_flags(predictions: list[WeekPrediction]) -> dict[str, list[str]]:
    """Flag text -> the games it was raised for, in prediction order."""
    flags: dict[str, list[str]] = {}
    for p in predictions:
        if p.result is not None:
            label = f"{p.result.away}@{p.result.home}"
            for flag in p.result.flags:
                flags.setdefault(flag, []).append(label)
    return flags


def run_week(
    cache: DataCache,
    settings: Settings,
    season: int,
    week: int,
    n: int,
    seed: int,
    loaded: tuple[pl.DataFrame, pl.DataFrame | None] | None = None,
) -> WeekRun:
    """Simulate every scheduled game of ``season`` week ``week`` (see ``predict_week``).

    ``loaded`` is ``load_games_with_lineups(cache, settings, season)`` when the caller
    already has it (the dashboard keeps it across weeks). Raises
    ``ValueError``/``FileNotFoundError`` when the week has no games or data is missing.
    """
    schedules = cache.load("schedules", [season])
    games, lineup = loaded or load_games_with_lineups(cache, settings, season)
    preds = predict_week(games, settings, season, week, n, seed, lineup)
    schedule = normalize_teams(
        schedules.filter((pl.col("season") == season) & (pl.col("week") == week)),
        ["home_team", "away_team"],
    )
    last, done = ratings_cutoff(schedules, season, week)
    return WeekRun(
        season=season,
        week=week,
        n=n,
        seed=seed,
        frame=week_frame(preds, schedule),
        predictions=preds,
        flags=group_flags(preds),
        ratings_through=last,
        completed_before=done,
        market=bool(set(settings.model.features) & set(MARKET_FEATURES)),
        injury_aware=lineup is not None,
        pbp_cached_at=cached_at(cache, "pbp", season),
        offline=cache.offline,
    )


@dataclass(frozen=True)
class PickRecord:
    """Winners picked in a week's played games: by the model and by the Vegas line."""

    games: int  # played games without a tie
    model: int  # winners the model picked (home win probability above or below 50%)
    vegas: int  # winners the Vegas favorite won
    vegas_picks: int  # games with a Vegas favorite (pick'em lines excluded)


def pick_record(frame: pl.DataFrame) -> PickRecord:
    """Count correct picks over the played, simulated games of a ``week_frame``."""
    played = frame.filter(
        pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
        & pl.col("home_win").is_not_null()
        & (pl.col("home_score") != pl.col("away_score"))
    )
    home_won = pl.col("home_score") > pl.col("away_score")
    model = played.filter((pl.col("home_win") > 0.5) == home_won).height
    lined = played.filter(pl.col("vegas_spread").is_not_null() & (pl.col("vegas_spread") != 0))
    vegas = lined.filter((pl.col("vegas_spread") < 0) == home_won).height
    return PickRecord(games=played.height, model=model, vegas=vegas, vegas_picks=lined.height)


_SEASON_FILE = re.compile(r"season=(\d{4})\.parquet$")


def cached_seasons(cache: DataCache, name: str) -> list[int]:
    """Seasons of dataset ``name`` in the cache, oldest first (no download)."""
    folder = cache.path(name, 2000).parent
    if not folder.exists():
        return []
    found = (_SEASON_FILE.search(p.name) for p in folder.iterdir())
    return sorted(int(m.group(1)) for m in found if m)


def scheduled_weeks(cache: DataCache, season: int) -> list[int]:
    """Weeks on the cached ``season`` schedule (empty when it is not cached)."""
    if season not in cached_seasons(cache, "schedules"):
        return []
    weeks = cache.load("schedules", [season])["week"].cast(pl.Int64).unique().sort()
    return weeks.to_list()


def current_week(cache: DataCache, season: int) -> int | None:
    """First week of ``season`` with a game not yet played (last week when all are)."""
    if season not in cached_seasons(cache, "schedules"):
        return None
    sched = cache.load("schedules", [season])
    open_weeks = sched.filter(pl.col("home_score").is_null())["week"]
    if open_weeks.len():
        return int(open_weeks.cast(pl.Int64).min())  # type: ignore[arg-type]
    weeks = sched["week"]
    return int(weeks.cast(pl.Int64).max()) if weeks.len() else None  # type: ignore[arg-type]


__all__ = [
    "PickRecord",
    "WeekRun",
    "cached_seasons",
    "current_week",
    "group_flags",
    "injury_settings",
    "pick_record",
    "run_week",
    "scheduled_weeks",
]
