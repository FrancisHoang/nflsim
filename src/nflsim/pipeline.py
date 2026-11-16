"""Glue between the data cache and the models: build the game table and predict matchups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nflsim.config import Settings
from nflsim.data import DataCache
from nflsim.data.teams import TEAM_ALIASES
from nflsim.model import (
    PointsModel,
    build_games,
    fit_points_model,
    simulate_scores,
    team_game_stats,
)

MATCHUP_ID = "SIM"


def load_games(
    cache: DataCache, settings: Settings, last_season: int, extra: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Build the game table (see ``build_games``) for every cached season through ``last_season``.

    ``extra`` holds schedule-shaped rows to add, e.g. a hypothetical matchup.
    """
    seasons = list(range(settings.data.default_first_season, last_season + 1))
    pbp = cache.scan("pbp", seasons)
    schedules = cache.load("schedules", seasons)
    if extra is not None:
        schedules = pl.concat([schedules, extra], how="diagonal_relaxed")
    team_games = team_game_stats(pbp, settings.ratings)
    return build_games(schedules, team_games, settings.ratings)


def matchup_row(season: int, week: int, home: str, away: str, neutral: bool) -> pl.DataFrame:
    """A schedule row for a hypothetical (or upcoming) game with no score yet."""
    home, away = home.upper(), away.upper()
    if home == away:
        raise ValueError("home and away must differ")
    return pl.DataFrame(
        {
            "game_id": [MATCHUP_ID],
            "season": [season],
            "week": [week],
            "game_type": [MATCHUP_ID],
            "home_team": [TEAM_ALIASES.get(home, home)],
            "away_team": [TEAM_ALIASES.get(away, away)],
            "home_score": [None],
            "away_score": [None],
            "location": ["Neutral" if neutral else "Home"],
            "spread_line": [None],
            "total_line": [None],
            "home_moneyline": [None],
            "away_moneyline": [None],
        },
        schema_overrides={
            "season": pl.Int32,
            "week": pl.Int32,
            "home_score": pl.Int64,
            "away_score": pl.Int64,
            "spread_line": pl.Float64,
            "total_line": pl.Float64,
            "home_moneyline": pl.Float64,
            "away_moneyline": pl.Float64,
        },
    )


@dataclass(frozen=True)
class MatchupResult:
    home: str
    away: str
    season: int
    week: int
    n: int
    seed: int
    home_win_prob: float
    mean_margin: float
    mean_total: float
    expected_home: float
    expected_away: float
    margin_quantiles: dict[float, float]
    home_quantiles: dict[float, float]
    away_quantiles: dict[float, float]
    flags: list[str]
    model: PointsModel


QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def predict_matchup(games: pl.DataFrame, settings: Settings, n: int, seed: int) -> MatchupResult:
    """Simulate the ``MATCHUP_ID`` row of ``games``, fitting on seasons before its season."""
    row = games.filter(pl.col("game_id") == MATCHUP_ID)
    if row.height != 1:
        raise ValueError("games must contain exactly one matchup row")
    info = row.row(0, named=True)
    season = int(info["season"])
    k = settings.model.train_seasons
    train = games.filter(
        pl.col("season").is_between(season - k, season - 1) & pl.col("home_score").is_not_null()
    )
    if train.is_empty():
        raise ValueError(f"no completed seasons before {season} to fit the model on")

    others = games.filter(pl.col("game_id") != MATCHUP_ID)
    known = set(others["home_team"].to_list()) | set(others["away_team"].to_list())
    unknown = [t for t in (info["home_team"], info["away_team"]) if t not in known]
    if unknown:
        raise ValueError(f"unknown team abbreviation(s): {', '.join(unknown)}")

    flags: list[str] = []
    for prefix, name in (("h_", info["home_team"]), ("a_", info["away_team"])):
        if info[f"{prefix}off_epa"] is None:
            raise ValueError(
                f"no ratings for {name!r} before {season} week {info['week']}: "
                "unknown team abbreviation, or no data cached for that time"
            )
        if info[f"{prefix}prior_source"] == "league":
            flags.append(f"{name}: previous season not cached; prior is league average")
        if info[f"{prefix}games_played"] == 0:
            flags.append(f"{name}: no games yet this season; rating is the prior only")

    model = fit_points_model(train, settings.model)
    mu_home, mu_away = model.expected_points(row)
    sim = simulate_scores(mu_home, mu_away, model.dispersion, n, seed, settings.sim.overtime_points)

    def quantiles(values: np.ndarray) -> dict[float, float]:
        return {q: float(np.quantile(values, q)) for q in QUANTILES}

    return MatchupResult(
        home=info["home_team"],
        away=info["away_team"],
        season=season,
        week=int(info["week"]),
        n=n,
        seed=seed,
        home_win_prob=float(sim.home_win_prob()[0]),
        mean_margin=float(sim.mean_margin()[0]),
        mean_total=float(sim.mean_total()[0]),
        expected_home=float(mu_home[0]),
        expected_away=float(mu_away[0]),
        margin_quantiles=quantiles(sim.margin[0]),
        home_quantiles=quantiles(sim.home[0]),
        away_quantiles=quantiles(sim.away[0]),
        flags=flags,
        model=model,
    )
