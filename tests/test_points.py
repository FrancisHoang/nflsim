from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from nflsim.config import ModelConfig, RatingsConfig
from nflsim.data import normalize_teams
from nflsim.model import build_games, fit_points_model, league_points_as_of, team_game_stats
from nflsim.model.points import PointsModel
from tests.test_ratings import pbp, play, season_rows

RCFG = RatingsConfig(min_wp=0.0, max_wp=1.0)


def schedule(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """Rows: (season, week, home, away, home_score, away_score[, location])."""
    full = [(*r, "Home") if len(r) == 6 else r for r in rows]
    frame = pl.DataFrame(
        full,
        schema=["season", "week", "home_team", "away_team", "home_score", "away_score", "location"],
        orient="row",
    )
    return frame.with_columns(
        pl.col("season", "week").cast(pl.Int32),
        pl.col("home_score", "away_score").cast(pl.Int64),
        pl.format("{}_{}_{}_{}", "season", "week", "away_team", "home_team").alias("game_id"),
        pl.lit("REG").alias("game_type"),
        pl.lit(-3.0).alias("spread_line"),
        pl.lit(44.0).alias("total_line"),
    )


def test_normalize_teams_maps_relocations() -> None:
    frame = pl.DataFrame({"t": ["OAK", "SD", "STL", "KC"]})
    assert normalize_teams(frame, ["t"])["t"].to_list() == ["LV", "LAC", "LA", "KC"]


def test_league_points_pools_prior_season_and_excludes_target_week() -> None:
    sched = schedule(
        [
            (2020, 1, "KC", "BUF", 30, 20),  # 50 points, 2 team-games
            (2021, 1, "KC", "BUF", 10, 10),  # 20 points
            (2021, 2, "KC", "BUF", 99, 99),  # must not affect week 2
        ]
    )
    lg = league_points_as_of(sched)

    def at(season: int, week: int) -> Any:
        return lg.filter(season=season, week=week)["league_ppg"].item()

    assert at(2020, 1) is None
    assert at(2021, 1) == pytest.approx(25.0)
    assert at(2021, 2) == pytest.approx((50 + 20) / 4)


def test_build_games_joins_ratings_with_normalized_teams() -> None:
    # Play-by-play uses LV; the 2019 schedule says OAK.
    kinds = ["pass", "run"] * 5
    rows = [play(2019, 1, "LV", "KC", 0.1, play_type=k) for k in kinds]
    rows += [play(2019, 1, "KC", "LV", 0.3, play_type=k) for k in kinds]
    games = team_game_stats(pbp(rows), RCFG)
    sched = schedule(
        [
            (2019, 1, "OAK", "KC", 17, 24),
            (2019, 2, "KC", "OAK", 28, 10, "Neutral"),
        ]
    )
    out = build_games(sched, games, RCFG)
    week1, week2 = out.sort("week").to_dicts()

    assert week1["home_sign"] == 1.0
    assert not week1["has_ratings"]  # no data before week 1 of the first season
    assert week2["away_team"] == "LV"
    assert week2["home_sign"] == 0.0
    assert week2["has_ratings"]
    assert week2["a_games_played"] == 1
    assert week2["a_off_epa"] < week2["h_off_epa"]


def synthetic_games(n: int, truth: dict[str, float], seed: int = 0) -> pl.DataFrame:
    """Games whose scores follow the points model exactly, with dispersion 6."""
    rng = np.random.default_rng(seed)
    cols: dict[str, Any] = {}
    for p in ("h_", "a_"):
        cols[f"{p}off_epa"] = rng.normal(0, 0.1, n)
        cols[f"{p}def_epa"] = rng.normal(0, 0.1, n)
        cols[f"{p}off_success"] = rng.normal(0.45, 0.03, n)
        cols[f"{p}def_success"] = rng.normal(0.45, 0.03, n)
        cols[f"{p}pace"] = rng.normal(62, 3, n)
        cols[f"{p}league_epa"] = np.zeros(n)
        cols[f"{p}league_success"] = np.full(n, 0.45)
        cols[f"{p}league_pace"] = np.full(n, 62.0)
    sign = (rng.random(n) < 0.9).astype(float)
    frame = pl.DataFrame(cols).with_columns(
        league_ppg=pl.lit(22.0), home_sign=pl.Series(sign), has_ratings=pl.lit(True)
    )
    from nflsim.model.points import design_matrices

    feats = list(truth)[2:]
    x_h, x_a = design_matrices(frame, feats)
    coefs = np.array([truth[f] for f in feats])
    mu_h = 22 + truth["intercept"] + x_h @ coefs + truth["home"] * sign
    mu_a = 22 + truth["intercept"] + x_a @ coefs - truth["home"] * sign
    r = 6.0
    return frame.with_columns(
        home_score=pl.Series(rng.negative_binomial(r, r / (r + mu_h)).astype(float)),
        away_score=pl.Series(rng.negative_binomial(r, r / (r + mu_a)).astype(float)),
    )


def test_fit_recovers_known_coefficients() -> None:
    truth = {"intercept": 0.5, "home": 1.0, "off_epa": 30.0, "def_epa": 25.0, "off_pace": 0.3}
    games = synthetic_games(40_000, truth)
    model = fit_points_model(games, ModelConfig(features=("off_epa", "def_epa", "off_pace")))
    assert model.intercept == pytest.approx(0.5, abs=0.2)
    assert model.home == pytest.approx(1.0, abs=0.15)
    assert model.coefs[0] == pytest.approx(30.0, abs=1.5)
    assert model.coefs[1] == pytest.approx(25.0, abs=1.5)
    assert model.coefs[2] == pytest.approx(0.3, abs=0.05)
    assert model.dispersion == pytest.approx(6.0, rel=0.15)
    assert model.train_games == 40_000


def test_fit_rejects_tiny_samples() -> None:
    games = synthetic_games(20, {"intercept": 0.0, "home": 1.0, "off_epa": 30.0})
    with pytest.raises(ValueError, match="games to fit"):
        fit_points_model(games, ModelConfig(features=("off_epa",)))


def test_expected_points_home_sign_and_floor() -> None:
    games = synthetic_games(3, {"intercept": 0.0, "home": 0.0}).with_columns(
        home_sign=pl.Series([1.0, 0.0, 1.0]),
        h_off_epa=pl.Series([0.0, 0.0, -10.0]),
        a_off_epa=pl.Series([0.0, 0.0, 0.0]),
        a_def_epa=pl.Series([0.0, 0.0, 0.0]),
        h_def_epa=pl.Series([0.0, 0.0, 0.0]),
    )
    model = PointsModel(
        features=("off_epa", "def_epa"),
        intercept=0.0,
        coefs=(30.0, 25.0),
        home=1.0,
        dispersion=6.0,
        min_expected_points=3.0,
        train_games=0,
    )
    mu_h, mu_a = model.expected_points(games)
    assert mu_h[0] - mu_a[0] == pytest.approx(2.0)  # home value counts for both sides
    assert mu_h[1] == pytest.approx(mu_a[1])  # neutral site, equal teams
    assert mu_h[2] == 3.0  # floored


def test_expected_points_requires_league_baseline() -> None:
    games = synthetic_games(2, {"intercept": 0.0, "home": 0.0}).with_columns(
        league_ppg=pl.lit(None, dtype=pl.Float64)
    )
    model = PointsModel(("off_epa",), 0.0, (30.0,), 1.0, 6.0, 3.0, 0)
    with pytest.raises(ValueError, match="league_ppg"):
        model.expected_points(games)


def test_build_and_fit_end_to_end_on_fixture() -> None:
    rows: list[dict[str, Any]] = []
    sched_rows: list[tuple[Any, ...]] = []
    rng = np.random.default_rng(1)
    for season in (2020, 2021):
        rows += season_rows(season, 0.2, -0.2, weeks=8)
        for week in range(1, 9):
            sched_rows.append(
                (season, week, "KC", "BUF", int(rng.integers(17, 35)), int(rng.integers(10, 28)))
            )
    games = build_games(schedule(sched_rows), team_game_stats(pbp(rows), RCFG), RCFG)
    assert games.filter(pl.col("has_ratings")).height == 15
    model = fit_points_model(games.vstack(games), ModelConfig(features=("off_epa",)))
    mu_h, mu_a = model.expected_points(games.filter(pl.col("has_ratings")))
    assert np.all(mu_h > mu_a)  # KC (home, better) favored every week
