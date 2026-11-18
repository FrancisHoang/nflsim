from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl
import pytest

from nflsim.config import ModelConfig, RatingsConfig
from nflsim.data import normalize_teams
from nflsim.model import build_games, fit_points_model, league_points_as_of, team_game_stats
from nflsim.model.points import PointsModel, design_matrices
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
        pl.lit(-150.0).alias("home_moneyline"),
        pl.lit(130.0).alias("away_moneyline"),
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


def test_direct_margin_matches_points_model_when_points_model_is_true() -> None:
    truth = {"intercept": 0.0, "home": 1.0, "off_epa": 30.0}
    games = synthetic_games(40_000, truth)
    cfg = ModelConfig(features=("off_epa",), margin_fit="direct")
    model = fit_points_model(games, cfg)
    assert model.margin_coefs is not None
    assert model.margin_coefs[0] == pytest.approx(30.0, abs=2.0)
    assert model.margin_home == pytest.approx(2.0, abs=0.3)  # both sides' home value
    assert model.dispersion == pytest.approx(6.0, rel=0.15)


def test_direct_margin_captures_cross_effects_the_points_model_misses() -> None:
    # Each team also scores less against a better opposing offense (fewer possessions);
    # a per-team points model with only its own offense cannot see that.
    games = synthetic_games(40_000, {"intercept": 0.0, "home": 1.0, "off_epa": 30.0})
    games = games.with_columns(
        home_score=pl.col("home_score") - 10.0 * pl.col("a_off_epa"),
        away_score=pl.col("away_score") - 10.0 * pl.col("h_off_epa"),
    )
    points = fit_points_model(games, ModelConfig(features=("off_epa",)))
    direct = fit_points_model(games, ModelConfig(features=("off_epa",), margin_fit="direct"))
    assert points.margin_coefs is None
    assert points.coefs[0] == pytest.approx(30.0, abs=2.0)  # margin slope stuck at 30
    assert direct.margin_coefs is not None
    assert direct.margin_coefs[0] == pytest.approx(40.0, abs=2.0)


def test_direct_margin_keeps_total_and_flips_with_sides() -> None:
    games = synthetic_games(4, {"intercept": 0.0, "home": 0.0}).with_columns(
        home_sign=pl.lit(0.0),
        h_off_epa=pl.Series([0.1, -0.1, 0.0, 0.2]),
        a_off_epa=pl.Series([-0.1, 0.1, 0.0, 0.0]),
    )
    base = PointsModel(("off_epa",), 0.0, (30.0,), 1.0, 6.0, 3.0, 0)
    direct = PointsModel(("off_epa",), 0.0, (30.0,), 1.0, 6.0, 3.0, 0, (50.0,), 2.0)
    bh, ba = base.expected_points(games)
    dh, da = direct.expected_points(games)
    assert np.allclose(dh + da, bh + ba)  # total from the points model
    assert np.allclose(dh - da, [10.0, -10.0, 0.0, 10.0])  # 50 * (h - a) at neutral site
    assert dh[0] - da[0] == pytest.approx(-(dh[1] - da[1]))


def test_build_games_adds_player_features_with_league_average_fallback() -> None:
    kinds = ["pass", "run"] * 5
    rows = [play(2019, 1, "LV", "KC", 0.1, play_type=k) for k in kinds]
    rows += [play(2019, 1, "KC", "LV", 0.3, play_type=k) for k in kinds]
    games = team_game_stats(pbp(rows), RCFG)
    sched = schedule([(2019, 1, "OAK", "KC", 17, 24), (2019, 2, "KC", "OAK", 28, 10)])
    features = pl.DataFrame(
        {
            "season": [2019],
            "week": [2],
            "team": ["KC"],
            "off_qb": [0.05],
            "off_rush": [0.01],
            "off_recv": [0.02],
            "player_proj_missing": [0],
        }
    )
    built = build_games(sched, games, RCFG, features).filter(pl.col("week") == 2)
    out = built.row(0, named=True)
    assert out["h_off_qb"] == pytest.approx(0.05)
    assert out["a_off_qb"] == 0.0  # LV has no row: league average
    assert out["a_player_proj_missing"] == 3
    assert out["has_ratings"]

    x_home, x_away = design_matrices(built, ["off_qb"])
    assert x_home[0, 0] == pytest.approx(0.05)
    assert x_away[0, 0] == 0.0
    with pytest.raises(ValueError, match="player features"):
        design_matrices(build_games(sched, games, RCFG), ["off_qb"])


def test_build_games_passes_through_injury_report_flag() -> None:
    kinds = ["pass", "run"] * 5
    rows = [play(2019, 1, "LV", "KC", 0.1, play_type=k) for k in kinds]
    rows += [play(2019, 1, "KC", "LV", 0.3, play_type=k) for k in kinds]
    games = team_game_stats(pbp(rows), RCFG)
    sched = schedule([(2019, 2, "KC", "OAK", 28, 10)])
    features = pl.DataFrame(
        {
            "season": [2019, 2019],
            "week": [2, 2],
            "team": ["KC", "LV"],
            "off_qb": [0.05, 0.0],
            "off_rush": [0.0, 0.0],
            "off_recv": [0.0, 0.0],
            "player_proj_missing": [0, 0],
            "injury_report": [True, False],
        }
    )
    out = build_games(sched, games, RCFG, features).row(0, named=True)
    assert out["h_injury_report"] is True
    assert out["a_injury_report"] is False


def test_expected_points_with_per_run_player_features() -> None:
    games = synthetic_games(4, {"intercept": 0.0, "home": 1.0, "off_epa": 30.0}, seed=3)
    games = games.with_columns(
        h_off_qb=pl.lit(0.05), a_off_qb=pl.lit(-0.02), h_off_rush=pl.lit(0.0)
    )
    model = PointsModel(
        features=("off_epa", "off_qb"),
        intercept=0.5,
        coefs=(30.0, 40.0),
        home=1.0,
        dispersion=6.0,
        min_expected_points=3.0,
        train_games=100,
    )
    mu_h, mu_a = model.expected_points(games)
    n = 3
    same = {"h_off_qb": np.full((4, n), 0.05), "a_off_qb": np.full((4, n), -0.02)}
    run_h, run_a = model.expected_points(games, same)
    assert run_h.shape == (4, n)
    assert run_h == pytest.approx(np.repeat(mu_h[:, None], n, axis=1))
    assert run_a == pytest.approx(np.repeat(mu_a[:, None], n, axis=1))
    # Home QB out in run 1 (feature 0.05 -> -0.10): 40 * 0.15 = 6 points fewer.
    runs = {"h_off_qb": np.array([[0.05, -0.10, 0.05]] * 4)}
    run_h, run_a = model.expected_points(games, runs)
    assert run_h[:, 1] == pytest.approx(mu_h - 6.0)
    assert run_a == pytest.approx(np.repeat(mu_a[:, None], n, axis=1))
    # Runs for features the model does not use are ignored.
    unused, _ = model.expected_points(games, {"h_off_rush": np.zeros((4, n))})
    assert unused.shape == (4,)
    direct = replace(model, margin_coefs=(25.0, 40.0), margin_home=1.5)
    d_h, d_a = direct.expected_points(games, runs)
    flat_h, flat_a = direct.expected_points(games)
    assert (d_h + d_a)[:, 0] == pytest.approx(flat_h + flat_a)
    # Run 1: the margin fit moves the margin by 40 * -0.15; the total drops by 6 as well.
    assert (d_h - d_a)[:, 1] == pytest.approx(flat_h - flat_a - 6.0)
    assert (d_h + d_a)[:, 1] == pytest.approx(flat_h + flat_a - 6.0)
