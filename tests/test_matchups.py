from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from nflsim.config import MATCHUP_METRICS, MatchupsConfig
from nflsim.matchup_eval import pass_epa_gate, pressure_gate, rating_skill
from nflsim.model.matchups import (
    LEAGUE_COLUMNS,
    LOG5,
    MATCHUP_COLUMNS,
    PressureModel,
    dropback_matchups,
    expected_pass_epa,
    fit_pressure_model,
    matchup_game_stats,
    matchup_ratings_as_of,
    pressure_terms,
)

METRICS = list(MATCHUP_METRICS)

# Small priors so the arithmetic is easy to follow.
CFG = MatchupsConfig(
    prior_dropbacks={"pressure": 4.0, "epa_pressured": 2.0, "epa_clean": 2.0, "epa": 4.0},
    carryover=0.5,
)


def dropbacks(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """(season, week, game_id, offense, defense, epa, pressure, scramble)."""
    return pl.DataFrame(
        rows,
        schema={
            "season": pl.Int32,
            "week": pl.Int32,
            "game_id": pl.String,
            "offense": pl.String,
            "defense": pl.String,
            "epa": pl.Float64,
            "pressure": pl.Boolean,
            "scramble": pl.Boolean,
        },
        orient="row",
    ).with_columns(play_id=pl.int_range(pl.len()).cast(pl.Float64))


WEEK1 = [
    # AAA offense vs. BBB defense: 2 pressured (-1, -1), 2 clean (1.0, 0.5).
    (2023, 1, "g1", "AAA", "BBB", -1.0, True, False),
    (2023, 1, "g1", "AAA", "BBB", -1.0, True, False),
    (2023, 1, "g1", "AAA", "BBB", 1.0, False, False),
    (2023, 1, "g1", "AAA", "BBB", 0.5, False, False),
    (2023, 1, "g1", "AAA", "BBB", 3.0, None, True),  # scramble: left out
    (2023, 1, "g1", "AAA", "BBB", 3.0, None, False),  # unknown pressure: left out
    # BBB offense vs. AAA defense: 4 clean dropbacks at 0 EPA.
    *[(2023, 1, "g1", "BBB", "AAA", 0.0, False, False)] * 4,
]
# Week 2 is extreme; it must not affect ratings for week 2 itself.
WEEK2 = [(2023, 2, "g2", "AAA", "BBB", -9.0, True, False)] * 10
PLAYS = dropbacks([*WEEK1, *WEEK2])


def targets(rows: list[tuple[int, int, str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["season", "week", "team"], orient="row").with_columns(
        pl.col("season", "week").cast(pl.Int32)
    )


def test_game_stats_count_each_side_and_skip_unknowns() -> None:
    stats = matchup_game_stats(PLAYS).filter(pl.col("week") == 1)
    off = stats.filter((pl.col("team") == "AAA") & (pl.col("side") == "off")).row(0, named=True)
    assert off["dropbacks"] == 4.0
    assert off["pressures"] == 2.0
    assert off["epa_pressured_sum"] == pytest.approx(-2.0)
    assert off["epa_clean_sum"] == pytest.approx(1.5)
    defense = stats.filter((pl.col("team") == "BBB") & (pl.col("side") == "def"))
    assert defense.row(0, named=True)["pressures"] == 2.0


def test_ratings_are_shrunk_deviations_from_weeks_before() -> None:
    stats = matchup_game_stats(PLAYS)
    out = matchup_ratings_as_of(
        stats, targets([(2023, 1, "AAA"), (2023, 2, "AAA"), (2023, 2, "BBB")]), CFG
    )
    assert out.columns[:3] == ["season", "week", "team"]
    assert set(MATCHUP_COLUMNS + LEAGUE_COLUMNS) <= set(out.columns)
    week1 = out.filter(pl.col("week") == 1).row(0, named=True)
    assert week1["off_pressure"] is None  # no earlier data at all
    a = out.filter((pl.col("week") == 2) & (pl.col("team") == "AAA")).row(0, named=True)
    # League week 1: 2 pressures on 8 dropbacks; clean EPA 1.5 over 6 clean dropbacks.
    assert a["league_pressure"] == pytest.approx(0.25)
    assert a["league_epa_clean"] == pytest.approx(0.25)
    # AAA offense: (2 - 4 * 0.25) / (4 + 4); AAA defense: (0 - 4 * 0.25) / (4 + 4).
    assert a["off_pressure"] == pytest.approx(0.125)
    assert a["def_pressure"] == pytest.approx(-0.125)
    # AAA clean EPA: (1.5 - 2 * 0.25) / (2 + 2).
    assert a["off_epa_clean"] == pytest.approx(0.25)
    assert a["off_dropbacks"] == 4.0
    assert a["matchup_prior"] == "league"
    b = out.filter((pl.col("week") == 2) & (pl.col("team") == "BBB")).row(0, named=True)
    assert b["def_pressure"] == pytest.approx(0.125)


def test_new_season_starts_from_carried_over_deviation() -> None:
    stats = matchup_game_stats(dropbacks(WEEK1))
    out = matchup_ratings_as_of(stats, targets([(2024, 1, "AAA")]), CFG).row(0, named=True)
    # Last season's shrunk deviation 0.125 (week 1 only), times carryover 0.5.
    assert out["off_pressure"] == pytest.approx(0.0625)
    assert out["league_pressure"] == pytest.approx(0.25)  # last season's league rate
    assert out["matchup_prior"] == "prior_season"
    assert out["off_dropbacks"] == 0.0


def test_rating_skill_is_zero_for_league_average_predictions() -> None:
    stats = matchup_game_stats(PLAYS)
    flat = MatchupsConfig(
        prior_dropbacks={"pressure": 1e9, "epa_pressured": 1e9, "epa_clean": 1e9, "epa": 1e9},
        carryover=0.0,
    )
    ratings = matchup_ratings_as_of(stats, stats.select("season", "week", "team"), flat)
    skill = rating_skill(stats, ratings, [2023])
    assert skill.height == 8  # 4 metrics x 2 sides
    pressure = skill.filter((pl.col("side") == "off") & (pl.col("metric") == "pressure"))
    assert pressure["skill"][0] == pytest.approx(0.0, abs=1e-6)


def synthetic_terms(n: int, truth: tuple[float, float, float, float], seed: int) -> pl.DataFrame:
    """Pressure outcomes drawn from a known logistic model on random terms."""
    rng = np.random.default_rng(seed)
    base = np.full(n, np.log(0.3 / 0.7))
    lo, ld = rng.normal(0, 0.3, n), rng.normal(0, 0.3, n)
    b0, b1, b2, b3 = truth
    p = 1 / (1 + np.exp(-(base + b0 + b1 * lo + b2 * ld + b3 * lo * ld)))
    return pl.DataFrame(
        {
            "base": base,
            "lo": lo,
            "ld": ld,
            "pressure": rng.random(n) < p,
            "season": np.where(np.arange(n) < n // 2, 2021, 2024).astype(np.int32),
            "game_id": [f"g{i // 40}" for i in range(n)],
        }
    )


def test_fit_pressure_model_recovers_known_coefficients() -> None:
    terms = synthetic_terms(200_000, (0.1, 1.2, 0.7, 0.8), seed=1)
    m = fit_pressure_model(terms, matchup=True)
    assert m.intercept == pytest.approx(0.1, abs=0.03)
    assert m.w_off == pytest.approx(1.2, abs=0.05)
    assert m.w_def == pytest.approx(0.7, abs=0.05)
    assert m.w_matchup == pytest.approx(0.8, abs=0.2)
    additive = fit_pressure_model(terms, matchup=False)
    assert additive.w_matchup == 0.0


def test_log5_and_league_probabilities() -> None:
    frame = pressure_terms(
        pl.DataFrame({"league_pressure": [0.3], "off_pressure": [0.05], "def_pressure": [-0.1]})
    )
    p = float(LOG5.probability(frame)[0])
    odds = (0.35 / 0.65) * (0.2 / 0.8) / (0.3 / 0.7)  # log5 on the odds scale
    assert p == pytest.approx(odds / (1 + odds))
    league = PressureModel(w_off=0.0, w_def=0.0)
    assert float(league.probability(frame)[0]) == pytest.approx(0.3)


def test_expected_pass_epa_additive_and_mixture() -> None:
    row = {f"{side}_{m}": [0.0] for side in ("league", "off", "def") for m in METRICS}
    row.update(
        league_epa=[0.1], off_epa=[0.05], def_epa=[-0.02],
        league_epa_pressured=[-0.4], off_epa_pressured=[0.1], def_epa_pressured=[0.0],
        league_epa_clean=[0.25], off_epa_clean=[0.0], def_epa_clean=[-0.05],
    )  # fmt: skip
    out = expected_pass_epa(pl.DataFrame(row), np.array([0.25])).row(0, named=True)
    assert out["additive"] == pytest.approx(0.13)
    assert out["mixture"] == pytest.approx(0.25 * -0.3 + 0.75 * 0.2)


def test_dropback_matchups_uses_offense_and_defense_ratings() -> None:
    stats = matchup_game_stats(PLAYS)
    ratings = matchup_ratings_as_of(stats, stats.select("season", "week", "team"), CFG)
    frame = dropback_matchups(PLAYS, ratings)
    assert set(frame["week"].to_list()) == {2}  # week 1 has no ratings yet
    row = frame.row(0, named=True)
    assert row["offense"] == "AAA"
    assert row["off_pressure"] == pytest.approx(0.125)  # AAA's offense
    assert row["def_pressure"] == pytest.approx(0.125)  # BBB's defense


def test_gates_are_reproducible_and_compare_with_additive() -> None:
    terms = synthetic_terms(20_000, (0.0, 1.0, 1.0, 0.0), seed=2)
    league = 0.3
    frame = terms.with_columns(
        league_pressure=pl.lit(league),
        off_pressure=(1 / (1 + (-(pl.col("lo") + pl.col("base"))).exp())) - league,
        def_pressure=(1 / (1 + (-(pl.col("ld") + pl.col("base"))).exp())) - league,
        offense=pl.lit("AAA"),
        epa=pl.when(pl.col("pressure")).then(-0.4).otherwise(0.2),
        **{f"{side}_{m}": pl.lit(0.0) for side in ("off", "def") for m in METRICS[1:]},
        **{f"league_{m}": pl.lit(v) for m, v in zip(METRICS[1:], (-0.4, 0.2, 0.02), strict=True)},
    ).drop("base", "lo", "ld")
    a, models = pressure_gate(frame, [2021], [2024], n_boot=200, seed=5)
    b, _ = pressure_gate(frame, [2021], [2024], n_boot=200, seed=5)
    assert a.equals(b)
    assert a["model"].to_list() == ["league", "log5", "additive", "matchup"]
    additive = a.filter(pl.col("model") == "additive").row(0, named=True)
    assert additive["vs_additive"] == 0.0
    league_row = a.filter(pl.col("model") == "league").row(0, named=True)
    assert league_row["vs_additive"] > 0  # the ratings carry real signal here
    epa = pass_epa_gate(frame, models, [2024], n_boot=200, seed=5)
    assert epa["prediction"].to_list() == [
        "league", "additive", "mixture/log5", "mixture/additive", "mixture/matchup",
    ]  # fmt: skip
    assert epa.equals(pass_epa_gate(frame, models, [2024], n_boot=200, seed=5))
