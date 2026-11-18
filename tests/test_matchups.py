from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from nflsim.config import MatchupsConfig
from nflsim.matchup_eval import rating_skill
from nflsim.model.matchups import (
    LEAGUE_COLUMNS,
    MATCHUP_COLUMNS,
    matchup_game_stats,
    matchup_ratings_as_of,
)

# Small priors so the arithmetic is easy to follow.
CFG = MatchupsConfig(
    prior_dropbacks={"pressure": 4.0, "epa_pressured": 2.0, "epa_clean": 2.0}, carryover=0.5
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
    )


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
        prior_dropbacks={"pressure": 1e9, "epa_pressured": 1e9, "epa_clean": 1e9}, carryover=0.0
    )
    ratings = matchup_ratings_as_of(stats, stats.select("season", "week", "team"), flat)
    skill = rating_skill(stats, ratings, [2023])
    assert skill.height == 6
    pressure = skill.filter((pl.col("side") == "off") & (pl.col("metric") == "pressure"))
    assert pressure["skill"][0] == pytest.approx(0.0, abs=1e-6)
