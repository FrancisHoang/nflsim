from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from nflsim.config import StatLinesConfig
from nflsim.model.stat_lines import (
    OPPORTUNITIES,
    STATS,
    box_scores,
    expected_stats,
    fit_ranges,
    player_rates_as_of,
    team_volume_as_of,
)
from nflsim.stat_eval import baseline_per_game, evaluate_stat_lines

CFG = StatLinesConfig(
    volume_prior_games=1.0,
    carryover=0.5,
    rate_prior={"attempts": 10.0, "carries": 10.0, "targets": 10.0},
)


def raw_stats(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """(season, week, team, player_id, attempts, passing_yards, carries, rushing_yards,
    targets, receiving_yards); everything else 0."""
    frame = pl.DataFrame(
        rows,
        schema=[
            "season",
            "week",
            "team",
            "player_id",
            "attempts",
            "passing_yards",
            "carries",
            "rushing_yards",
            "targets",
            "receiving_yards",
        ],
        orient="row",
    )
    zeros = [c for c in STATS if c not in frame.columns]
    return frame.with_columns(
        game_id=pl.format("{}_{}_{}", "season", "week", "team"),
        **{c: pl.lit(0.0) for c in zeros},
    )


RAW = raw_stats(
    [
        # 2023: OAK (old code) passes 30 and runs 20 per game.
        (2023, 1, "OAK", "qb", 30, 210, 4, 40, 0, 0),
        (2023, 1, "OAK", "rb", 0, 0, 16, 64, 4, 20),
        (2024, 1, "OAK", "qb", 40, 320, 2, 10, 0, 0),
        (2024, 1, "OAK", "rb", 0, 0, 20, 100, 6, 60),
        (2024, 2, "OAK", "qb", 50, 500, 0, 0, 0, 0),  # week 2 must not leak into week 2
        (2024, 2, "OAK", "rb", 0, 0, 30, 300, 0, 0),
    ]
)
BOX = box_scores(RAW)
POSITIONS = pl.DataFrame(
    {
        "season": [2023, 2023, 2024, 2024, 2024, 2024],
        "week": [1, 1, 1, 1, 2, 2],
        "gsis_id": ["qb", "rb", "qb", "rb", "qb", "rb"],
        "position": ["QB", "RB", "QB", "RB", "QB", "RB"],
    },
    schema_overrides={"season": pl.Int32, "week": pl.Int32},
)
TARGET = pl.DataFrame(
    {"season": [2024], "week": [2], "team": ["LV"]},
    schema_overrides={"season": pl.Int32, "week": pl.Int32},
)


def test_box_scores_normalizes_teams_and_fills_zeros() -> None:
    assert set(BOX["team"]) == {"LV"}
    assert BOX.columns[:5] == ["season", "week", "game_id", "team", "gsis_id"]
    assert BOX["passing_tds"].to_list() == [0.0] * 6


def test_team_volume_blends_this_season_last_season_and_league() -> None:
    vol = team_volume_as_of(BOX, TARGET, CFG).row(0, named=True)
    # Attempts: this season (week 1 only) 40 over 1 game, last season 30 over 1 game at
    # weight 0.5, league average (same window, one team) 70 / 2 = 35 at prior 1 game.
    assert vol["vol_attempts"] == pytest.approx((40 + 0.5 * 30 + 1 * 35) / (1 + 0.5 + 1))
    carries_league = (22 + 20) / 2
    assert vol["vol_carries"] == pytest.approx((22 + 0.5 * 20 + carries_league) / (1 + 0.5 + 1))


def test_player_rates_shrink_and_rate_qb_rushing_per_attempt() -> None:
    keys = pl.DataFrame(
        {"season": [2024, 2024], "week": [2, 2], "gsis_id": ["qb", "rb"], "position": ["QB", "RB"]},
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )
    rates = {
        r["gsis_id"]: r for r in player_rates_as_of(BOX, POSITIONS, keys, CFG).iter_rows(named=True)
    }
    # QB passing yards per attempt: own 320/40 this season, 210/30 last (weight 0.5), and
    # the QB average (530 / 70) with a prior of 10 attempts.
    want = (320 + 0.5 * 210 + 10 * (530 / 70)) / (40 + 0.5 * 30 + 10)
    assert rates["qb"]["rate_passing_yards"] == pytest.approx(want)
    want_rush = (10 + 0.5 * 40 + 10 * (50 / 70)) / (40 + 0.5 * 30 + 10)
    assert rates["qb"]["rate_rushing_yards_per_att"] == pytest.approx(want_rush)
    assert np.isfinite(rates["rb"]["rate_passing_yards"])  # no 0/0 NaN for non-passers


def test_expected_stats_multiply_volume_share_and_rate() -> None:
    keys = pl.DataFrame(
        {"season": [2024, 2024], "week": [2, 2], "gsis_id": ["qb", "rb"], "position": ["QB", "RB"]},
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )
    rates = player_rates_as_of(BOX, POSITIONS, keys, CFG)
    vol = team_volume_as_of(BOX, TARGET, CFG)
    shares = pl.DataFrame(
        {
            "season": [2024] * 3,
            "week": [2] * 3,
            "team": ["LV"] * 3,
            "gsis_id": ["qb", "rb", "rb"],
            "role": ["dropback", "carry", "target"],
            "share": [1.0, 0.5, 0.2],
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )
    exp = {r["gsis_id"]: r for r in expected_stats(shares, vol, rates).iter_rows(named=True)}
    v, rt = vol.row(0, named=True), {r["gsis_id"]: r for r in rates.iter_rows(named=True)}
    assert exp["qb"]["exp_passing_yards"] == pytest.approx(
        v["vol_attempts"] * rt["qb"]["rate_passing_yards"]
    )
    # The QB's rushing comes from his dropback share, not designed carries.
    assert exp["qb"]["exp_rushing_yards"] == pytest.approx(
        v["vol_attempts"] * rt["qb"]["rate_rushing_yards_per_att"]
    )
    assert exp["rb"]["exp_rushing_yards"] == pytest.approx(
        0.5 * v["vol_carries"] * rt["rb"]["rate_rushing_yards"]
    )
    assert exp["rb"]["exp_receiving_yards"] == pytest.approx(
        0.2 * v["vol_targets"] * rt["rb"]["rate_receiving_yards"]
    )
    assert exp["rb"]["exp_passing_yards"] == 0.0


def test_ranges_and_evaluation() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    expected = rng.uniform(5, 80, n)
    actual = np.maximum(expected + rng.normal(0, 1, n) * np.sqrt(expected) * 3, 0)
    pairs = pl.DataFrame(
        {
            "season": np.where(np.arange(n) < n // 2, 2021, 2024),
            "game_id": [f"g{i // 20}" for i in range(n)],
            "gsis_id": [f"p{i}" for i in range(n)],
            **{f"exp_{s}": expected for s in STATS},
            **{s: actual for s in STATS},
            **{f"base_{s}": expected * 1.3 for s in STATS},  # a worse baseline
        }
    )
    table, ranges = evaluate_stat_lines(pairs, [2021], [2024], 0.1, 0.9, 0.5, 300, 1)
    row = table.filter(pl.col("stat") == "receiving_yards").row(0, named=True)
    assert row["diff"] < 0 and row["ci_hi"] < 0
    assert row["in_range"] == pytest.approx(0.8, abs=0.04)
    assert ranges.low["receiving_yards"] < 0 < ranges.high["receiving_yards"]
    lo, hi = fit_ranges(pairs, 0.1, 0.9, 0.5).interval("receptions", np.array([0.0, 25.0]))
    assert lo[0] == 0.0 and hi[0] == 0.0 and lo[1] < 25 < hi[1]
    a, _ = evaluate_stat_lines(pairs, [2021], [2024], 0.1, 0.9, 0.5, 300, 1)
    assert a.equals(table)  # reproducible


def test_baseline_uses_earlier_games_then_last_season() -> None:
    keys = pl.DataFrame(
        {"season": [2024, 2024], "week": [1, 2], "gsis_id": ["rb", "rb"]},
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )
    base = {r["week"]: r for r in baseline_per_game(BOX, keys).iter_rows(named=True)}
    assert base[1]["base_rushing_yards"] == pytest.approx(64.0)  # last season's average
    assert base[2]["base_rushing_yards"] == pytest.approx(100.0)  # week 1 this season
    assert set(OPPORTUNITIES) == {"attempts", "carries", "targets"}
