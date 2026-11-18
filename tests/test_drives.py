from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from nflsim.data.drives import DRIVE_COLUMNS, drive_coverage, drives

_SCHEMA = {
    "play_id": pl.Float64,
    "fixed_drive": pl.Float64,
    "posteam": pl.String,
    "play_type": pl.String,
    "yardline_100": pl.Float64,
    "game_seconds_remaining": pl.Float64,
    "total_home_score": pl.Float64,
    "total_away_score": pl.Float64,
    "fixed_drive_result": pl.String,
    "drive_time_of_possession": pl.String,
}


def pbp(rows: list[tuple[Any, ...]]) -> pl.LazyFrame:
    """One game, OAK (old code) at home vs. KC. Rows follow ``_SCHEMA``."""
    frame = pl.DataFrame(rows, schema=_SCHEMA, orient="row")
    return frame.with_columns(
        game_id=pl.lit("g1"),
        season=pl.lit(2019, pl.Int32),
        week=pl.lit(1, pl.Int32),
        season_type=pl.lit("REG"),
        home_team=pl.lit("OAK"),
        defteam=pl.when(pl.col("posteam") == "OAK").then(pl.lit("KC")).otherwise(pl.lit("OAK")),
        half_seconds_remaining=pl.col("game_seconds_remaining") - 1800,
        game_half=pl.lit("Half1"),
        score_differential=pl.lit(0.0),
        drive_play_count=pl.lit(3.0),
        drive_start_transition=pl.lit("KICKOFF"),
    ).lazy()


PLAYS = pbp(
    [
        (1, 1, None, None, None, 3600, 0, 0, "Touchdown", None),  # game start, no offense
        (2, 1, "OAK", "kickoff", 35, 3600, 0, 0, "Touchdown", "2:30"),
        (3, 1, "OAK", "run", 75, 3600, 0, 0, "Touchdown", "2:30"),
        (4, 1, "OAK", "pass", 20, 3500, 6, 0, "Touchdown", "2:30"),
        (5, 1, "OAK", "extra_point", 15, 3450, 7, 0, "Touchdown", "2:30"),
        # KC drive: interception returned for an OAK touchdown; the extra point is OAK's.
        (6, 2, "KC", "kickoff", 35, 3450, 7, 0, "Opp touchdown", "0:40"),
        (7, 2, "KC", "pass", 70, 3445, 13, 0, "Opp touchdown", "0:40"),
        (8, 2, "OAK", "extra_point", 15, 3440, 14, 0, "Opp touchdown", "0:40"),
        (9, 3, "KC", "kickoff", 35, 3440, 14, 0, "Field goal", "4:10"),
        (10, 3, "KC", "pass", 75, 3435, 14, 0, "Field goal", "4:10"),
        (11, 3, None, "timeout", None, 3300, 14, 0, "Field goal", None),
        (12, 3, "KC", "field_goal", 30, 3200, 14, 3, "Field goal", "4:10"),
    ]
)


def test_drives_score_from_running_totals_and_start_after_kickoff() -> None:
    out = drives(PLAYS)
    assert out.columns == DRIVE_COLUMNS
    rows = {r["drive"]: r for r in out.iter_rows(named=True)}
    assert rows[1.0]["offense"] == "LV" and rows[1.0]["defense"] == "KC"
    assert rows[1.0]["start_yardline"] == 75.0  # first play after the kickoff
    assert (rows[1.0]["off_points"], rows[1.0]["def_points"]) == (7.0, 0.0)
    # Defensive touchdown plus the defense's extra point, all on KC's drive.
    assert rows[2.0]["result"] == "Opp touchdown"
    assert (rows[2.0]["off_points"], rows[2.0]["def_points"]) == (0.0, 7.0)
    assert (rows[3.0]["off_points"], rows[3.0]["def_points"]) == (3.0, 0.0)
    assert rows[1.0]["seconds"] == 150.0
    assert rows[3.0]["start_seconds"] == 3435.0


def test_drive_coverage_reconciles_final_scores() -> None:
    d = drives(PLAYS)
    sched = pl.DataFrame(
        {
            "game_id": ["g1"],
            "home_team": ["OAK"],
            "away_team": ["KC"],
            "home_score": [14],
            "away_score": [3],
        }
    )
    row = drive_coverage(d, sched).row(0, named=True)
    assert row["score_match"] == 1.0
    assert row["drives_per_game"] == 3.0
    assert row["touchdown"] == pytest.approx(1 / 3)
    wrong = sched.with_columns(home_score=pl.lit(13))
    assert drive_coverage(d, wrong).row(0, named=True)["score_match"] == 0.0
