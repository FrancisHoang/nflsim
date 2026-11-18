from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from nflsim.data import DROPBACK_COLUMNS, dropbacks, pfr_team_pressure
from nflsim.matchup_eval import coverage


def pbp(rows: list[tuple[Any, ...]]) -> pl.LazyFrame:
    """(season, game_id, play_id, posteam, defteam, qb_dropback, epa, sack, scramble, two_pt)."""
    frame = pl.DataFrame(
        rows,
        schema={
            "season": pl.Int32,
            "game_id": pl.String,
            "play_id": pl.Float64,
            "posteam": pl.String,
            "defteam": pl.String,
            "qb_dropback": pl.Float64,
            "epa": pl.Float64,
            "sack": pl.Float64,
            "qb_scramble": pl.Float64,
            "two_point_attempt": pl.Float64,
        },
        orient="row",
    )
    return frame.with_columns(
        week=pl.lit(1, pl.Int32),
        season_type=pl.lit("REG"),
        passer_player_id=pl.lit("qb"),
        qb_hit=pl.lit(0.0),
    ).lazy()


def participation(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """(game_id, play_id, was_pressure, man_zone, coverage)."""
    frame = pl.DataFrame(
        rows,
        schema={
            "nflverse_game_id": pl.String,
            "play_id": pl.Float64,
            "was_pressure": pl.Boolean,
            "defense_man_zone_type": pl.String,
            "defense_coverage_type": pl.String,
        },
        orient="row",
    )
    return frame.with_columns(
        number_of_pass_rushers=pl.lit(4, pl.Int32), defenders_in_box=pl.lit(6, pl.Int32)
    )


PBP = pbp(
    [
        (2022, "g1", 1.0, "OAK", "KC", 1.0, 0.5, 0.0, 0.0, 0.0),  # clean
        (2022, "g1", 2.0, "OAK", "KC", 1.0, -2.0, 1.0, 0.0, 0.0),  # sack, no flag (pre-2023)
        (2022, "g1", 3.0, "OAK", "KC", 1.0, 1.0, 0.0, 1.0, 0.0),  # scramble, no flag
        (2022, "g1", 4.0, "OAK", "KC", 1.0, -0.4, 0.0, 0.0, 0.0),  # pressured
        (2022, "g1", 5.0, "OAK", "KC", 0.0, 0.2, 0.0, 0.0, 0.0),  # run: not a dropback
        (2022, "g1", 6.0, "OAK", "KC", 1.0, 0.9, 0.0, 0.0, 1.0),  # two-point try
        (2022, "g1", 7.0, "OAK", "KC", 1.0, 0.3, 0.0, 0.0, 0.0),  # no participation row
    ]
)
PART = participation(
    [
        ("g1", 1.0, False, "ZONE_COVERAGE", "COVER_3"),
        ("g1", 2.0, None, "MAN_COVERAGE", ""),
        ("g1", 3.0, None, "", None),
        ("g1", 4.0, True, "MAN_COVERAGE", "COVER_1"),
        ("g1", 5.0, None, "", ""),
    ]
)


def test_dropbacks_counts_sacks_as_pressure_and_cleans_unknowns() -> None:
    out = dropbacks(PBP, PART)
    assert out.columns == DROPBACK_COLUMNS
    assert out["play_id"].to_list() == [1.0, 2.0, 3.0, 4.0, 7.0]
    rows = {r["play_id"]: r for r in out.iter_rows(named=True)}
    assert rows[1.0]["offense"] == "LV"  # current team code
    assert rows[1.0]["pressure"] is False
    assert rows[2.0]["pressure_flag"] is None
    assert rows[2.0]["pressure"] is True  # sack counts as pressure
    assert rows[2.0]["coverage"] is None  # blank -> unknown
    assert rows[3.0]["scramble"] is True
    assert rows[3.0]["pressure"] is None
    assert rows[3.0]["man_zone"] is None
    assert rows[4.0]["man_zone"] == "man"
    assert rows[1.0]["man_zone"] == "zone"
    assert rows[7.0]["matched"] is False
    assert rows[7.0]["pressure"] is None


def test_pfr_team_pressure_sums_passers_per_team_game() -> None:
    raw = pl.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "season": [2022, 2022, 2022],
            "week": [1, 1, 1],
            "team": ["OAK", "OAK", "KC"],
            "opponent": ["KC", "KC", "OAK"],
            "times_pressured": [5.0, 1.0, 7.0],
            "times_sacked": [2.0, 0.0, 1.0],
        }
    )
    out = pfr_team_pressure(raw)
    lv = out.filter(pl.col("offense") == "LV").row(0, named=True)
    assert lv["pfr_pressures"] == 6.0
    assert lv["pfr_sacks"] == 2.0
    assert lv["defense"] == "KC"


def test_coverage_leaves_out_scrambles_and_compares_with_pfr() -> None:
    plays = dropbacks(PBP, PART)
    pfr = pl.DataFrame({"game_id": ["g1"], "offense": ["LV"], "pfr_pressures": [4.0]})
    row = coverage(plays, pfr).row(0, named=True)
    assert row["dropbacks"] == 5
    assert row["sacks_flagged"] == 0.0
    # Non-scramble dropbacks: plays 1, 2, 4, 7; pressure known on 1, 2, 4.
    assert row["pressure_known"] == pytest.approx(3 / 4)
    assert row["pressure_rate"] == pytest.approx(2 / 3)
    assert row["scramble_known"] == 0.0
    assert row["epa_pressured"] == pytest.approx((-2.0 - 0.4) / 2)
    assert row["epa_clean"] == pytest.approx(0.5)
    assert row["pfr_ratio"] == pytest.approx(2 / 4)
    no_pfr = coverage(plays, None).row(0, named=True)
    assert no_pfr["pfr_ratio"] is None
