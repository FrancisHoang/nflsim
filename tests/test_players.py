from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl
import pytest

from nflsim.data import (
    id_coverage,
    pfr_crosswalk,
    player_snaps,
    roster_weeks,
    unmatched_snaps,
)


def players(rows: Sequence[tuple[str | None, str | None, str | None]]) -> pl.DataFrame:
    """``players`` crosswalk rows: (gsis_id, pfr_id, position_group)."""
    return pl.DataFrame(rows, schema=["gsis_id", "pfr_id", "position_group"], orient="row")


PLAYERS = players(
    [
        ("00-1", "QbaaAa00", "QB"),
        ("00-2", "WrbbBb00", "WR"),
        ("00-3", "DlccCc00", "DL"),
        ("00-4", None, "LB"),  # no PFR ID
    ]
)


def rosters(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """``rosters_weekly`` rows: (season, week, gsis_id, name, team, position, status)."""
    return pl.DataFrame(
        rows,
        schema=["season", "week", "gsis_id", "full_name", "team", "position", "status"],
        orient="row",
    )


def snaps(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """``snap_counts`` rows: (season, week, pfr_id, name, team, opp, pos, off, def)."""
    frame = pl.DataFrame(
        rows,
        schema=[
            "season",
            "week",
            "pfr_player_id",
            "player",
            "team",
            "opponent",
            "position",
            "offense_snaps",
            "defense_snaps",
        ],
        orient="row",
    )
    return frame.with_columns(
        game_id=pl.format("{}_{}", "season", "week"),
        game_type=pl.lit("REG"),
        offense_pct=pl.col("offense_snaps") / 60,
        defense_pct=pl.col("defense_snaps") / 60,
        st_snaps=pl.lit(0.0),
        st_pct=pl.lit(0.0),
    )


def test_crosswalk_maps_pfr_to_gsis_and_skips_missing_ids() -> None:
    xwalk = pfr_crosswalk(PLAYERS)
    assert dict(xwalk.select("pfr_id", "gsis_id").iter_rows()) == {
        "QbaaAa00": "00-1",
        "WrbbBb00": "00-2",
        "DlccCc00": "00-3",
    }


@pytest.mark.parametrize(
    "rows",
    [
        [("00-1", "X", "QB"), ("00-2", "X", "WR")],  # one PFR ID, two players
        [("00-1", "X", "QB"), ("00-1", "Y", "QB")],  # one player, two PFR IDs
    ],
)
def test_crosswalk_rejects_ambiguous_ids(rows: list[tuple[str, str, str]]) -> None:
    with pytest.raises(ValueError, match="more than one player"):
        pfr_crosswalk(players(rows))


def test_roster_weeks_dedupes_normalizes_and_falls_back() -> None:
    frame = roster_weeks(
        rosters(
            [
                # Listed by two teams in one week: the active row wins.
                (2019, 2, "00-3", "Dee Ell", "NYG", "DL", "RES"),
                (2019, 2, "00-3", "Dee Ell", "NYJ", "DL", "ACT"),
                (2019, 2, "00-1", "Cue Bee", "OAK", "QB", "ACT"),  # relocated team
                (2019, 2, "00-4", "Lee Bee", "NE", None, "ACT"),  # roster lacks position
                (2019, 2, None, "No Id", "NE", "WR", "ACT"),  # cannot be linked
                (2020, 1, "00-1", "Cue Bee", "LV", "WR", "ACT"),  # position changed
            ]
        ),
        PLAYERS,
    )
    assert frame.height == 4
    rows = {(r["season"], r["gsis_id"]): r for r in frame.iter_rows(named=True)}
    assert rows[(2019, "00-3")]["team"] == "NYJ"
    assert rows[(2019, "00-1")]["team"] == "LV"
    assert rows[(2019, "00-4")]["position"] == "LB"
    assert rows[(2019, "00-4")]["position_source"] == "players"
    assert rows[(2019, "00-1")]["position"] == "QB"
    assert rows[(2020, "00-1")]["position"] == "WR"
    assert rows[(2020, "00-1")]["position_source"] == "roster"


def test_player_snaps_flags_unmatched_rows_instead_of_dropping() -> None:
    frame = player_snaps(
        snaps(
            [
                (2019, 1, "QbaaAa00", "Cue Bee", "OAK", "SD", "QB", 60.0, 0.0),
                (2019, 1, "ZzzzZz00", "Unknown", "OAK", "SD", "G", 30.0, 0.0),
            ]
        ),
        PLAYERS,
    )
    assert frame.height == 2
    assert frame["team"].to_list() == ["LV", "LV"]
    assert frame["opponent"].to_list() == ["LAC", "LAC"]
    assert frame["gsis_id"].to_list() == ["00-1", None]
    assert frame["id_matched"].to_list() == [True, False]


def test_id_coverage_weights_by_snaps_and_plays() -> None:
    snap_frame = player_snaps(
        snaps(
            [
                (2019, 1, "QbaaAa00", "Cue Bee", "LV", "LAC", "QB", 60.0, 0.0),
                (2019, 1, "ZzzzZz00", "Unknown", "LV", "LAC", "G", 20.0, 0.0),
                (2019, 1, "DlccCc00", "Dee Ell", "LAC", "LV", "DL", 0.0, 50.0),
            ]
        ),
        PLAYERS,
    )
    roster_frame = roster_weeks(
        rosters(
            [
                (2019, 1, "00-1", "Cue Bee", "LV", "QB", "ACT"),
                (2019, 1, "00-2", "Wes Arr", "LV", "WR", "ACT"),
            ]
        ),
        PLAYERS,
    )
    pbp_ids = pl.DataFrame(
        {
            "season": [2019] * 4,
            "passer_player_id": ["00-1", "00-1", None, "00-1"],
            "rusher_player_id": [None, None, "00-9", None],  # not on any roster
            "receiver_player_id": ["00-2", "00-2", None, None],
        }
    )
    report = id_coverage(snap_frame, pbp_ids, roster_frame).row(0, named=True)
    assert report["snap_rows"] == 3
    assert report["snap_rows_matched"] == pytest.approx(2 / 3)
    assert report["off_snaps_matched"] == pytest.approx(60 / 80)
    assert report["def_snaps_matched"] == pytest.approx(1.0)
    assert report["passer_plays_linked"] == pytest.approx(1.0)
    assert report["rusher_plays_linked"] == pytest.approx(0.0)
    assert report["receiver_plays_linked"] == pytest.approx(1.0)


def test_unmatched_snaps_lists_one_row_per_player_most_snaps_first() -> None:
    frame = player_snaps(
        snaps(
            [
                (2024, 1, "AndeAl01", "Al Anderson", "LV", "KC", "G", 30.0, 0.0),
                (2025, 1, "AndeAl01", "Al Anderson", "LV", "KC", "T", 40.0, 0.0),
                (2024, 1, "ParkSt00", "Steve Parker", "KC", "LV", "FS", 0.0, 50.0),
                (2024, 1, "QbaaAa00", "Cue Bee", "LV", "KC", "QB", 60.0, 0.0),
            ]
        ),
        PLAYERS,
    )
    worst = unmatched_snaps(frame, top=5)
    assert worst["pfr_player_id"].to_list() == ["AndeAl01", "ParkSt00"]
    first = worst.row(0, named=True)
    assert first["snap_position"] == "G/T"
    assert first["offense_snaps"] == 70.0
    assert (first["first_season"], first["last_season"]) == (2024, 2025)
