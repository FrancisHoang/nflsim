from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from nflsim.config import PRACTICES, REPORT_STATUSES, InjuriesConfig, load_settings
from nflsim.data import depth_chart_weeks, injury_reports, kickoffs
from nflsim.injury_eval import (
    brier_scores,
    coverage,
    depth_coverage,
    rate_table,
    report_outcomes,
)
from nflsim.model.injuries import UNLISTED, attach_play_probability, play_probability_table

UTC = dt.UTC


def injuries(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """Raw ``injuries`` rows.

    (season, week, team, gsis_id, position, report, practice, modified)
    """
    frame = pl.DataFrame(
        rows,
        schema={
            "season": pl.Float64,  # stacked cache frames come back as floats
            "week": pl.Float64,
            "team": pl.String,
            "gsis_id": pl.String,
            "position": pl.String,
            "report_status": pl.String,
            "practice_status": pl.String,
            "date_modified": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )
    return frame.with_columns(
        game_type=pl.lit("REG"),
        full_name=pl.col("gsis_id"),
        report_primary_injury=pl.lit("Knee"),
        practice_primary_injury=pl.lit("Ankle"),
    )


T0 = dt.datetime(2024, 1, 1, tzinfo=UTC)
T1 = dt.datetime(2024, 1, 2, tzinfo=UTC)
DNP = "Did Not Participate In Practice"
LIM = "Limited Participation in Practice"
FULL = "Full Participation in Practice"


def test_injury_reports_maps_statuses_and_keeps_latest_row() -> None:
    raw = injuries(
        [
            (2019, 3, "OAK", "00-1", "QB", "Questionable", LIM, T0),
            (2019, 3, "OAK", "00-1", "QB", "Out", DNP, T1),  # later revision wins
            (2019, 3, "KC", "00-2", "WR", None, FULL, T0),
            (2019, 3, "KC", "00-3", "CB", "Note", None, T0),
            (2019, 3, "KC", None, "S", "Out", DNP, T0),  # no ID: dropped
        ]
    )
    out = injury_reports(raw).sort("gsis_id")
    assert out["gsis_id"].to_list() == ["00-1", "00-2", "00-3"]
    assert out["team"].to_list() == ["LV", "KC", "KC"]
    assert out["status"].to_list() == ["out", "none", "none"]
    assert out["practice"].to_list() == ["dnp", "full", "none"]
    assert out["season"].dtype == pl.Int32
    assert out["injury"][0] == "Knee"


def schedules(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """(season, week, gameday, gametime, home, away)."""
    return pl.DataFrame(
        rows,
        schema=["season", "week", "gameday", "gametime", "home_team", "away_team"],
        orient="row",
    )


SCHED = schedules([(2025, 1, "2025-09-07", "13:00", "KC", "SD")])


def test_kickoffs_convert_eastern_to_utc() -> None:
    k = kickoffs(SCHED).sort("team")
    assert k["team"].to_list() == ["KC", "LAC"]
    assert k["kickoff"].dtype == pl.Datetime("us", "UTC")
    # Compare naive values: reading zone-aware values back needs tzdata on Windows.
    assert k["kickoff"].dt.replace_time_zone(None)[0] == dt.datetime(2025, 9, 7, 17, 0)


def weekly_chart(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """Pre-2025 rows: (season, week, club, gsis, formation, depth_team, position, depth_pos)."""
    frame = pl.DataFrame(
        rows,
        schema=[
            "season",
            "week",
            "club_code",
            "gsis_id",
            "formation",
            "depth_team",
            "position",
            "depth_position",
        ],
        orient="row",
    )
    return frame.with_columns(full_name=pl.col("gsis_id"))


def snapshot_chart(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """2025+ rows: (dt, team, gsis_id, pos_grp, pos_abb, pos_rank)."""
    frame = pl.DataFrame(
        rows, schema=["dt", "team", "gsis_id", "pos_grp", "pos_abb", "pos_rank"], orient="row"
    )
    return frame.with_columns(player_name=pl.col("gsis_id"), pos_rank=pl.col("pos_rank"))


def test_weekly_depth_chart_normalizes() -> None:
    raw = weekly_chart(
        [
            (2019, 1, "OAK", "00-1", "Offense", "1", "QB", "QB"),
            (2019, 1, "OAK", "00-2", "Offense", "2", "WR", ""),  # blank slot -> position
            (2019, 1, "OAK", "00-3", "Defense", "1", "CB", "LCB"),
            (2019, 1, "OAK", None, "Defense", "2", "CB", "LCB"),
        ]
    )
    out = depth_chart_weeks(raw, SCHED)
    assert out.height == 3
    assert set(out["team"]) == {"LV"}
    row = out.filter(pl.col("gsis_id") == "00-2").row(0, named=True)
    assert (row["side"], row["slot"], row["depth"]) == ("offense", "WR", 2)
    assert out.filter(pl.col("gsis_id") == "00-3")["side"][0] == "defense"


def test_snapshot_depth_chart_uses_last_snapshot_before_kickoff() -> None:
    raw = snapshot_chart(
        [
            ("2025-09-06T07:30:00Z", "KC", "00-old", "3WR 1TE", "QB", 1),
            ("2025-09-07T07:30:00Z", "KC", "00-new", "3WR 1TE", "QB", 1),
            ("2025-09-07T07:30:00Z", "KC", "00-lb", "Base 4-3 D", "MLB", 2),
            ("2025-09-07T07:30:00Z", "KC", "00-lb", "Base 3-4 D", "MLB", 1),
            ("2025-09-07T07:30:00Z", "KC", "00-k", "Special Teams", "PK", 1),
            # Taken after kickoff (13:00 ET = 17:00 UTC): must not be used.
            ("2025-09-07T18:00:00Z", "KC", "00-late", "3WR 1TE", "QB", 1),
        ]
    )
    out = depth_chart_weeks(raw, SCHED).filter(pl.col("team") == "KC")
    assert set(out["gsis_id"]) == {"00-new", "00-lb", "00-k"}
    lb = out.filter(pl.col("gsis_id") == "00-lb")
    assert lb.height == 1 and lb["depth"][0] == 1 and lb["side"][0] == "defense"
    assert out.filter(pl.col("gsis_id") == "00-k")["side"][0] == "special"
    assert (out["season"][0], out["week"][0]) == (2025, 1)


def test_depth_chart_handles_both_formats_stacked() -> None:
    weekly = weekly_chart([(2024, 1, "KC", "00-a", "Offense", "1", "QB", "QB")])
    snap = snapshot_chart([("2025-09-07T07:30:00Z", "KC", "00-b", "3WR 1TE", "QB", 1)])
    stacked = pl.concat([weekly, snap], how="diagonal_relaxed")
    out = depth_chart_weeks(stacked, SCHED)
    assert out.select("season", "gsis_id").rows() == [(2024, "00-a"), (2025, "00-b")]


def test_depth_chart_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="neither"):
        depth_chart_weeks(pl.DataFrame({"x": [1]}), SCHED)


def test_play_probability_table_and_position_adjustment() -> None:
    cfg = InjuriesConfig()
    table = play_probability_table(cfg)
    # 2 groups x 4 statuses x 4 practices + unlisted regulars.
    assert table.height == 33
    frame = pl.DataFrame(
        {
            "regular": [True, True, True, False],
            "status": ["questionable", "questionable", UNLISTED, "out"],
            "practice": ["limited", "limited", "none", "dnp"],
            "position": ["QB", "WR", "QB", "QB"],
        }
    )
    p = attach_play_probability(frame, cfg)["play_prob"].to_list()
    q = cfg.play_probability_regular["questionable"]["limited"]
    assert p == pytest.approx([q - 0.15, q, cfg.unlisted_regular, 0.0])

    big = InjuriesConfig(questionable_position_adjustment={"QB": -5.0})
    assert attach_play_probability(frame, big)["play_prob"][0] == 0.0


def test_injuries_config_validates_tables() -> None:
    table = dict(InjuriesConfig().play_probability_regular)
    del table["doubtful"]
    with pytest.raises(ValidationError, match="missing status"):
        InjuriesConfig(play_probability_regular=table)
    bad = {s: {p: 1.5 for p in PRACTICES} for s in REPORT_STATUSES}
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        InjuriesConfig(play_probability_other=bad)


def test_packaged_config_matches_defaults() -> None:
    assert load_settings().injuries == InjuriesConfig()


def snaps(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """``player_snaps``-shaped rows: (season, week, team, gsis_id, pos, off_pct, def_pct, st)."""
    frame = pl.DataFrame(
        rows,
        schema={
            "season": pl.Int32,
            "week": pl.Int32,
            "team": pl.String,
            "gsis_id": pl.String,
            "snap_position": pl.String,
            "offense_pct": pl.Float64,
            "defense_pct": pl.Float64,
            "st_snaps": pl.Float64,
        },
        orient="row",
    )
    return frame.with_columns(
        id_matched=pl.col("gsis_id").is_not_null(),
        offense_snaps=pl.col("offense_pct") * 60,
        defense_snaps=pl.col("defense_pct") * 60,
    )


def reports(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """``injury_reports``-shaped rows: (season, week, team, gsis_id, position, status, practice)."""
    frame = pl.DataFrame(
        rows,
        schema={
            "season": pl.Int32,
            "week": pl.Int32,
            "team": pl.String,
            "gsis_id": pl.String,
            "position": pl.String,
            "status": pl.String,
            "practice": pl.String,
        },
        orient="row",
    )
    return frame.with_columns(name=pl.col("gsis_id"))


SNAPS = snaps(
    [
        # 2023 week 18: last game of the prior season.
        (2023, 18, "KC", "qb", "QB", 1.0, 0.0, 0.0),
        # 2024 week 1.
        (2024, 1, "KC", "qb", "QB", 1.0, 0.0, 0.0),
        (2024, 1, "KC", "wr", "WR", 0.8, 0.0, 0.0),
        (2024, 1, "KC", "te", "TE", 0.1, 0.0, 5.0),  # not a regular (10% of snaps)
        (2024, 1, "KC", "cb", "CB", 0.0, 0.9, 0.0),
        (2024, 1, "KC", "ir", "LB", 0.0, 0.5, 0.0),
        # 2024 week 3 (week 2 bye): previous game is week 1.
        (2024, 3, "KC", "qb", "QB", 1.0, 0.0, 0.0),
        (2024, 3, "KC", "te", "TE", 0.0, 0.0, 3.0),  # special teams only still counts
        (2024, 3, "BUF", "cb", "CB", 0.0, 0.7, 0.0),  # traded, played for BUF
    ]
)
REPORTS = reports(
    [
        (2024, 1, "KC", "qb", "QB", "questionable", "limited"),
        (2024, 3, "KC", "wr", "WR", "out", "dnp"),
        (2024, 3, "KC", "te", "TE", "questionable", "full"),
    ]
)


def test_report_outcomes_population_and_flags() -> None:
    out = report_outcomes(REPORTS, SNAPS, 0.25)
    rows = {(r["week"], r["gsis_id"]): r for r in out.iter_rows(named=True)}
    # Week 1: QB was a regular in the prior season's last game and played.
    assert rows[(1, "qb")]["regular"] and rows[(1, "qb")]["played"]
    assert rows[(1, "qb")]["status"] == "questionable"
    # Week 3: listed players.
    assert rows[(3, "wr")]["regular"] and not rows[(3, "wr")]["played"]
    assert not rows[(3, "te")]["regular"] and rows[(3, "te")]["played"]
    # Week 3: unlisted regulars from week 1 (qb, cb, ir); te is listed and below the bar.
    unlisted = out.filter((pl.col("week") == 3) & (pl.col("status") == UNLISTED))
    assert sorted(unlisted["gsis_id"]) == ["cb", "ir", "qb"]
    assert rows[(3, "cb")]["played"]  # played for another team
    assert not rows[(3, "ir")]["played"]
    assert rows[(3, "ir")]["position"] == "LB"


def test_report_outcomes_drops_players_ruled_out_by_roster() -> None:
    rosters = pl.DataFrame(
        {"season": [2024], "week": [3], "gsis_id": ["ir"], "status": ["RES"]},
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )
    out = report_outcomes(REPORTS, SNAPS, 0.25, rosters, ("RES",))
    assert "ir" not in out.filter(pl.col("week") == 3)["gsis_id"].to_list()
    kept = report_outcomes(REPORTS, SNAPS, 0.25, rosters, ())
    assert "ir" in kept.filter(pl.col("week") == 3)["gsis_id"].to_list()


def test_coverage_counts_missing_reports_and_orphans() -> None:
    extra = pl.concat([REPORTS, reports([(2024, 2, "KC", "x", "WR", "out", "dnp")])])
    cov = coverage(extra, SNAPS).filter(pl.col("season") == 2024).row(0, named=True)
    # 2024 team-games: KC 1, KC 3, BUF 3; BUF has no report. Week 2 KC was a bye.
    assert cov["team_games"] == 3
    assert cov["no_report"] == 1
    assert cov["orphan_rows"] == 1


def test_rate_table_and_brier_scores() -> None:
    cfg = InjuriesConfig()
    out = report_outcomes(REPORTS, SNAPS, 0.25)
    table = rate_table(out, cfg)
    assert table["players"].sum() == out.height
    wr = table.filter((pl.col("status") == "out") & pl.col("regular")).row(0, named=True)
    assert wr["observed"] == 0.0 and wr["config"] == 0.0
    briers = brier_scores(out, cfg)
    assert briers["group"].to_list() == ["on report", "unlisted regulars", "all"]
    assert briers.filter(pl.col("group") == "all")["players"][0] == out.height
    assert (briers["brier_cell_rate"] <= briers["brier_status_rate"] + 1e-12).all()


def test_depth_coverage() -> None:
    depth = pl.DataFrame(
        {
            "season": [2024, 2024, 2024],
            "week": [1, 1, 1],
            "team": ["KC", "KC", "KC"],
            "gsis_id": ["qb", "wr", "missing"],
            "name": ["qb", "wr", "missing"],
            "side": ["offense", "offense", "defense"],
            "slot": ["QB", "WR", "LCB"],
            "depth": [1, 1, 1],
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32, "depth": pl.Int32},
    )
    cov = depth_coverage(depth, SNAPS).filter(pl.col("season") == 2024).row(0, named=True)
    assert cov["charted"] == pytest.approx(1 / 3)  # KC week 1 only
    assert cov["starters_played"] == pytest.approx(2 / 3)
    assert cov["qb_starter_led"] == 1.0
