from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from nflsim.config import PlayerRatingsConfig, RatingsConfig
from nflsim.model.players import (
    attach_position,
    player_game_stats,
    player_ratings_as_of,
    projected_usage,
    role_plays,
    usage_history,
)
from nflsim.player_eval import evaluate_ratings, evaluate_usage

RCFG = RatingsConfig()

_PBP_DEFAULTS: dict[str, Any] = {
    "season": 2020,
    "week": 1,
    "game_id": "g",
    "posteam": "KC",
    "play_type": "pass",
    "epa": 0.0,
    "two_point_attempt": 0.0,
    "wp": 0.5,
    "qb_dropback": 0.0,
    "qb_scramble": 0.0,
    "sack": 0.0,
    "passer_id": None,
    "passer_player_id": None,
    "rusher_player_id": None,
    "rusher_id": None,
    "receiver_player_id": None,
}


def pbp(rows: list[dict[str, Any]]) -> pl.DataFrame:
    full = [{**_PBP_DEFAULTS, **r} for r in rows]
    return pl.DataFrame(
        full,
        schema={
            k: (pl.String if isinstance(v, str) or v is None else pl.Float64)
            for k, v in _PBP_DEFAULTS.items()
        }
        | {"season": pl.Int32, "week": pl.Int32},
    )


def completion(qb: str, wr: str | None, epa: float, **kw: Any) -> dict[str, Any]:
    return {
        "qb_dropback": 1.0,
        "passer_id": qb,
        "passer_player_id": qb,
        "receiver_player_id": wr,
        "epa": epa,
        **kw,
    }


def run(rb: str, epa: float, **kw: Any) -> dict[str, Any]:
    return {"play_type": "run", "rusher_player_id": rb, "rusher_id": rb, "epa": epa, **kw}


def rosters(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """``roster_weeks`` rows: (season, week, gsis_id, team, position, status)."""
    return pl.DataFrame(
        rows,
        schema={
            "season": pl.Int32,
            "week": pl.Int32,
            "gsis_id": pl.String,
            "team": pl.String,
            "position": pl.String,
            "status": pl.String,
        },
        orient="row",
    )


def test_role_plays_assigns_roles_and_filters_plays() -> None:
    plays = pbp(
        [
            completion("QB1", "WR1", 0.5),  # dropback + target
            {"qb_dropback": 1.0, "sack": 1.0, "passer_id": "QB1", "epa": -1.0},  # sack
            # Scramble: play_type "run", passer_player_id empty, passer_id names the QB.
            {
                "play_type": "run",
                "qb_dropback": 1.0,
                "qb_scramble": 1.0,
                "passer_id": "QB1",
                "rusher_player_id": "QB1",
                "epa": 0.4,
            },
            run("RB1", 0.1),  # designed run
            run("RB1", 0.2, two_point_attempt=1.0),  # two-point try: dropped
            run("RB1", 0.3, wp=0.99),  # garbage time: dropped
            run("RB1", 0.3, play_type="qb_kneel"),  # not a pass/run: dropped
        ]
    )
    out = role_plays(plays, RCFG).collect()
    got = sorted(zip(out["role"], out["gsis_id"], out["epa"], strict=True))
    assert got == [
        ("carry", "RB1", 0.1),
        ("dropback", "QB1", -1.0),
        ("dropback", "QB1", 0.4),
        ("dropback", "QB1", 0.5),
        ("target", "WR1", 0.5),
    ]


def test_attach_position_uses_latest_then_earliest_roster_week() -> None:
    roster = rosters(
        [
            (2020, 3, "A", "KC", "QB", "ACT"),
            (2021, 1, "A", "KC", "WR", "ACT"),
            (2020, 5, "B", "KC", "TE", "ACT"),
        ]
    )
    frame = pl.DataFrame(
        {
            "season": [2020, 2020, 2021, 2020, 2020],
            "week": [4, 3, 2, 1, 1],
            "gsis_id": ["A", "A", "A", "B", "C"],
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )
    out = attach_position(frame, roster).sort("gsis_id", "season", "week")
    assert out["position"].to_list() == ["QB", "QB", "WR", "TE", None]


def test_player_game_stats_groups_and_team_totals() -> None:
    plays = pbp(
        [
            completion("QB1", "WR1", 0.5),
            completion("QB1", None, -0.2),  # throwaway: dropback, no target row
            run("QB1", 0.6),  # QB designed run -> QB carry group
            run("OL1", 1.0),  # tackle-eligible run -> "other"
        ]
    )
    roster = rosters(
        [
            (2020, 1, "QB1", "KC", "QB", "ACT"),
            (2020, 1, "WR1", "KC", "WR", "ACT"),
            (2020, 1, "OL1", "KC", "OL", "ACT"),
        ]
    )
    players, teams = player_game_stats(plays, roster, RCFG)
    rows = {(r["gsis_id"], r["role"]): r for r in players.iter_rows(named=True)}
    assert rows[("QB1", "dropback")]["n"] == 2
    assert rows[("QB1", "dropback")]["epa"] == pytest.approx(0.3)
    assert rows[("QB1", "carry")]["group"] == "QB"
    assert rows[("OL1", "carry")]["group"] == "other"
    assert rows[("WR1", "target")]["group"] == "WR"
    assert dict(zip(teams["role"], teams["team_n"], strict=True)) == {
        "carry": 2.0,
        "dropback": 2.0,
        "target": 1.0,
    }


def player_games(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """Rows: (season, week, gsis_id, role, group, n, epa)."""
    frame = pl.DataFrame(
        rows,
        schema={
            "season": pl.Int32,
            "week": pl.Int32,
            "gsis_id": pl.String,
            "role": pl.String,
            "group": pl.String,
            "n": pl.Float64,
            "epa": pl.Float64,
        },
        orient="row",
    )
    return frame.with_columns(
        game_id=pl.format("{}_{}_{}", "season", "week", "gsis_id"), team=pl.lit("KC")
    )


CFG = PlayerRatingsConfig(
    prior_plays={"dropback": 100.0, "carry": 100.0, "target": 100.0},
    carryover=0.5,
    unknown_deviation={"dropback": -0.1, "carry": 0.0, "target": 0.0},
)


def target(season: int, week: int, gsis_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [season],
            "week": [week],
            "gsis_id": [gsis_id],
            "role": ["dropback"],
            "group": ["QB"],
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )


def test_rating_formula_with_prior_season_and_current_data() -> None:
    games = player_games(
        [
            # 2019: group QB average = (60 + 0) / (300 + 300) = 0.1; A is +0.1 above it.
            (2019, 1, "A", "dropback", "QB", 300, 60.0),
            (2019, 1, "B", "dropback", "QB", 300, 0.0),
            # 2020 week 1: A posts 50 dropbacks at +0.3.
            (2020, 1, "A", "dropback", "QB", 50, 15.0),
            # Week 2 must not leak into the week-2 rating.
            (2020, 2, "A", "dropback", "QB", 50, -50.0),
        ]
    )
    row = player_ratings_as_of(games, target(2020, 2, "A"), CFG).row(0, named=True)
    group_mean = (60 + 0 + 15) / (600 + 50)  # last season pooled with week 1
    excess = 60 - 300 * 0.1  # +30 EPA above 2019's group average
    deviation = 0.5 * excess / (300 + 100) + (-0.1) * 100 / (300 + 100)
    expected = (15 + 100 * (group_mean + deviation)) / (50 + 100)
    assert row["group_mean"] == pytest.approx(group_mean)
    assert row["rating"] == pytest.approx(expected)
    assert row["cur_n"] == 50
    assert row["prior_source"] == "prior_season"


def test_unknown_player_starts_at_group_mean_plus_unknown_deviation() -> None:
    games = player_games([(2020, 1, "A", "dropback", "QB", 100, 10.0)])
    row = player_ratings_as_of(games, target(2020, 2, "ROOKIE"), CFG).row(0, named=True)
    assert row["prior_source"] == "unknown"
    assert row["rating"] == pytest.approx(0.1 - 0.1)


def test_rating_is_null_without_any_group_data() -> None:
    games = player_games([(2020, 1, "A", "dropback", "QB", 100, 10.0)])
    row = player_ratings_as_of(games, target(2020, 1, "A"), CFG).row(0, named=True)
    assert row["rating"] is None  # week 1 of the first season: nothing before it


def usage_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """KC: QB1 starts weeks 1-2, QB2 takes over in week 3. WR1 catches everything."""
    rows = []
    for week, qb in ((1, "QB1"), (2, "QB1"), (3, "QB2")):
        rows += [completion(qb, "WR1", 0.1, week=week, game_id=f"g{week}")] * 10
    roster_rows = []
    for week in (1, 2, 3, 4):
        roster_rows += [
            (2020, week, "QB1", "KC", "QB", "ACT"),
            (2020, week, "QB2", "KC", "QB", "ACT"),
            (2020, week, "WR1", "KC", "WR", "ACT"),
        ]
    roster_rows.append((2020, 4, "ROOK", "KC", "QB", "ACT"))  # signed before week 4
    roster = rosters(roster_rows)
    players, teams = player_game_stats(pbp(rows), roster, RCFG)
    return players, teams, roster


def test_projected_usage_follows_recent_games_and_renormalizes() -> None:
    players, teams, roster = usage_inputs()
    cfg = PlayerRatingsConfig(usage_half_life_games={"dropback": 1.0, "carry": 1.0, "target": 1.0})
    history = usage_history(players, teams, roster, cfg)
    weeks = pl.DataFrame(
        {"season": [2020, 2020], "week": [3, 4], "team": ["KC", "KC"]},
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )
    proj = projected_usage(history, roster, weeks, cfg).filter(pl.col("role") == "dropback")
    share = {(r["week"], r["gsis_id"]): r for r in proj.iter_rows(named=True)}
    assert share[(3, "QB1")]["share"] == pytest.approx(1.0)  # before the switch
    assert share[(4, "QB2")]["share"] > 0.5  # after one game, the new starter leads
    assert share[(4, "QB1")]["share"] + share[(4, "QB2")]["share"] == pytest.approx(1.0)
    assert share[(4, "ROOK")]["share"] == 0.0
    assert not share[(4, "ROOK")]["has_history"]
    # WR1 has no dropbacks and is not a QB: never a dropback candidate.
    assert (4, "WR1") not in share


def test_projected_usage_excludes_unavailable_and_non_qb_passers() -> None:
    rows = [completion("QB1", "WR1", 0.1)] * 9
    rows.append(completion("WR1", "QB1", 0.5))  # trick play: WR throws to the QB
    roster = rosters(
        [
            (2020, 1, "QB1", "KC", "QB", "ACT"),
            (2020, 1, "WR1", "KC", "WR", "ACT"),
            (2020, 2, "QB1", "KC", "QB", "RES"),  # on injured reserve for week 2
            (2020, 2, "WR1", "KC", "WR", "ACT"),
            (2020, 2, "QB2", "KC", "QB", "ACT"),
        ]
    )
    players, teams = player_game_stats(pbp(rows), roster, RCFG)
    history = usage_history(players, teams, roster, PlayerRatingsConfig())
    weeks = pl.DataFrame(
        {"season": [2020], "week": [2], "team": ["KC"]},
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )
    proj = projected_usage(history, roster, weeks, PlayerRatingsConfig())
    dropback = proj.filter(pl.col("role") == "dropback")
    assert dropback["gsis_id"].to_list() == ["QB2"]  # QB1 out, WR1 not eligible
    assert dropback["share"].to_list() == [None]  # nobody eligible has history: flagged
    target_rows = proj.filter(pl.col("role") == "target")
    # WR1 had 0.9 of week-1 targets; without the unavailable QB1 it renormalizes to 1.
    assert dict(zip(target_rows["gsis_id"], target_rows["share"], strict=True)) == {
        "QB2": 0.0,
        "WR1": pytest.approx(1.0),
    }


def test_evaluate_ratings_rewards_persistent_skill() -> None:
    rows = []
    for season in (2019, 2020):
        for week in range(1, 9):
            rows.append((season, week, "GOOD", "dropback", "QB", 30, 30 * 0.3))
            rows.append((season, week, "BAD", "dropback", "QB", 30, 30 * -0.3))
    report = evaluate_ratings(player_games(rows), [2020], CFG).row(0, named=True)
    assert report["role"] == "dropback"
    assert report["skill_rating"] > 0.5
    assert report["mse_rating"] < report["mse_position_mean"]


def test_evaluate_usage_is_zero_when_projection_is_exact() -> None:
    players, teams, roster = usage_inputs()
    cfg = PlayerRatingsConfig()
    history = usage_history(players, teams, roster, cfg)
    report = evaluate_usage(history, teams, roster, [2020], cfg)
    target_row = report.filter(pl.col("role") == "target").row(0, named=True)
    assert target_row["tv_distance"] == pytest.approx(0.0, abs=1e-9)  # WR1 every week
    dropback_row = report.filter(pl.col("role") == "dropback").row(0, named=True)
    assert dropback_row["no_projection"] == pytest.approx(1 / 3)  # week 1: no history
    assert dropback_row["tv_distance"] > 0.0  # the week-3 switch was not foreseeable


def test_config_requires_every_role() -> None:
    with pytest.raises(ValidationError, match="missing roles"):
        PlayerRatingsConfig(prior_plays={"dropback": 100.0, "carry": 100.0})
    with pytest.raises(ValidationError, match="must be positive"):
        PlayerRatingsConfig(usage_half_life_games={"dropback": 0.0, "carry": 1.0, "target": 1.0})
