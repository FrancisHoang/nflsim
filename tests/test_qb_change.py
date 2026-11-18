from __future__ import annotations

import polars as pl
import pytest

from nflsim.config import PlayerRatingsConfig
from nflsim.model.qb_change import listed_starters, qb_baseline, qb_change_features

CFG = PlayerRatingsConfig()
GOOD, BAD, OTHER = "00-good", "00-bad", "00-other"


def player_games(rows: list[tuple[int, int, str, str, float, float]]) -> pl.DataFrame:
    """Rows: (season, week, team, gsis_id, dropbacks, epa per dropback)."""
    return pl.DataFrame(
        [
            {
                "season": s,
                "week": w,
                "game_id": f"{s}_{w:02d}_{t}",
                "team": t,
                "gsis_id": q,
                "role": "dropback",
                "n": n,
                "epa": n * e,
                "position": "QB",
                "group": "QB",
            }
            for s, w, t, q, n, e in rows
        ]
    ).cast({"season": pl.Int32, "week": pl.Int32})


# Team A: the good QB plays weeks 1-4 of 2021; team B: an average QB every week.
GAMES = player_games(
    [(2021, w, "A", GOOD, 40.0, 0.3) for w in range(1, 5)]
    + [(2021, w, "B", OTHER, 40.0, 0.0) for w in range(1, 5)]
    + [(2020, w, "A", BAD, 40.0, -0.3) for w in range(1, 3)]  # last season's QB
    + [(2020, w, "B", OTHER, 40.0, 0.0) for w in range(1, 3)]
)


def targets(week: int, teams: tuple[str, ...] = ("A", "B")) -> pl.DataFrame:
    return pl.DataFrame(
        {"season": [2021] * len(teams), "week": [week] * len(teams), "team": list(teams)},
        schema={"season": pl.Int32, "week": pl.Int32, "team": pl.String},
    )


def schedule(week: int, home_qb: str | None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2021],
            "week": [week],
            "home_team": ["A"],
            "away_team": ["B"],
            "home_qb_id": [home_qb],
            "away_qb_id": [OTHER],
        },
        schema_overrides={"home_qb_id": pl.String},
    )


def test_baseline_uses_this_season_before_the_week_then_last_season() -> None:
    week5 = qb_baseline(GAMES, targets(5), CFG).sort("team")
    a = week5.row(0, named=True)
    assert a["qb_base_source"] == "season" and a["qb_base"] > 0  # the good QB
    week1 = qb_baseline(GAMES, targets(1), CFG).sort("team").row(0, named=True)
    assert week1["qb_base_source"] == "prior_season" and week1["qb_base"] < 0  # the bad QB
    nobody = qb_baseline(GAMES, targets(5, ("Z",)), CFG).row(0, named=True)
    assert (nobody["qb_base"], nobody["qb_base_source"]) == (0.0, "none")


def test_baseline_ignores_the_target_week_and_later() -> None:
    later = pl.concat([GAMES, player_games([(2021, w, "A", BAD, 40.0, -0.3) for w in (5, 6)])])
    assert (
        qb_baseline(GAMES, targets(5), CFG)
        .sort("team")
        .equals(qb_baseline(later, targets(5), CFG).sort("team"))
    )


def test_listed_starter_change_is_negative_and_same_starter_is_zero() -> None:
    off_qb = targets(5).with_columns(off_qb=pl.lit(0.0))
    changed = qb_change_features(GAMES, schedule(5, BAD), targets(5), off_qb, CFG)
    same = qb_change_features(GAMES, schedule(5, GOOD), targets(5), off_qb, CFG)
    a_changed = changed.filter(pl.col("team") == "A").row(0, named=True)
    a_same = same.filter(pl.col("team") == "A").row(0, named=True)
    assert a_changed["qb_starter_listed"]
    assert a_changed["qb_change"] < -0.1
    assert a_same["qb_change"] == pytest.approx(0.0, abs=1e-12)
    # B's listed starter is the QB its baseline is made of.
    b = changed.filter(pl.col("team") == "B").row(0, named=True)
    assert b["qb_change"] == pytest.approx(0.0, abs=1e-12)


def test_unlisted_starter_falls_back_to_the_projection() -> None:
    off_qb = targets(5).with_columns(off_qb=pl.Series([0.05, 0.0]))
    out = qb_change_features(GAMES, schedule(5, None), targets(5), off_qb, CFG)
    a = out.filter(pl.col("team") == "A").row(0, named=True)
    base = qb_baseline(GAMES, targets(5), CFG).filter(pl.col("team") == "A")["qb_base"][0]
    assert not a["qb_starter_listed"]
    assert a["qb_change"] == a["qb_change_proj"] == pytest.approx(0.05 - base)


def test_listed_starters_normalizes_teams_and_skips_missing_ids() -> None:
    sched = pl.DataFrame(
        {
            "season": [2019, 2019],
            "week": [1, 2],
            "home_team": ["OAK", "KC"],
            "away_team": ["KC", "OAK"],
            "home_qb_id": ["00-carr", None],
            "away_qb_id": ["00-mahomes", "00-carr"],
        }
    )
    out = listed_starters(sched).sort("week", "team")
    assert out.select("week", "team", "gsis_id").rows() == [
        (1, "KC", "00-mahomes"),
        (1, "LV", "00-carr"),
        (2, "LV", "00-carr"),
    ]
    assert listed_starters(sched.drop("home_qb_id", "away_qb_id")).is_empty()
