from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from nflsim.config import RatingsConfig
from nflsim.model.ratings import KEYS, STAT_COLUMNS, opponent_adjusted, ratings_as_of

# True per-play EPA = offense + defense (league 0). B's defense is bad, C's is good.
OFF = {"A": 0.2, "B": 0.0, "C": 0.0, "D": -0.2}
DEF = {"A": 0.0, "B": 0.2, "C": -0.2, "D": 0.0}
# Lopsided schedule: A and D meet the bad defense B or the good defense C mostly.
SCHEDULE = [(w, ("A", "B"), ("C", "D")) for w in range(1, 6)] + [
    (6, ("A", "C"), ("B", "D")),
    (7, ("A", "D"), ("B", "C")),
]
PLAYS = 1000.0
WEAK: dict[str, Any] = {
    "prior_plays_epa": 1.0,
    "prior_plays_success": 1.0,
    "prior_plays_pass_epa": 1.0,
    "prior_plays_rush_epa": 1.0,
}


def team_games(schedule: list[tuple[int, tuple[str, str], tuple[str, str]]]) -> pl.DataFrame:
    rows = []
    for week, *pairs in schedule:
        for x, y in pairs:
            game_id = f"2020_{week:02d}_{x}_{y}"
            for team, opp in ((x, y), (y, x)):
                rate = OFF[team] + DEF[opp]
                allowed = OFF[opp] + DEF[team]
                stats = {c: 0.0 for c in STAT_COLUMNS}
                for unit, value in (("off", rate), ("def", allowed)):
                    stats |= {
                        f"{unit}_n": PLAYS,
                        f"{unit}_epa": value * PLAYS,
                        f"{unit}_success": 0.5 * PLAYS,
                        f"{unit}_pass_n": PLAYS,
                        f"{unit}_pass_epa": value * PLAYS,
                    }
                stats |= {"plays": 60.0, "games": 1.0}
                rows.append(
                    {"season": 2020, "week": week, "game_id": game_id, "team": team, **stats}
                )
    return (
        pl.DataFrame(rows).cast({"season": pl.Int32, "week": pl.Int32}).select(*KEYS, *STAT_COLUMNS)
    )


def targets(week: int) -> pl.DataFrame:
    return pl.DataFrame(
        {"season": [2020] * 4, "week": [week] * 4, "team": list(OFF)},
        schema={"season": pl.Int32, "week": pl.Int32, "team": pl.String},
    )


def deviations(config: RatingsConfig, week: int = 8) -> dict[str, tuple[float, float]]:
    r = ratings_as_of(team_games(SCHEDULE), targets(week), config)
    return {
        row["team"]: (row["off_epa"] - row["league_epa"], row["def_epa"] - row["league_epa"])
        for row in r.iter_rows(named=True)
    }


def test_adjustment_recovers_offense_and_defense_from_a_lopsided_schedule() -> None:
    raw = deviations(RatingsConfig(min_wp=0.0, max_wp=1.0, **WEAK))
    adj = deviations(RatingsConfig(min_wp=0.0, max_wp=1.0, opponent_adjust=True, **WEAK))

    def error(dev: dict[str, tuple[float, float]]) -> float:
        return sum((dev[t][0] - OFF[t]) ** 2 + (dev[t][1] - DEF[t]) ** 2 for t in OFF)

    assert error(adj) < 0.05 * error(raw)
    # A's offense looks better than it is when it mostly faced B's bad defense.
    assert raw["A"][0] > OFF["A"] + 0.1
    assert adj["A"][0] == pytest.approx(OFF["A"], abs=0.02)
    assert adj["C"][1] == pytest.approx(DEF["C"], abs=0.02)


def test_adjustment_uses_only_earlier_weeks() -> None:
    cfg = RatingsConfig(min_wp=0.0, max_wp=1.0, opponent_adjust=True)
    base = opponent_adjusted(team_games(SCHEDULE), targets(6), cfg).sort("team")
    later = [*SCHEDULE[:5], (6, ("A", "D"), ("B", "C")), (7, ("A", "B"), ("C", "D"))]
    changed = opponent_adjusted(team_games(later), targets(6), cfg).sort("team")
    assert base.equals(changed)


def test_adjustment_shrinks_toward_the_prior_with_few_plays() -> None:
    cfg = RatingsConfig(min_wp=0.0, max_wp=1.0, opponent_adjust=True)
    one_game = opponent_adjusted(team_games(SCHEDULE[:1]), targets(2), cfg)
    a = one_game.filter(pl.col("team") == "A").row(0, named=True)
    # Week 1: A gained 0.4 per play against B (league 0). One game and a 300-play prior
    # split that between A's offense and B's defense, both well under the full 0.4.
    b = one_game.filter(pl.col("team") == "B").row(0, named=True)
    assert 0 < a["off_epa"] < 0.4
    assert 0 < b["def_epa"] < 0.4
    assert a["off_epa"] + b["def_epa"] < 0.4


def test_no_data_gives_null_and_default_is_unadjusted() -> None:
    empty = team_games(SCHEDULE).head(0)
    adj = opponent_adjusted(empty, targets(1), RatingsConfig(opponent_adjust=True))
    assert adj["off_epa"].null_count() == 4
    off = RatingsConfig(min_wp=0.0, max_wp=1.0)
    assert not off.opponent_adjust
    on = RatingsConfig(min_wp=0.0, max_wp=1.0, opponent_adjust=True)
    a = ratings_as_of(team_games(SCHEDULE), targets(8), off)
    b = ratings_as_of(team_games(SCHEDULE), targets(8), on)
    assert a["pace"].equals(b["pace"])  # pace is not adjusted
    assert not a["off_epa"].equals(b["off_epa"])
