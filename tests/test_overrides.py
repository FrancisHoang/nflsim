from __future__ import annotations

import polars as pl
import pytest

from nflsim.config import InjuriesConfig
from nflsim.model.availability import expected_role_values
from nflsim.overrides import (
    Override,
    apply_overrides,
    normalize_name,
    override_probability,
    resolve,
    week_roster,
)
from tests.test_availability import LINEUP

ROSTERS = pl.DataFrame(
    {
        "season": [2024] * 7,
        "week": [5, 5, 5, 5, 5, 4, 3],
        "gsis_id": ["00-0000001", "00-0000002", "00-0000003", "00-0000004", "00-0000005",
                    "00-0000006", "00-0000007"],
        "name": ["Patrick Mahomes II", "José Núñez", "D.J. Moore", "Chris Jones",
                 "Chris Jones", "Josh Allen", "Old Week"],
        "team": ["KC", "KC", "KC", "KC", "KC", "BUF", "BUF"],
        "position": ["QB", "WR", "WR", "DL", "DB", "QB", "WR"],
    },
    schema_overrides={"season": pl.Int32, "week": pl.Int32},
)  # fmt: skip


def test_normalize_name_drops_case_accents_punctuation_and_suffixes() -> None:
    assert normalize_name("Patrick Mahomes II") == "patrick mahomes"
    assert normalize_name("José Núñez") == "jose nunez"
    assert normalize_name("D.J. Moore") == "dj moore"
    assert normalize_name("Omar Cooper Jr.") == "omar cooper"


def test_week_roster_uses_latest_week_up_to_the_game() -> None:
    roster = week_roster(ROSTERS, 2024, 5, ("KC", "BUF"))
    # BUF has no week-5 roster: its week-4 roster is used, not week 3's.
    assert set(roster.filter(pl.col("team") == "BUF")["name"]) == {"Josh Allen"}
    with pytest.raises(ValueError, match="no 2024 roster"):
        week_roster(ROSTERS, 2024, 2, ("KC", "BUF"))


def test_resolve_by_id_full_name_and_last_name() -> None:
    roster = week_roster(ROSTERS, 2024, 5, ("KC", "BUF"))
    found = resolve(
        [
            Override("out", "patrick mahomes"),
            Override("plays", "Jose Nunez"),
            Override("doubtful", "DJ Moore"),
            Override("out", "00-0000006"),
        ],
        roster,
    )
    assert [r.gsis_id for r in found] == ["00-0000001", "00-0000002", "00-0000003", "00-0000006"]
    assert found[3].team == "BUF"
    assert resolve([Override("out", "Mahomes")], roster)[0].name == "Patrick Mahomes II"


def test_resolve_rejects_ambiguous_unknown_and_repeated_players() -> None:
    roster = week_roster(ROSTERS, 2024, 5, ("KC", "BUF"))
    with pytest.raises(ValueError, match=r"several players.*00-0000004.*00-0000005"):
        resolve([Override("out", "Chris Jones")], roster)
    with pytest.raises(ValueError, match="no player matching"):
        resolve([Override("out", "Old Week")], roster)  # not on the roster used
    with pytest.raises(ValueError, match="more than once"):
        resolve([Override("out", "Mahomes"), Override("plays", "patrick mahomes")], roster)


def test_override_probability() -> None:
    cfg = InjuriesConfig()
    assert override_probability("out", cfg) == 0.0
    assert override_probability("plays", cfg) == 1.0
    assert override_probability("doubtful", cfg) == cfg.play_probability_regular["doubtful"]["none"]


def test_apply_overrides_changes_lineup_and_game_features() -> None:
    from nflsim.overrides import ResolvedOverride

    games = pl.DataFrame(
        {
            "game_id": ["SIM", "OTHER"],
            "season": [2024, 2024],
            "week": [5, 5],
            "home_team": ["KC", "LV"],
            "away_team": ["BUF", "DEN"],
            "h_off_qb": [0.5, 0.5],
            "a_off_qb": [0.5, 0.5],
            "h_off_recv": [0.5, 0.5],
            "a_off_recv": [0.5, 0.5],
        }
    )
    lineup = LINEUP.with_columns(practice=pl.lit("none"))
    resolved = [
        ResolvedOverride("plays", "qb1", "qb1", "Qb One", "KC", "QB"),
        ResolvedOverride("out", "dl1", "dl1", "Big Tackle", "KC", "DL"),
    ]
    new_games, new_lineup, effects = apply_overrides(
        games, "SIM", lineup, resolved, InjuriesConfig()
    )
    assert new_lineup.filter(pl.col("gsis_id") == "qb1")["play_prob"][0] == 1.0
    sim = new_games.filter(pl.col("game_id") == "SIM").row(0, named=True)
    assert sim["h_off_qb"] == pytest.approx(0.20)  # qb1 now always plays (dev 0.20)
    values = {
        (v["team"], v["role"]): v["value"]
        for v in expected_role_values(new_lineup).iter_rows(named=True)
    }
    assert sim["a_off_qb"] == pytest.approx(values[("BUF", "dropback")])
    assert sim["h_off_recv"] == pytest.approx(values[("KC", "target")])
    other = new_games.filter(pl.col("game_id") == "OTHER").row(0, named=True)
    assert other["h_off_qb"] == 0.5  # other games untouched
    rows = {r["player"]: r for r in effects.iter_rows(named=True)}
    assert rows["Qb One"]["before"] == pytest.approx(0.6)
    assert rows["Qb One"]["after"] == 1.0
    assert rows["Big Tackle"]["before"] is None  # no projected role: no effect
    with pytest.raises(ValueError, match="exactly one row"):
        apply_overrides(games, "NOPE", lineup, resolved, InjuriesConfig())
