from __future__ import annotations

import polars as pl
import pytest

from nflsim.live import MATCHUP_ID, find_matchup, ratings_cutoff, report_summary

SCHEDULES = pl.DataFrame(
    {
        "game_id": ["2019_03_KC_OAK", "2019_03_BUF_NE", "2019_04_LA_TB", "2019_02_KC_OAK"],
        "season": [2019, 2019, 2019, 2019],
        "week": [3, 3, 4, 2],
        "game_type": ["REG", "REG", "REG", "REG"],
        "home_team": ["OAK", "NE", "TB", "OAK"],
        "away_team": ["KC", "BUF", "LA", "KC"],
        "home_score": [10, None, None, 20],
        "away_score": [28, None, None, 17],
        "location": ["Home", "Neutral", "Home", "Home"],
        "roof": ["outdoors", "outdoors", "outdoors", "outdoors"],
        "home_rest": [7, 7, 6, 7],
        "away_rest": [10, 7, 7, 7],
        "temp": [70, None, None, 75],
        "wind": [5, None, None, 3],
        "spread_line": [-7.0, None, 3.5, -3.0],
        "total_line": [52.5, None, 47.0, 51.0],
    },
    schema_overrides={"season": pl.Int32, "week": pl.Int32},
)


def test_find_matchup_uses_the_scheduled_game() -> None:
    m = find_matchup(SCHEDULES, 2019, 3, "lv", "kc", neutral=False)
    row = m.row.row(0, named=True)
    assert m.scheduled_id == "2019_03_KC_OAK"
    assert row["game_id"] == MATCHUP_ID
    assert (row["home_team"], row["away_team"]) == ("LV", "KC")
    assert (row["home_rest"], row["away_rest"], row["wind"]) == (7, 10, 5)
    assert row["home_score"] is None  # the simulated row never carries the result
    assert m.final == (10, 28)
    assert (m.spread_line, m.total_line) == (-7.0, 52.5)


def test_find_matchup_keeps_neutral_sites_and_checks_orientation() -> None:
    m = find_matchup(SCHEDULES, 2019, 3, "NE", "BUF", neutral=False)
    assert m.row["location"][0] == "Neutral"
    assert m.final is None
    assert (m.spread_line, m.total_line) == (None, None)  # no line published
    with pytest.raises(ValueError, match="use --home LV --away KC"):
        find_matchup(SCHEDULES, 2019, 3, "KC", "LV", neutral=False)
    with pytest.raises(ValueError, match="not at a neutral site"):
        find_matchup(SCHEDULES, 2019, 3, "LV", "KC", neutral=True)


def test_find_matchup_falls_back_to_a_hypothetical_game() -> None:
    m = find_matchup(SCHEDULES, 2019, 5, "KC", "BUF", neutral=True)
    assert m.scheduled_id is None and m.final is None
    assert m.spread_line is None and m.total_line is None
    assert m.row["location"][0] == "Neutral"


def test_ratings_cutoff_counts_completed_games_before_the_week() -> None:
    assert ratings_cutoff(SCHEDULES, 2019, 4) == (3, 2)
    assert ratings_cutoff(SCHEDULES, 2019, 3) == (2, 1)
    assert ratings_cutoff(SCHEDULES, 2019, 1) == (None, 0)


def test_report_summary_counts_and_lists_all_positions() -> None:
    reports = pl.DataFrame(
        {
            "season": [2019] * 5,
            "week": [3, 3, 3, 3, 2],
            "team": ["KC", "KC", "KC", "LV", "KC"],
            "name": ["Edge Rusher", "Wide Out", "Limited Guy", "Corner", "Old Week"],
            "position": ["DE", "WR", "LB", "CB", "QB"],
            "status": ["questionable", "out", "none", "doubtful", "out"],
            "practice": ["limited", "dnp", "limited", "dnp", "dnp"],
            "injury": ["Ankle", "Knee", "Hip", "Calf", "Ribs"],
        },
        schema_overrides={"season": pl.Int32, "week": pl.Int32},
    )
    counts, listed = report_summary(reports, 2019, 3, ("KC", "LV"))
    assert counts == {"KC": 3, "LV": 1}
    # Out before questionable; players with no game status are counted, not listed.
    assert listed["name"].to_list() == ["Wide Out", "Edge Rusher", "Corner"]
    assert listed.columns == ["team", "name", "position", "status", "practice", "injury"]
    empty_counts, empty = report_summary(reports, 2019, 9, ("KC", "LV"))
    assert empty_counts == {"KC": 0, "LV": 0} and empty.is_empty()
