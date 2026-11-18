from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from nflsim.dashboard.tables import (
    final_text,
    kickoff_text,
    lean,
    lines_chart_data,
    pick_mark,
    spread_text,
    week_table,
)
from nflsim.summary import WEEK_SCHEMA
from nflsim.weekly import WeekRun


def week_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=WEEK_SCHEMA)


GAMES = [
    {
        "game_id": "g1",
        "kickoff": "2025-11-23 13:00",
        "away": "IND",
        "home": "KC",
        "home_win": 0.594,
        "model_spread": -3.4,
        "vegas_spread": -3.5,
        "home_cover": 0.492,
        "model_total": 52.0,
        "vegas_total": 50.5,
        "over": 0.503,
        "home_score": 23,
        "away_score": 20,
    },
    {
        "game_id": "g2",
        "kickoff": "2025-11-23 16:25",
        "away": "NE",
        "home": "CIN",
        "home_win": 0.473,
        "model_spread": 0.9,
        "vegas_spread": 7.5,
        "home_cover": 0.691,
        "model_total": 49.8,
        "vegas_total": 50.5,
        "over": 0.44,
        "home_score": 20,
        "away_score": 26,
    },
]


def test_spread_and_lean_text() -> None:
    assert spread_text("KC", -3.44) == "KC -3.4"
    assert spread_text("KC", 0.04) == "KC pk"
    assert spread_text("KC", None) is None
    # Model makes CIN a 0.9-point underdog, Vegas 7.5: the model likes CIN by 6.6.
    team, by = lean("CIN", "NE", 0.9, 7.5)
    assert team == "CIN" and by == pytest.approx(6.6)
    team, by = lean("GB", "MIN", -6.5, -12.4)
    assert team == "MIN" and by == pytest.approx(5.9)
    assert lean("KC", "IND", -3.5, -3.5) == ("none", 0.0)
    assert lean("KC", "IND", None, -3.5) == (None, None)


def test_kickoff_final_and_pick() -> None:
    assert kickoff_text("2025-11-23 13:00") == "Sun 11/23 13:00"
    assert kickoff_text("2025-11-23") == "2025-11-23"
    assert kickoff_text(None) is None
    row = dict(GAMES[1])
    assert final_text(row) == "NE 26-20"
    assert pick_mark(row) == "right"  # model had CIN under 50%
    assert final_text(row | {"home_score": 20, "away_score": 20}) == "tie 20-20"
    assert pick_mark(row | {"home_score": None}) is None


def test_week_table_keeps_order_and_shows_chances_in_percent() -> None:
    table = week_table(week_frame(GAMES))
    assert table["Game"].to_list() == ["IND @ KC", "NE @ CIN"]
    assert table["Home win"].to_list() == pytest.approx([59.4, 47.3])
    assert table["Model leans"].to_list() == ["IND", "CIN"]
    assert table["Final"].to_list() == ["KC 23-20", "NE 26-20"]
    long = lines_chart_data(week_frame(GAMES))
    assert long.height == 4 and set(long["source"]) == {"nflsim", "Vegas"}


Calls = list[tuple[int, int, int, int, bool]]


@pytest.fixture
def calls() -> Calls:
    return []


@pytest.fixture
def app_test(monkeypatch: pytest.MonkeyPatch, calls: Calls) -> AppTest:
    import nflsim.dashboard.app as app
    from nflsim.dashboard import data

    run = WeekRun(
        season=2025,
        week=12,
        n=1000,
        seed=42,
        frame=week_frame(GAMES),
        predictions=[],
        flags={"wind unknown for this outdoor game": ["IND@KC"]},
        ratings_through=11,
        completed_before=164,
        market=True,
    )

    def week_run(season: int, week: int, n: int, seed: int, injuries: bool) -> WeekRun:
        calls.append((season, week, n, seed, injuries))
        return run

    monkeypatch.setattr(data, "seasons", lambda: [2025])
    monkeypatch.setattr(data, "weeks", lambda season: ([11, 12, 13], 12))
    monkeypatch.setattr(data, "games", lambda season, injuries: (pl.DataFrame(), None))
    monkeypatch.setattr(data, "week_run", week_run)
    return AppTest.from_file(str(Path(app.__file__)), default_timeout=60)


def test_week_page_renders_the_week(app_test: AppTest, calls: Calls) -> None:
    at = app_test.run()
    assert not at.exception
    assert at.title[0].value == "2025 week 12"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Winners picked"] == "2 of 2"
    assert metrics["Biggest gap to Vegas"] == "6.6 pts"
    assert list(at.dataframe[0].value["Game"]) == ["IND @ KC", "NE @ CIN"]
    assert calls == [(2025, 12, 10_000, 42, False)]


def test_week_page_without_data_explains_how_to_fetch(
    app_test: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nflsim.dashboard import data

    monkeypatch.setattr(data, "seasons", lambda: [])
    at = app_test.run()
    assert not at.exception
    assert "nflsim fetch" in at.error[0].value
