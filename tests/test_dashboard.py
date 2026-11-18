from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from nflsim.dashboard.game_tables import (
    compare_rows,
    line_rows,
    margin_distribution,
    roster_choices,
    score_ranges,
)
from nflsim.dashboard.tables import (
    final_text,
    kickoff_text,
    lean,
    lines_chart_data,
    pick_mark,
    spread_text,
    week_table,
)
from nflsim.game import GameRequest, GameRun, run_game
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


# Game page ------------------------------------------------------------------------------


@pytest.fixture
def game_run(tmp_path: Path) -> GameRun:
    from nflsim.config import DataConfig
    from nflsim.data import DataCache
    from tests.test_game import SCHEDULE
    from tests.test_pipeline import SETTINGS, rated_games, with_matchup

    c = DataCache(DataConfig(cache_dir=tmp_path), offline=True)
    path = c.path("schedules", 2021)
    path.parent.mkdir(parents=True)
    SCHEDULE.write_parquet(path)
    games = with_matchup(rated_games())
    return run_game(
        c, SETTINGS, GameRequest(2021, 3, "KC", "BUF", n=2000, seed=7), loaded=(games, None)
    )


def test_margin_distribution_shares_and_winners(game_run: GameRun) -> None:
    r = game_run.result
    dist = margin_distribution(r)
    assert dist["share"].sum() == pytest.approx(1.0)
    home_share = dist.filter(pl.col("winner") == "KC")["share"].sum()
    assert home_share == pytest.approx(r.home_win_prob)
    assert (dist.filter(pl.col("margin") > 0)["winner"] == "KC").all()
    assert (dist.filter(pl.col("margin") < 0)["winner"] == "BUF").all()


def test_score_ranges_and_line_rows(game_run: GameRun) -> None:
    r = game_run.result
    ranges = score_ranges(r)
    assert ranges[""].to_list() == ["KC points", "BUF points", "Margin (KC minus BUF)"]
    assert ranges["50%"][2] == r.margin_quantiles[0.5]
    lines = line_rows(game_run.lines, r)
    assert lines["Line"].to_list() == ["Spread KC -2.5", "Total 54.5"]
    spread = lines.row(0, named=True)
    assert spread["Chance"] + spread["Other chance"] + spread["Push"] == pytest.approx(100.0)


def test_compare_rows_and_roster_choices(game_run: GameRun) -> None:
    r = game_run.result
    rows = compare_rows(r, r)
    assert rows["Change"].to_list() == pytest.approx([0.0] * 5)
    roster = pl.DataFrame(
        {
            "gsis_id": ["1", "2", "3", "4"],
            "name": ["Wide Out", "Big Lineman", "Quarter Back", "Run Ner"],
            "team": ["BUF", "BUF", "BUF", "BUF"],
            "position": ["WR", "OT", "QB", "RB"],
        }
    )
    assert list(roster_choices(roster).items()) == [
        ("Quarter Back (BUF QB)", "3"),
        ("Run Ner (BUF RB)", "4"),
        ("Wide Out (BUF WR)", "1"),
    ]


@pytest.fixture
def game_app(monkeypatch: pytest.MonkeyPatch, game_run: GameRun, tmp_path: Path) -> AppTest:
    from nflsim.dashboard import data

    roster = pl.DataFrame(
        {"gsis_id": ["00-qb"], "name": ["Josh Allen"], "team": ["BUF"], "position": ["QB"]}
    )
    monkeypatch.setattr(data, "seasons", lambda: [2021])
    monkeypatch.setattr(data, "weeks", lambda season: ([2, 3], 3))
    monkeypatch.setattr(data, "week_games", lambda season, week: [("BUF", "KC")])
    monkeypatch.setattr(data, "roster", lambda season, week, home, away: roster)
    seen: list[tuple[object, ...]] = []

    def fake_game_run(game: GameRequest, injuries: bool, *rest: object) -> GameRun:
        seen.append((game, injuries, *rest))
        out = rest[0] if rest else ()
        return replace(game_run, base=game_run.result) if out else game_run

    monkeypatch.setattr(data, "game_run", fake_game_run)
    script = tmp_path / "game_only.py"
    script.write_text("from nflsim.dashboard import game_page\n\ngame_page.render()\n")
    at = AppTest.from_file(str(script), default_timeout=60)
    at.session_state["_seen"] = seen
    return at


def test_game_page_renders_and_passes_what_ifs(game_app: AppTest) -> None:
    at = game_app.run()
    assert not at.exception
    assert at.title[0].value == "BUF at KC"
    labels = [m.label for m in at.metric]
    assert labels == ["KC win", "BUF win", "Expected score (KC-BUF)", "Model line"]
    at.multiselect(key="game_out").set_value(["Josh Allen (BUF QB)"]).run()
    assert not at.exception
    seen = at.session_state["_seen"]
    game, _injuries, out, *_ = seen[-1]
    assert isinstance(game, GameRequest) and (game.home, game.away, game.week) == ("KC", "BUF", 3)
    assert out == ("00-qb",)
    assert at.metric[0].delta is not None  # compared with the run without the what-if
