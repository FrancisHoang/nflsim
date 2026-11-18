from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nflsim.config import DataConfig
from nflsim.data import DataCache
from nflsim.game import GameRequest, run_game, stat_line_cells
from nflsim.pipeline import predict_matchup
from tests.test_pipeline import SETTINGS, rated_games, with_matchup

SCHEDULE = pl.DataFrame(
    {
        "game_id": ["2021_03_BUF_KC"],
        "season": [2021],
        "week": [3],
        "game_type": ["REG"],
        "home_team": ["KC"],
        "away_team": ["BUF"],
        "home_score": [27],
        "away_score": [24],
        "location": ["Home"],
        "roof": ["outdoors"],
        "home_rest": [7],
        "away_rest": [7],
        "temp": [70],
        "wind": [5],
        "spread_line": [2.5],
        "total_line": [54.5],
    },
    schema_overrides={"season": pl.Int32, "week": pl.Int32},
)


@pytest.fixture
def cache(tmp_path: Path) -> DataCache:
    c = DataCache(DataConfig(cache_dir=tmp_path), offline=True)
    path = c.path("schedules", 2021)
    path.parent.mkdir(parents=True)
    SCHEDULE.write_parquet(path)
    return c


def test_run_game_matches_predict_matchup_and_scores_the_lines(cache: DataCache) -> None:
    games = with_matchup(rated_games())
    g = run_game(
        cache, SETTINGS, GameRequest(2021, 3, "KC", "BUF", n=500, seed=3), loaded=(games, None)
    )
    alone = predict_matchup(games, SETTINGS, 500, 3)
    np.testing.assert_array_equal(g.result.home_scores, alone.home_scores)
    assert g.result.home_win_prob == alone.home_win_prob
    assert g.matchup.scheduled_id == "2021_03_BUF_KC" and g.matchup.final == (27, 24)
    assert not g.neutral_site
    # The schedule's lines: spread (home KC -2.5 in sportsbook sign) and total.
    assert [x.kind for x in g.lines] == ["spread", "total"]
    assert g.lines[0].line == -2.5 and g.lines[1].line == 54.5
    assert g.report is None and g.effects is None and g.base is None and g.stat_frame is None
    assert g.used.report_error is not None  # no injury reports cached: reported, not fatal


def test_run_game_uses_given_lines(cache: DataCache) -> None:
    games = with_matchup(rated_games())
    g = run_game(
        cache,
        SETTINGS,
        GameRequest(2021, 3, "KC", "BUF", n=200, seed=1),
        spread=-7.0,
        total=40.0,
        loaded=(games, None),
    )
    assert [(x.line, x.source) for x in g.lines] == [(-7.0, "given"), (40.0, "given")]


def test_stat_line_cells_formats_expected_and_range() -> None:
    frame = pl.DataFrame(
        {
            "team": ["KC"],
            "player": ["A"],
            "gsis_id": ["00-1"],
            "position": [None],
            **{
                f"{k}_{s}": [v]
                for s in (
                    "passing_yards",
                    "passing_tds",
                    "rushing_yards",
                    "rushing_tds",
                    "receptions",
                    "receiving_yards",
                    "receiving_tds",
                )
                for k, v in (("exp", 0.0), ("low", 0.0), ("high", 0.0))
            },
        },
        schema_overrides={"position": pl.String},
    ).with_columns(exp_passing_yards=250.4, low_passing_yards=150.0, high_passing_yards=380.0)
    cells = stat_line_cells(frame).row(0, named=True)
    assert cells["pass yds"] == "250 (150-380)"
    assert cells["rush yds"] == ""  # rounds to 0: left blank
    assert cells["pos"] == ""
