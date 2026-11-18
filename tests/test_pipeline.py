from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nflsim.config import ModelConfig, Settings
from nflsim.data import DataCache
from nflsim.pipeline import MATCHUP_ID, matchup_row, predict_matchup, predict_week
from tests.test_backtest import season_games

SETTINGS = Settings(model=ModelConfig(features=("off_epa", "def_epa"), train_seasons=2))


def with_matchup(games: pl.DataFrame, **changes: object) -> pl.DataFrame:
    template = games.filter(pl.col("season") == 2021).head(1)
    row = template.with_columns(
        game_id=pl.lit(MATCHUP_ID),
        home_score=pl.lit(None, dtype=pl.Float64),
        away_score=pl.lit(None, dtype=pl.Float64),
        h_prior_source=pl.lit("prior_season"),
        a_prior_source=pl.lit("prior_season"),
        h_games_played=pl.lit(5),
        a_games_played=pl.lit(0),
        **{k: pl.lit(v) for k, v in changes.items()},
    )
    return pl.concat([games, row], how="diagonal_relaxed")


def test_matchup_row_normalizes_and_validates() -> None:
    row = matchup_row(2019, 3, "oak", "kc", neutral=True).row(0, named=True)
    assert row["home_team"] == "LV"
    assert row["away_team"] == "KC"
    assert row["location"] == "Neutral"
    assert row["home_score"] is None
    with pytest.raises(ValueError, match="differ"):
        matchup_row(2019, 3, "KC", "kc", neutral=False)


def test_predict_matchup_reproducible_with_flags() -> None:
    games = with_matchup(season_games(range(2019, 2022)))
    a = predict_matchup(games, SETTINGS, n=2000, seed=11)
    b = predict_matchup(games, SETTINGS, n=2000, seed=11)
    assert a.home_win_prob == b.home_win_prob
    assert 0 < a.home_win_prob < 1
    assert a.model.train_games == 600  # 2019 and 2020 only; the 2021 season is excluded
    assert any("no games yet this season" in f for f in a.flags)
    assert list(a.margin_quantiles) == sorted(a.margin_quantiles)
    # The runs behind the summary are kept (line probabilities, JSON).
    assert a.home_scores.shape == a.away_scores.shape == (2000,)
    assert (a.home_scores > a.away_scores).mean() == a.home_win_prob
    assert (a.home_scores - a.away_scores).mean() == pytest.approx(a.mean_margin)
    np.testing.assert_array_equal(a.home_scores, b.home_scores)


def test_predict_matchup_unknown_team_raises() -> None:
    games = with_matchup(season_games(range(2019, 2022)), h_off_epa=None)
    with pytest.raises(ValueError, match="no ratings"):
        predict_matchup(games, SETTINGS, n=100, seed=0)


def test_cache_scan_is_lazy_and_stacks_seasons(cache: DataCache) -> None:
    lazy = cache.scan("games", [2022, 2023])
    assert isinstance(lazy, pl.LazyFrame)
    frame = lazy.collect()
    assert frame.height == 4
    assert frame.filter(pl.col("season") == 2022)["spread_line"].is_null().all()


def test_predict_matchup_rejects_unknown_abbreviation() -> None:
    games = with_matchup(season_games(range(2019, 2022)), home_team="XYZ")
    with pytest.raises(ValueError, match="unknown team"):
        predict_matchup(games, SETTINGS, n=100, seed=0)


def test_predict_matchup_flags_missing_injury_report() -> None:
    games = with_matchup(
        season_games(range(2019, 2022)), h_injury_report=True, a_injury_report=False
    )
    result = predict_matchup(games, SETTINGS, n=200, seed=1)
    report_flags = [f for f in result.flags if "no injury report" in f]
    assert len(report_flags) == 1
    assert report_flags[0].startswith(result.away)


def test_predict_matchup_samples_availability_from_lineups() -> None:
    from tests.test_availability import LINEUP

    games = with_matchup(
        season_games(range(2019, 2022)), h_off_qb=0.0, a_off_qb=0.0, season=2024, week=5
    ).with_columns(
        pl.col("h_off_qb", "a_off_qb").fill_null(0.0),
    )
    games = games.with_columns(
        pl.when(pl.col("game_id") == MATCHUP_ID)
        .then(pl.lit(2024))
        .otherwise(pl.col("season"))
        .cast(pl.Int32)
        .alias("season")
    )
    settings = Settings(
        model=ModelConfig(features=("off_epa", "off_qb"), train_seasons=5, injury_aware=True)
    )
    a = predict_matchup(games, settings, n=4000, seed=2, lineup=LINEUP)
    b = predict_matchup(games, settings, n=4000, seed=2, lineup=LINEUP)
    flat = predict_matchup(games, settings, n=4000, seed=2)
    assert a.sampled_availability and not flat.sampled_availability
    assert a.home_win_prob == b.home_win_prob
    assert a.margin_quantiles == b.margin_quantiles


def test_home_roof_fills_hypothetical_games_from_latest_home_game() -> None:
    from nflsim.pipeline import home_roof

    schedules = pl.DataFrame(
        {
            "season": [2019, 2020, 2020],
            "week": [1, 1, 2],
            "home_team": ["OAK", "LV", "KC"],
            "away_team": ["KC", "KC", "LV"],
            "location": ["Home", "Home", "Home"],
            "roof": ["outdoors", "dome", "outdoors"],
        }
    )
    extra = pl.concat(
        [
            matchup_row(2021, 3, "LV", "KC", neutral=False),
            matchup_row(2021, 3, "KC", "LV", neutral=True),
        ]
    ).with_columns(game_id=pl.Series(["SIM", "SIM2"]))
    out = {r["game_id"]: r for r in home_roof(extra, schedules).iter_rows(named=True)}
    assert out["SIM"]["roof"] == "dome"  # LV's latest home game (2020), not OAK's
    assert out["SIM"]["roof_inferred"] is True
    assert out["SIM2"]["roof"] is None  # neutral site: unknown
    assert out["SIM2"]["roof_inferred"] is False
    known = extra.with_columns(roof=pl.lit("closed"))
    assert home_roof(known, schedules)["roof"].to_list() == ["closed", "closed"]


def test_predict_matchup_flags_teams_without_closing_lines() -> None:
    market = Settings(
        model=ModelConfig(features=("off_epa", "def_epa", "mkt", "opp_mkt"), train_seasons=2)
    )
    games = season_games(range(2019, 2022)).with_columns(
        h_mkt=pl.lit(0.0), a_mkt=pl.lit(0.0), h_mkt_games=pl.lit(3), a_mkt_games=pl.lit(3)
    )
    games = with_matchup(
        games, h_mkt_games=0, h_mkt_prior_games=17, a_mkt_games=0, a_mkt_prior_games=0
    )
    flags = predict_matchup(games, market, n=200, seed=1).flags
    assert any(f.startswith("KC: no closing lines yet") and "carried over" in f for f in flags)
    assert any(f.startswith("BUF: no closing lines cached") for f in flags)
    plain = predict_matchup(games, SETTINGS, n=200, seed=1).flags
    assert not any("closing lines" in f for f in plain)


def rated_games() -> pl.DataFrame:
    """``season_games`` with the prior and games-played columns real game tables carry."""
    return season_games(range(2019, 2022)).with_columns(
        h_prior_source=pl.lit("prior_season"),
        a_prior_source=pl.lit("prior_season"),
        h_games_played=pl.lit(5),
        a_games_played=pl.lit(5),
    )


def test_predict_week_matches_predict_matchup_game_by_game() -> None:
    games = rated_games()
    week = predict_week(games, SETTINGS, 2021, 3, n=300, seed=5)
    ids = games.filter((pl.col("season") == 2021) & (pl.col("week") == 3))["game_id"]
    assert [p.game_id for p in week] == sorted(ids.to_list())
    first = week[0]
    alone = predict_matchup(
        games.with_columns(
            pl.when(pl.col("game_id") == first.game_id)
            .then(pl.lit(MATCHUP_ID))
            .otherwise(pl.col("game_id"))
            .alias("game_id")
        ),
        SETTINGS,
        n=300,
        seed=5,
    )
    assert first.result is not None
    assert first.result.home_win_prob == alone.home_win_prob
    np.testing.assert_array_equal(first.result.home_scores, alone.home_scores)


def test_predict_week_reports_unratable_games_and_rejects_empty_weeks() -> None:
    games = rated_games()
    target = games.filter((pl.col("season") == 2021) & (pl.col("week") == 3))["game_id"][0]
    broken = games.with_columns(
        pl.when(pl.col("game_id") == target)
        .then(None)
        .otherwise(pl.col("h_off_epa"))
        .alias("h_off_epa")
    )
    week = predict_week(broken, SETTINGS, 2021, 3, n=100, seed=0)
    bad = next(p for p in week if p.game_id == target)
    assert bad.result is None and bad.error is not None and "no ratings" in bad.error
    assert all(p.result is not None for p in week if p.game_id != target)
    with pytest.raises(ValueError, match="no games scheduled"):
        predict_week(games, SETTINGS, 2021, 40, n=100, seed=0)


def test_predict_matchup_flags_teams_without_a_listed_starter() -> None:
    qb = Settings(model=ModelConfig(features=("off_epa", "def_epa", "qb_change"), train_seasons=2))
    games = rated_games().with_columns(
        h_qb_change=pl.lit(0.0),
        a_qb_change=pl.lit(0.0),
        h_qb_starter_listed=pl.lit(True),
        a_qb_starter_listed=pl.lit(True),
    )
    games = with_matchup(games, h_qb_starter_listed=True, a_qb_starter_listed=False)
    flags = predict_matchup(games, qb, n=200, seed=1).flags
    listed = [f for f in flags if "no starting QB listed" in f]
    assert len(listed) == 1 and listed[0].startswith("BUF")
    assert not any("starting QB" in f for f in predict_matchup(games, SETTINGS, 200, 1).flags)
