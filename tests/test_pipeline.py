from __future__ import annotations

import polars as pl
import pytest

from nflsim.config import ModelConfig, Settings
from nflsim.data import DataCache
from nflsim.pipeline import MATCHUP_ID, matchup_row, predict_matchup
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
