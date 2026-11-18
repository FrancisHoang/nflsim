from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nflsim.config import DataConfig, ModelConfig, Settings
from nflsim.data import DataCache
from nflsim.pipeline import predict_week
from nflsim.weekly import (
    cached_seasons,
    current_week,
    group_flags,
    injury_settings,
    pick_record,
    run_week,
    scheduled_weeks,
)
from tests.test_pipeline import SETTINGS, rated_games


def schedule_from(games: pl.DataFrame, season: int) -> pl.DataFrame:
    """A schedules-shaped frame for ``season`` from the synthetic rated games."""
    return games.filter(pl.col("season") == season).select(
        "game_id",
        "season",
        "week",
        "game_type",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "spread_line",
        "total_line",
        pl.lit("2021-09-12").alias("gameday"),
        pl.lit("13:00").alias("gametime"),
        pl.lit("Home").alias("location"),
    )


@pytest.fixture
def week_cache(tmp_path: Path) -> DataCache:
    cache = DataCache(DataConfig(cache_dir=tmp_path), offline=True)
    sched = schedule_from(rated_games(), 2021).with_columns(
        # Weeks 10 and later not played yet.
        pl.when(pl.col("week") >= 10).then(None).otherwise(pl.col(c)).alias(c)
        for c in ("home_score", "away_score")
    )
    path = cache.path("schedules", 2021)
    path.parent.mkdir(parents=True)
    sched.write_parquet(path)
    return cache


def test_run_week_matches_predict_week(week_cache: DataCache) -> None:
    games = rated_games()
    run = run_week(week_cache, SETTINGS, 2021, 3, n=300, seed=5, loaded=(games, None))
    alone = predict_week(games, SETTINGS, 2021, 3, n=300, seed=5)
    by_id = {p.game_id: p.result for p in alone}
    assert run.frame.height == len(alone) == run.simulated
    for row in run.frame.iter_rows(named=True):
        result = by_id[row["game_id"]]
        assert result is not None and row["home_win"] == result.home_win_prob
    assert run.ratings_through == 2
    assert (
        run.completed_before
        == games.filter((pl.col("season") == 2021) & (pl.col("week") < 3)).height
    )
    assert not run.market and not run.injury_aware and run.offline
    assert run.pbp_cached_at is None  # no play-by-play file in this cache


def test_cached_seasons_and_weeks(week_cache: DataCache) -> None:
    assert cached_seasons(week_cache, "schedules") == [2021]
    assert cached_seasons(week_cache, "pbp") == []
    assert scheduled_weeks(week_cache, 2021) == list(range(1, 19))
    assert scheduled_weeks(week_cache, 2020) == []
    assert current_week(week_cache, 2021) == 10
    assert current_week(week_cache, 2020) is None


def test_group_flags_lists_games_per_flag(week_cache: DataCache) -> None:
    games = rated_games()
    run = run_week(week_cache, SETTINGS, 2021, 3, n=100, seed=1, loaded=(games, None))
    flags = group_flags(run.predictions)
    assert flags == run.flags
    for flag, labels in flags.items():
        assert flag and all(label == "BUF@KC" for label in labels)


def frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "home_win": pl.Float64,
            "vegas_spread": pl.Float64,
            "home_score": pl.Int64,
            "away_score": pl.Int64,
        },
    )


def test_pick_record_counts_model_and_vegas_winners() -> None:
    f = frame(
        [
            {"home_win": 0.7, "vegas_spread": -3.0, "home_score": 24, "away_score": 17},  # both
            {"home_win": 0.4, "vegas_spread": -1.0, "home_score": 10, "away_score": 20},  # model
            {"home_win": 0.6, "vegas_spread": 0.0, "home_score": 20, "away_score": 13},  # pick'em
            {"home_win": 0.6, "vegas_spread": -2.0, "home_score": 17, "away_score": 17},  # tie
            {"home_win": 0.6, "vegas_spread": -2.0, "home_score": None, "away_score": None},
        ]
    )
    r = pick_record(f)
    assert (r.games, r.model, r.vegas, r.vegas_picks) == (3, 3, 1, 2)


def test_injury_settings_adds_player_features_only_when_on() -> None:
    base = Settings(model=ModelConfig(features=("off_epa",)))
    assert injury_settings(base, None) is base
    on = injury_settings(base, True)
    assert on.model.injury_aware and {"off_qb", "off_rush", "off_recv"} <= set(on.model.features)
    off = injury_settings(base, False)
    assert not off.model.injury_aware and off.model.features == ("off_epa",)
