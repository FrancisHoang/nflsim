from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from nflsim.backtest import american_to_prob, calibration, format_table, summarize, walk_forward
from nflsim.config import ModelConfig, Settings
from tests.test_points import synthetic_games

SETTINGS = Settings(model=ModelConfig(features=("off_epa", "def_epa"), train_seasons=2))
TRUTH = {"intercept": 0.0, "home": 1.0, "off_epa": 30.0, "def_epa": 25.0}


def season_games(seasons: range, per_season: int = 300) -> pl.DataFrame:
    frames = []
    for i, season in enumerate(seasons):
        g = synthetic_games(per_season, TRUTH, seed=i)
        frames.append(
            g.with_columns(
                season=pl.lit(season, dtype=pl.Int32),
                week=pl.Series(np.arange(per_season) % 18 + 1, dtype=pl.Int32),
                game_id=pl.format("{}_{}", pl.lit(season), pl.int_range(pl.len())),
                game_type=pl.lit("REG"),
                home_team=pl.lit("KC"),
                away_team=pl.lit("BUF"),
                spread_line=pl.lit(3.0),
                total_line=pl.lit(44.0),
                home_moneyline=pl.lit(-150.0),
                away_moneyline=pl.lit(130.0),
            )
        )
    return pl.concat(frames)


def test_american_odds_to_probability() -> None:
    out = pl.select(fav=american_to_prob(pl.lit(-150.0)), dog=american_to_prob(pl.lit(130.0))).row(
        0
    )
    assert out[0] == pytest.approx(0.6)
    assert out[1] == pytest.approx(100 / 230)


def test_walk_forward_trains_only_on_earlier_seasons() -> None:
    games = season_games(range(2018, 2023))
    preds, fits = walk_forward(games, SETTINGS, [2021, 2022], n=200, seed=7)
    assert [f.season for f in fits] == [2021, 2022]
    assert fits[0].train_seasons == (2019, 2020)
    assert fits[1].train_seasons == (2020, 2021)
    assert preds.height == 600
    assert preds["season"].unique().sort().to_list() == [2021, 2022]


def test_walk_forward_ignores_test_season_scores_when_fitting() -> None:
    games = season_games(range(2018, 2022))
    _, base = walk_forward(games, SETTINGS, [2021], n=50, seed=1)
    scrambled = games.with_columns(
        pl.when(pl.col("season") == 2021)
        .then(99.0)
        .otherwise(pl.col("home_score"))
        .alias("home_score")
    )
    _, again = walk_forward(scrambled, SETTINGS, [2021], n=50, seed=1)
    assert base[0].model == again[0].model


def test_walk_forward_is_reproducible_and_flags_unrated_games() -> None:
    games = season_games(range(2019, 2022))
    games = games.with_columns(
        has_ratings=pl.when(pl.col("game_id") == "2021_0").then(False).otherwise(True)
    )
    a, fits = walk_forward(games, SETTINGS, [2021], n=100, seed=3)
    b, _ = walk_forward(games, SETTINGS, [2021], n=100, seed=3)
    assert a.equals(b)
    assert fits[0].skipped_games == 1
    assert "2021_0" not in a["game_id"].to_list()
    p = a["model_p"].to_numpy()
    assert np.all((p > 0) & (p < 1))  # clipped away from 0 and 1 for log loss


def test_vegas_probability_moneyline_and_spread_fallback() -> None:
    games = season_games(range(2019, 2022)).with_columns(
        home_moneyline=pl.when(pl.col("game_id") == "2021_0")
        .then(None)
        .otherwise(pl.col("home_moneyline"))
    )
    preds, _ = walk_forward(games, SETTINGS, [2021], n=50, seed=0)
    ml = preds.filter(pl.col("vegas_p_source") == "moneyline")["vegas_p"][0]
    assert ml == pytest.approx(0.6 / (0.6 + 100 / 230))  # margin removed
    fallback = preds.filter(game_id="2021_0")
    assert fallback["vegas_p_source"].item() == "spread"
    assert 0.5 < fallback["vegas_p"].item() < 1.0  # home favored by 3


def predictions_frame() -> pl.DataFrame:
    rows = {
        "season": [2021, 2021, 2022, 2022],
        "week": [1, 2, 1, 2],
        "actual_margin": [7.0, -3.0, 0.0, 10.0],
        "actual_total": [41.0, 37.0, 40.0, 50.0],
        "outcome": [1.0, 0.0, 0.5, 1.0],
    }
    frame = pl.DataFrame(rows)
    for method, p in (
        ("model", [0.6, 0.4, 0.5, 0.7]),
        ("vegas", [0.7, 0.3, 0.5, 0.8]),
        ("home", [0.57] * 4),
    ):
        frame = frame.with_columns(
            pl.Series(f"{method}_p", p),
            pl.Series(f"{method}_margin", [3.0, -1.0, 0.0, 6.0]),
            pl.Series(f"{method}_total", [44.0] * 4),
        )
    return frame


def test_summarize_metrics_by_hand() -> None:
    out = summarize(predictions_frame())
    row = out.filter(seasons="all", method="model").row(0, named=True)
    assert row["games"] == 4
    assert row["margin_mae"] == pytest.approx((4 + 2 + 0 + 4) / 4)
    assert row["total_mae"] == pytest.approx((3 + 7 + 4 + 6) / 4)
    assert row["brier"] == pytest.approx((0.16 + 0.16 + 0 + 0.09) / 4)
    expected_ll = -(math.log(0.6) + math.log(0.6) + math.log(0.5) + math.log(0.7)) / 4
    assert row["log_loss"] == pytest.approx(expected_ll)
    assert row["pick_rate"] == pytest.approx(1.0)  # the tie is excluded
    assert set(out["seasons"]) == {"2021", "2022", "all"}


def test_calibration_bins() -> None:
    table = calibration(predictions_frame(), "vegas", bins=10)
    assert table["games"].sum() == 4
    top = table.filter((pl.col("lo") - 0.8).abs() < 1e-9).row(0, named=True)
    assert top["games"] == 1
    assert top["actual"] == 1.0


def test_format_table_is_ascii() -> None:
    text = format_table(pl.DataFrame({"a": [1.23456], "b": ["x"]}), [("a", 2)])
    assert text.isascii()
    assert "1.23" in text
