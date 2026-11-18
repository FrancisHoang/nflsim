from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nflsim.backtest import walk_forward
from nflsim.distribution import (
    beat_line_probability,
    crps_normal,
    crps_samples,
    key_number_table,
    line_calibration,
    margin_shares,
)
from tests.test_backtest import SETTINGS, season_games


def brute_crps(x: np.ndarray, y: float) -> float:
    return float(np.abs(x - y).mean() - 0.5 * np.abs(x[:, None] - x[None, :]).mean())


def test_crps_samples_matches_brute_force_and_point_forecast() -> None:
    rng = np.random.default_rng(0)
    samples = rng.integers(-20, 21, size=(3, 300))
    actual = np.array([3, -7, 0])
    got = crps_samples(samples, actual)
    want = [brute_crps(samples[i].astype(float), float(actual[i])) for i in range(3)]
    assert got == pytest.approx(want)
    point = crps_samples(np.full((1, 50), 6), np.array([2]))
    assert point[0] == pytest.approx(4.0)  # a point forecast scores its absolute error


def test_crps_normal_matches_large_sample() -> None:
    rng = np.random.default_rng(1)
    draws = rng.normal(3.0, 13.0, size=(1, 4000))
    sampled = crps_samples(draws, np.array([10.0]))[0]
    exact = crps_normal(np.array([3.0]), 13.0, np.array([10.0]))[0]
    assert exact == pytest.approx(sampled, rel=0.03)


def test_beat_line_probability_excludes_pushes_and_missing_lines() -> None:
    samples = np.array([[3, 3, 7, -1], [3, 3, 3, 3], [0, 1, 2, 3]])
    p = beat_line_probability(samples, np.array([3.0, 3.0, np.nan]))
    assert p[0] == pytest.approx(0.5)  # one above, one below, two pushes
    assert np.isnan(p[1])  # all pushes
    assert np.isnan(p[2])  # no line


def test_margin_shares_counts_absolute_margins() -> None:
    shares = margin_shares(np.array([[3, -3, 7, 1]]), [3, 7, 10])
    assert shares[3][0] == 0.5 and shares[7][0] == 0.25 and shares[10][0] == 0.0


def test_walk_forward_adds_distribution_columns() -> None:
    games = season_games(range(2019, 2022))
    preds, _ = walk_forward(games, SETTINGS, [2021], n=400, seed=2)
    for col in ("model_crps_margin", "model_crps_total", "vegas_crps_margin", "model_p_cover"):
        assert preds[col].null_count() == 0
    assert (preds["model_crps_margin"] > 0).all()
    keys = key_number_table(preds, SETTINGS.backtest.key_margins)
    assert keys["abs_margin"].to_list() == list(SETTINGS.backtest.key_margins)
    cal = line_calibration(preds, "cover", 5)
    assert (
        cal["games"].sum() == preds.filter(pl.col("actual_margin") != pl.col("vegas_margin")).height
    )
    with pytest.raises(ValueError, match="kind"):
        line_calibration(preds, "spread", 5)
