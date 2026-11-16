from __future__ import annotations

import numpy as np
import pytest

from nflsim.model import simulate_scores

MU_H = np.array([24.0, 20.0, 30.0])
MU_A = np.array([20.0, 20.0, 10.0])


def test_same_seed_same_result_different_seed_differs() -> None:
    a = simulate_scores(MU_H, MU_A, 6.0, 500, seed=42, overtime_points=3)
    b = simulate_scores(MU_H, MU_A, 6.0, 500, seed=42, overtime_points=3)
    c = simulate_scores(MU_H, MU_A, 6.0, 500, seed=43, overtime_points=3)
    assert np.array_equal(a.home, b.home)
    assert np.array_equal(a.away, b.away)
    assert not np.array_equal(a.home, c.home)


def test_shapes_no_ties_and_means() -> None:
    sim = simulate_scores(MU_H, MU_A, 6.0, 100_000, seed=1, overtime_points=3)
    assert sim.home.shape == (3, 100_000)
    assert not np.any(sim.home == sim.away)
    # Overtime adds a little to each side's mean; it should still be close to mu.
    assert np.allclose(sim.home.mean(axis=1), MU_H, atol=0.5)
    assert np.allclose(sim.away.mean(axis=1), MU_A, atol=0.5)
    assert np.allclose(sim.mean_margin(), MU_H - MU_A, atol=0.5)


def test_win_probabilities_ordered_and_symmetric() -> None:
    p = simulate_scores(MU_H, MU_A, 6.0, 100_000, seed=2, overtime_points=3).home_win_prob()
    assert p[1] == pytest.approx(0.5, abs=0.01)  # identical teams
    assert 0.5 < p[0] < p[2] < 1.0


def test_score_variance_matches_negative_binomial() -> None:
    sim = simulate_scores(np.array([22.0]), np.array([22.0]), 6.0, 200_000, 3, 3)
    # Var = mu + mu^2 / r = 22 + 484 / 6 ~ 102.7 (overtime adds a bit).
    assert sim.home.var() == pytest.approx(22 + 22**2 / 6, rel=0.05)


@pytest.mark.parametrize(
    ("mu_h", "mu_a", "r", "n"),
    [
        ([20.0], [20.0, 21.0], 6.0, 10),
        ([20.0], [20.0], 6.0, 0),
        ([20.0], [20.0], 0.0, 10),
        ([0.0], [20.0], 6.0, 10),
    ],
)
def test_invalid_inputs(mu_h: list[float], mu_a: list[float], r: float, n: int) -> None:
    with pytest.raises(ValueError):
        simulate_scores(np.array(mu_h), np.array(mu_a), r, n, seed=0, overtime_points=3)
