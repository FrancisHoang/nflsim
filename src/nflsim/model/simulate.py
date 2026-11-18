"""Monte Carlo score simulation.

Each team's points are drawn independently from a negative binomial with the model's
expected points and dispersion (historically, home and away scores are nearly
uncorrelated). A tied simulation goes to overtime: the winner is drawn with probability
proportional to each side's expected points and gets ``overtime_points``.

Expected points can be one value per game, or one per game and run (e.g. when player
availability is sampled per run).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SimResult:
    """Simulated scores, shape (games, runs)."""

    home: IntArray
    away: IntArray

    @property
    def margin(self) -> IntArray:
        """Home minus away."""
        return self.home - self.away

    @property
    def total(self) -> IntArray:
        return self.home + self.away

    def home_win_prob(self) -> FloatArray:
        return np.asarray(np.mean(self.home > self.away, axis=1), dtype=np.float64)

    def mean_margin(self) -> FloatArray:
        return np.asarray(np.mean(self.margin, axis=1), dtype=np.float64)

    def mean_total(self) -> FloatArray:
        return np.asarray(np.mean(self.total, axis=1), dtype=np.float64)


def simulate_scores(
    mu_home: FloatArray,
    mu_away: FloatArray,
    dispersion: float,
    n: int,
    seed: int,
    overtime_points: int,
) -> SimResult:
    """Simulate ``n`` runs of each game. Same inputs and seed give identical output.

    ``mu_home``/``mu_away`` have shape ``(games,)`` or ``(games, n)`` (per-run means).
    """
    mu_home = _per_run(mu_home, n)
    mu_away = _per_run(mu_away, n)
    if mu_home.shape[0] != mu_away.shape[0]:
        raise ValueError("mu_home and mu_away must have the same length")
    if dispersion <= 0 or (mu_home <= 0).any() or (mu_away <= 0).any():
        raise ValueError("dispersion and expected points must be positive")

    rng = np.random.default_rng(seed)
    size = (mu_home.shape[0], n)
    # numpy's negative_binomial(r, p) has mean r * (1 - p) / p, so p = r / (r + mu).
    home = rng.negative_binomial(dispersion, dispersion / (dispersion + mu_home), size=size)
    away = rng.negative_binomial(dispersion, dispersion / (dispersion + mu_away), size=size)

    tied = home == away
    home_wins_ot = rng.random(size) < mu_home / (mu_home + mu_away)
    home = home + np.where(tied & home_wins_ot, overtime_points, 0)
    away = away + np.where(tied & ~home_wins_ot, overtime_points, 0)
    return SimResult(home=home.astype(np.int64), away=away.astype(np.int64))


def _per_run(mu: FloatArray, n: int) -> FloatArray:
    if n < 1:
        raise ValueError("n must be at least 1")
    arr = np.asarray(mu, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[1] != n:
            raise ValueError(f"per-run expected points need {n} columns, got {arr.shape[1]}")
        return arr
    return arr.reshape(-1, 1)
