"""Scores for whole predicted distributions, not just their means (Phase 5, step 0).

- CRPS (continuous ranked probability score): how far the predicted distribution of a
  quantity is from the single observed value, in the quantity's units. For a point
  forecast it equals the absolute error; spreading probability sensibly lowers it.
  Lower is better.
- Cover and over probabilities against the closing lines (pushes excluded), so their
  calibration can be checked. The lines are a reference point, never a model input.
- Key numbers: the chance of each absolute margin (3, 7, ...), which a smooth score
  distribution gets wrong.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

FloatArray = npt.NDArray[np.float64]


def crps_samples(samples: np.ndarray, actual: np.ndarray) -> FloatArray:
    """CRPS of each row of ``samples`` (games x runs) against ``actual`` (one per game).

    Uses ``CRPS = E|X - y| - E|X - X'| / 2`` with the sorted-sample identity
    ``E|X - X'| = 2 / n^2 * sum_i (2i - n - 1) x_(i)`` (i = 1..n), so it is exact for the
    empirical distribution and needs one sort per game.
    """
    x = np.sort(np.asarray(samples, dtype=np.float64), axis=1)
    y = np.asarray(actual, dtype=np.float64)[:, None]
    n = x.shape[1]
    weights = 2.0 * np.arange(1, n + 1) - n - 1
    spread = 2.0 * (x @ weights) / (n * n)
    return np.asarray(np.abs(x - y).mean(axis=1) - spread / 2.0, dtype=np.float64)


def crps_normal(mean: np.ndarray, sd: float, actual: np.ndarray) -> FloatArray:
    """Closed-form CRPS of a normal distribution with ``mean`` and ``sd``."""
    z = (np.asarray(actual, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / sd
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    return np.asarray(sd * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi)), np.float64)


def beat_line_probability(samples: np.ndarray, line: np.ndarray) -> FloatArray:
    """Chance each game's samples land above its ``line``, pushes excluded.

    ``P(X > line) / (P(X > line) + P(X < line))``; NaN when the line is missing or every
    sample is a push.
    """
    x = np.asarray(samples, dtype=np.float64)
    ln = np.asarray(line, dtype=np.float64)[:, None]
    above = (x > ln).mean(axis=1)
    below = (x < ln).mean(axis=1)
    decided = above + below
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(decided > 0, above / decided, np.nan)
    return np.asarray(np.where(np.isnan(ln[:, 0]), np.nan, p), dtype=np.float64)


def margin_shares(margin: np.ndarray, keys: Sequence[int]) -> dict[int, FloatArray]:
    """Per game, the share of runs with each absolute margin in ``keys``."""
    absolute = np.abs(np.asarray(margin))
    return {k: np.asarray((absolute == k).mean(axis=1), dtype=np.float64) for k in keys}


def distribution_summary(predictions: pl.DataFrame) -> pl.DataFrame:
    """Mean CRPS of margin and total per season and overall, for the model and Vegas.

    Games missing a Vegas line are dropped for both, so the comparison is fair.
    """
    scored = predictions.drop_nulls(["vegas_crps_margin", "vegas_crps_total"])
    groups = [
        (str(s), scored.filter(pl.col("season") == s))
        for s in sorted(scored["season"].unique().to_list())
    ]
    groups.append(("all", scored))
    rows = []
    for label, frame in groups:
        for method in ("model", "vegas"):
            rows.append(
                {
                    "seasons": label,
                    "method": method,
                    "games": frame.height,
                    "crps_margin": _mean(frame, f"{method}_crps_margin"),
                    "crps_total": _mean(frame, f"{method}_crps_total"),
                }
            )
    return pl.DataFrame(rows)


def line_calibration(predictions: pl.DataFrame, kind: str, bins: int) -> pl.DataFrame:
    """Predicted vs. actual rate of covering the spread (``kind="cover"``) or going over
    the total (``kind="over"``), in equal-width probability bins; pushes left out."""
    if kind == "cover":
        actual, line = pl.col("actual_margin"), pl.col("vegas_margin")
    elif kind == "over":
        actual, line = pl.col("actual_total"), pl.col("vegas_total")
    else:
        raise ValueError("kind must be 'cover' or 'over'")
    p = pl.col(f"model_p_{kind}")
    index = (p * bins).floor().clip(0, bins - 1).cast(pl.Int64)
    return (
        predictions.filter(p.is_not_nan() & line.is_not_null() & (actual != line))
        .group_by(index.alias("bin"))
        .agg(
            pl.len().alias("games"),
            p.mean().alias("mean_pred"),
            (actual > line).mean().alias("actual"),
        )
        .with_columns((pl.col("bin") / bins).alias("lo"), ((pl.col("bin") + 1) / bins).alias("hi"))
        .sort("bin")
        .select("lo", "hi", "games", "mean_pred", "actual")
    )


def key_number_table(predictions: pl.DataFrame, keys: Sequence[int]) -> pl.DataFrame:
    """Actual share of games with each absolute margin vs. the model's average chance."""
    absolute = pl.col("actual_margin").abs()
    return pl.DataFrame(
        [
            {
                "abs_margin": k,
                "actual": float(predictions.select((absolute == k).mean()).item()),
                "model": _mean(predictions, f"model_key_{k}"),
            }
            for k in keys
        ]
    )


def _mean(frame: pl.DataFrame, column: str) -> float:
    return float(frame.select(pl.col(column).mean()).item())
