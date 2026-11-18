"""Out-of-sample check of per-player stat lines against box scores (``nflsim stat-eval``)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nflsim.model.stat_lines import STATS, Ranges, fit_ranges


def baseline_per_game(box: pl.DataFrame, keys: pl.DataFrame) -> pl.DataFrame:
    """Each player's average per game of every stat, from his games earlier that season
    (last season's average when he has none yet). ``keys``: ``season, week, gsis_id``.
    Returns ``season, week, gsis_id`` and ``base_<stat>`` (null with no history)."""
    weekly = box.group_by("season", "week", "gsis_id").agg(
        pl.col(STATS).sum(), pl.len().cast(pl.Float64).alias("g")
    )
    cumulative = weekly.sort("week").with_columns(
        pl.col([*STATS, "g"]).cum_sum().over("season", "gsis_id")
    )
    last = (
        weekly.group_by("season", "gsis_id")
        .agg(pl.col([*STATS, "g"]).sum())
        .with_columns(pl.col("season") + 1)
    )
    return (
        keys.select("season", "week", "gsis_id")
        .unique()
        .sort("week")
        .join_asof(
            cumulative,
            on="week",
            by=["season", "gsis_id"],
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .join(last, on=["season", "gsis_id"], how="left", suffix="_l")
        .select(
            "season",
            "week",
            "gsis_id",
            *[
                pl.when(pl.col("g") > 0)
                .then(pl.col(s) / pl.col("g"))
                .otherwise(pl.col(f"{s}_l") / pl.col("g_l"))
                .alias(f"base_{s}")
                for s in STATS
            ],
        )
    )


def evaluate_stat_lines(
    pairs: pl.DataFrame,
    fit_seasons: list[int],
    test_seasons: list[int],
    low_q: float,
    high_q: float,
    min_expected: float,
    n_boot: int,
    seed: int,
) -> tuple[pl.DataFrame, Ranges]:
    """Model vs. the per-game-average baseline on ``test_seasons``.

    ``pairs`` has, per player-game the player played, ``exp_<stat>``, ``base_<stat>`` and
    the actual ``<stat>`` plus ``game_id``. Rows are scored where a baseline exists and
    either prediction is at least ``min_expected`` (so the many zero-vs-zero rows for
    players who never get that stat do not dilute the comparison). Ranges are fit on
    ``fit_seasons``. Returns per stat: rows, both MAEs, the difference with a 95% interval
    from resampling games, and the share of actual results inside the range.
    """
    ranges = fit_ranges(
        pairs.filter(pl.col("season").is_in(fit_seasons)), low_q, high_q, min_expected
    )
    test = pairs.filter(pl.col("season").is_in(test_seasons))
    rng = np.random.default_rng(seed)
    rows = []
    for s in STATS:
        t = test.filter(
            pl.col(f"base_{s}").is_not_null()
            & ((pl.col(f"exp_{s}") >= min_expected) | (pl.col(f"base_{s}") >= min_expected))
        ).sort("game_id", "gsis_id")
        y, e, b = t[s].to_numpy(), t[f"exp_{s}"].to_numpy(), t[f"base_{s}"].to_numpy()
        diff = np.abs(e - y) - np.abs(b - y)
        per_game = (
            pl.DataFrame({"g": t["game_id"], "d": diff})
            .group_by("g")
            .agg(pl.col("d").sum(), pl.len().alias("n"))
            .sort("g")
        )
        d, n = per_game["d"].to_numpy(), per_game["n"].to_numpy().astype(np.float64)
        idx = rng.integers(0, d.size, (n_boot, d.size))
        boots = d[idx].sum(axis=1) / n[idx].sum(axis=1)
        lo_ci, hi_ci = np.quantile(boots, [0.025, 0.975])
        lo, hi = ranges.interval(s, e)
        rows.append(
            {
                "stat": s,
                "rows": int(y.size),
                "mae_model": float(np.abs(e - y).mean()),
                "mae_baseline": float(np.abs(b - y).mean()),
                "diff": float(diff.mean()),
                "ci_lo": float(lo_ci),
                "ci_hi": float(hi_ci),
                "in_range": float(((y >= lo) & (y <= hi)).mean()),
            }
        )
    return pl.DataFrame(rows), ranges
