"""Team strength and scoring tendency implied by closing lines of earlier weeks.

For each season and week W, team strengths ``s`` (points of margin against an average
team at a neutral site; features ``mkt``, ``opp_mkt``) come from a ridge fit over that
season's games before W:

    spread_line = home * home_sign + s_home - s_away

and scoring tendencies ``t`` (points above average in the team's games; features
``mkt_tot``, ``opp_mkt_tot``) from the same kind of fit of the closing totals:

    total_line = league + t_home + t_away

Both are shrunk toward ``carryover`` x last season's values, ``prior_games`` games of
weight per team; home field and the league total are unpenalized. ``spread_line`` is
nflverse's (positive when home is favored). Lines are public before kickoff, so a week-W
value uses no information from week W or later.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nflsim.config import MarketConfig


def market_ratings(schedules: pl.DataFrame, config: MarketConfig) -> pl.DataFrame:
    """``season, week, team, mkt, mkt_tot, mkt_games, mkt_prior_games`` per team and week.

    ``schedules`` must use normalized team codes. ``mkt_games`` counts the team's games
    this season with a spread before the week, ``mkt_prior_games`` its games with a spread
    last season; ``mkt`` and ``mkt_tot`` are 0 (average) with no lines at all.
    """
    sched = schedules.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "home_team",
        "away_team",
        pl.col("spread_line").cast(pl.Float64),
        pl.col("total_line").cast(pl.Float64),
        pl.when(pl.col("location") == "Neutral").then(0.0).otherwise(1.0).alias("home_sign"),
    )
    lined = sched.filter(pl.col("spread_line").is_not_null())
    totaled = sched.filter(pl.col("total_line").is_not_null())
    frames: list[pl.DataFrame] = []
    for season in sorted(sched["season"].unique().to_list()):
        cur_sched = sched.filter(pl.col("season") == season)
        cur = lined.filter(pl.col("season") == season)
        prev = lined.filter(pl.col("season") == season - 1)
        teams = sorted(
            set(cur_sched["home_team"].to_list())
            | set(cur_sched["away_team"].to_list())
            | set(prev["home_team"].to_list())
            | set(prev["away_team"].to_list())
        )
        index = {t: i for i, t in enumerate(teams)}
        zeros = np.zeros(len(teams))
        prior = config.carryover * _fit(prev, index, config.prior_season_ridge, zeros)
        cur_tot = totaled.filter(pl.col("season") == season)
        prev_tot = totaled.filter(pl.col("season") == season - 1)
        prior_tot = config.carryover * _fit(
            prev_tot, index, config.prior_season_ridge, zeros, total=True
        )
        weeks = sorted(cur_sched["week"].unique().to_list())
        prior_counts = _game_counts(prev, index)
        values: list[float] = []
        totals: list[float] = []
        counts: list[int] = []
        for week in weeks:
            before = cur.filter(pl.col("week") < week)
            values.extend(_fit(before, index, config.prior_games, prior).tolist())
            before_tot = cur_tot.filter(pl.col("week") < week)
            totals.extend(
                _fit(before_tot, index, config.prior_games, prior_tot, total=True).tolist()
            )
            counts.extend(_game_counts(before, index).tolist())
        frames.append(
            pl.DataFrame(
                {
                    "season": [season] * len(values),
                    "week": [w for w in weeks for _ in teams],
                    "team": teams * len(weeks),
                    "mkt": values,
                    "mkt_tot": totals,
                    "mkt_games": counts,
                    "mkt_prior_games": prior_counts.tolist() * len(weeks),
                },
                schema_overrides={
                    "season": pl.Int32,
                    "week": pl.Int32,
                    "mkt_games": pl.Int64,
                    "mkt_prior_games": pl.Int64,
                },
            )
        )
    return pl.concat(frames, how="vertical")


def _game_counts(games: pl.DataFrame, index: dict[str, int]) -> np.ndarray:
    """Games per team in ``games``."""
    counts = np.zeros(len(index), dtype=np.int64)
    for col in ("home_team", "away_team"):
        for t in games[col].to_list():
            counts[index[t]] += 1
    return counts


def _fit(
    games: pl.DataFrame,
    index: dict[str, int],
    strength: float,
    prior: np.ndarray,
    *,
    total: bool = False,
) -> np.ndarray:
    """Ridge fit of team values to the lines.

    Spreads (default): ``s_home - s_away`` plus an unpenalized home value. Totals
    (``total=True``): ``t_home + t_away`` plus an unpenalized league total.
    """
    size = len(index)
    if games.is_empty():
        return prior.copy()
    h = np.array([index[t] for t in games["home_team"].to_list()], dtype=np.int64)
    a = np.array([index[t] for t in games["away_team"].to_list()], dtype=np.int64)
    rows = np.arange(games.height)
    x = np.zeros((games.height, size + 1))
    x[rows, h] = 1.0
    x[rows, a] = 1.0 if total else -1.0
    x[:, size] = 1.0 if total else games["home_sign"].to_numpy()
    y = games["total_line" if total else "spread_line"].to_numpy()
    penalty = np.full(size + 1, strength)
    penalty[size] = 1e-6  # home field / league total: (almost) unpenalized
    lhs = x.T @ x + np.diag(penalty)
    rhs = x.T @ y + penalty * np.append(prior, 0.0)
    return np.asarray(np.linalg.solve(lhs, rhs)[:size], dtype=np.float64)


__all__ = ["market_ratings"]
