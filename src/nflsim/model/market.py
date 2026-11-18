"""Team strength implied by closing spreads of earlier weeks (opt-in ``mkt`` features).

For each season and week W, team strengths ``s`` (points of margin against an average
team at a neutral site) come from a ridge fit over that season's games before W:

    spread_line = home * home_sign + s_home - s_away

with ``s`` shrunk toward ``carryover`` x last season's strengths, ``prior_games`` games of
weight per team. ``spread_line`` is nflverse's (positive when home is favored). Lines are
public before kickoff, so a week-W strength uses no information from week W or later.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nflsim.config import MarketConfig


def market_ratings(schedules: pl.DataFrame, config: MarketConfig) -> pl.DataFrame:
    """``season, week, team, mkt, mkt_games`` for every team and week in ``schedules``.

    ``schedules`` must use normalized team codes. ``mkt_games`` counts the season's games
    with a line before the week; ``mkt`` is 0 (average) with no lines at all.
    """
    sched = schedules.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "home_team",
        "away_team",
        pl.col("spread_line").cast(pl.Float64),
        pl.when(pl.col("location") == "Neutral").then(0.0).otherwise(1.0).alias("home_sign"),
    )
    lined = sched.filter(pl.col("spread_line").is_not_null())
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
        weeks = sorted(cur_sched["week"].unique().to_list())
        values: list[float] = []
        counts: list[int] = []
        for week in weeks:
            before = cur.filter(pl.col("week") < week)
            values.extend(_fit(before, index, config.prior_games, prior).tolist())
            games = np.zeros(len(teams), dtype=np.int64)
            for col in ("home_team", "away_team"):
                for t in before[col].to_list():
                    games[index[t]] += 1
            counts.extend(games.tolist())
        frames.append(
            pl.DataFrame(
                {
                    "season": [season] * len(values),
                    "week": [w for w in weeks for _ in teams],
                    "team": teams * len(weeks),
                    "mkt": values,
                    "mkt_games": counts,
                },
                schema_overrides={"season": pl.Int32, "week": pl.Int32, "mkt_games": pl.Int64},
            )
        )
    return pl.concat(frames, how="vertical")


def _fit(
    games: pl.DataFrame, index: dict[str, int], strength: float, prior: np.ndarray
) -> np.ndarray:
    """Ridge fit of team strengths (and an unpenalized home value) to the spreads."""
    size = len(index)
    if games.is_empty():
        return prior.copy()
    h = np.array([index[t] for t in games["home_team"].to_list()], dtype=np.int64)
    a = np.array([index[t] for t in games["away_team"].to_list()], dtype=np.int64)
    rows = np.arange(games.height)
    x = np.zeros((games.height, size + 1))
    x[rows, h] = 1.0
    x[rows, a] = -1.0
    x[:, size] = games["home_sign"].to_numpy()
    y = games["spread_line"].to_numpy()
    penalty = np.full(size + 1, strength)
    penalty[size] = 1e-6  # home field: (almost) unpenalized
    lhs = x.T @ x + np.diag(penalty)
    rhs = x.T @ y + penalty * np.append(prior, 0.0)
    return np.asarray(np.linalg.solve(lhs, rhs)[:size], dtype=np.float64)


__all__ = ["market_ratings"]
