"""Per-player stat lines (Phase 7, step 3).

Expected stat for a player in a game = team volume x player share x player rate:

- team volume: pass attempts, carries and targets per game, from games before week W this
  season plus last season (weighted ``carryover``), shrunk toward the league average with
  ``volume_prior_games``;
- player share: expected share of the team's attempts / carries / targets (from the
  injury-aware lineup, so injuries and overrides flow through);
- player rate: yards (and touchdowns, receptions) per attempt / carry / target, from the
  same window, shrunk toward the position average with ``rate_prior`` opportunities.
  Quarterbacks' rushing is rated per pass attempt (``QB_RUSH_STATS``): it is mostly
  scrambles on dropbacks, not designed runs.

Everything uses games before week W only. Ranges come from the distribution of scaled
errors ``(actual - expected) / sqrt(expected)`` in the fit seasons (``fit_ranges``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nflsim.config import Opportunity, StatLinesConfig
from nflsim.data.teams import normalize_teams

# Opportunity behind each stat, and the usage role that sets a player's share of it.
OPPORTUNITY_ROLE: dict[Opportunity, str] = {
    "attempts": "dropback",
    "carries": "carry",
    "targets": "target",
}
STAT_OPPORTUNITY: dict[str, Opportunity] = {
    "passing_yards": "attempts",
    "passing_tds": "attempts",
    "rushing_yards": "carries",
    "rushing_tds": "carries",
    "receptions": "targets",
    "receiving_yards": "targets",
    "receiving_tds": "targets",
}
OPPORTUNITIES = list(OPPORTUNITY_ROLE)
STATS = list(STAT_OPPORTUNITY)
# A QB's rushing is mostly scrambles, which happen on dropbacks, so QBs' rushing stats are
# rated per pass attempt and driven by their dropback share instead of designed carries.
QB_RUSH_STATS = ["rushing_yards", "rushing_tds"]
_KEYS = ["season", "week", "team"]


def box_scores(player_stats: pl.DataFrame) -> pl.DataFrame:
    """One row per player-game: ``season, week, game_id, team, gsis_id`` plus
    ``OPPORTUNITIES`` and ``STATS`` (nulls as 0). Current team codes."""
    frame = normalize_teams(player_stats.filter(pl.col("player_id").is_not_null()), ["team"])
    return frame.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "game_id",
        "team",
        pl.col("player_id").alias("gsis_id"),
        *[pl.col(c).cast(pl.Float64).fill_null(0.0) for c in OPPORTUNITIES + STATS],
    )


def _before_week(
    targets: pl.DataFrame, weekly: pl.DataFrame, by: list[str], cols: list[str]
) -> pl.DataFrame:
    """Join each target row with ``cols`` summed over weeks of its season before its week."""
    cumulative = weekly.sort("week").with_columns(pl.col(cols).cum_sum().over(["season", *by]))
    return (
        targets.sort("week")
        .join_asof(
            cumulative.select("season", "week", *by, *cols),
            on="week",
            by=["season", *by],
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .with_columns(pl.col(cols).fill_null(0.0))
    )


def _last_season(weekly: pl.DataFrame, by: list[str], cols: list[str]) -> pl.DataFrame:
    return (
        weekly.group_by("season", *by)
        .agg(pl.col(cols).sum())
        .with_columns(pl.col("season") + 1)
        .rename({c: f"{c}_last" for c in cols})
    )


def team_volume_as_of(
    box: pl.DataFrame, targets: pl.DataFrame, config: StatLinesConfig
) -> pl.DataFrame:
    """Expected pass attempts, carries and targets per game for each target team-week.

    ``(this season + carryover x last season + k x league) / (games + carryover x last
    games + k)``, with the league per-team-game average from the same window. Returns
    ``season, week, team`` and ``vol_<opportunity>``.
    """
    games = (
        box.group_by("season", "week", "team", "game_id")
        .agg(pl.col(OPPORTUNITIES).sum())
        .with_columns(n_games=pl.lit(1.0))
    )
    team_weekly = games.group_by(*_KEYS).agg(pl.col([*OPPORTUNITIES, "n_games"]).sum())
    cols = [*OPPORTUNITIES, "n_games"]
    keys = targets.select(*_KEYS).unique()
    frame = _before_week(keys, team_weekly, ["team"], cols).join(
        _last_season(team_weekly, ["team"], cols), on=["season", "team"], how="left"
    )
    league_weekly = team_weekly.group_by("season", "week").agg(pl.col(cols).sum())
    league = (
        _before_week(keys.select("season", "week").unique(), league_weekly, [], cols)
        .join(_last_season(league_weekly, [], cols), on="season", how="left")
        .select(
            "season",
            "week",
            *[
                (
                    (pl.col(o) + pl.col(f"{o}_last").fill_null(0.0))
                    / (pl.col("n_games") + pl.col("n_games_last").fill_null(0.0))
                ).alias(f"lg_{o}")
                for o in OPPORTUNITIES
            ],
        )
    )
    c, k = config.carryover, config.volume_prior_games
    frame = frame.join(league, on=["season", "week"], how="left").with_columns(
        pl.col([f"{x}_last" for x in cols]).fill_null(0.0)
    )
    games_eff = pl.col("n_games") + c * pl.col("n_games_last") + k
    return frame.select(
        *_KEYS,
        *[
            ((pl.col(o) + c * pl.col(f"{o}_last") + k * pl.col(f"lg_{o}")) / games_eff).alias(
                f"vol_{o}"
            )
            for o in OPPORTUNITIES
        ],
    )


def player_rates_as_of(
    box: pl.DataFrame,
    positions: pl.DataFrame,
    targets: pl.DataFrame,
    config: StatLinesConfig,
) -> pl.DataFrame:
    """Each target player's shrunk rate for every stat, per opportunity.

    ``targets`` has ``season, week, gsis_id, position``; ``positions`` (``season, week,
    gsis_id, position``) labels the box scores for the position averages. Returns
    ``season, week, gsis_id, position``, ``rate_<stat>`` and, for ``QB_RUSH_STATS``,
    ``rate_<stat>_per_att`` (per pass attempt). Rates with no data at all are 0.
    """
    sums = [*OPPORTUNITIES, *STATS]
    weekly = box.group_by("season", "week", "gsis_id").agg(pl.col(sums).sum())
    keys = targets.select("season", "week", "gsis_id", "position").unique(
        ["season", "week", "gsis_id"]
    )
    frame = _before_week(keys, weekly, ["gsis_id"], sums).join(
        _last_season(weekly, ["gsis_id"], sums), on=["season", "gsis_id"], how="left"
    )
    by_pos = (
        weekly.join(positions, on=["season", "week", "gsis_id"], how="inner")
        .group_by("season", "week", "position")
        .agg(pl.col(sums).sum())
    )
    pos_keys = keys.select("season", "week", "position").unique()
    pos = (
        _before_week(pos_keys, by_pos, ["position"], sums)
        .join(_last_season(by_pos, ["position"], sums), on=["season", "position"], how="left")
        .with_columns(pl.col([f"{s}_last" for s in sums]).fill_null(0.0))
        .select(
            "season",
            "week",
            "position",
            *[
                (
                    (pl.col(s) + pl.col(f"{s}_last"))
                    / (pl.col(STAT_OPPORTUNITY[s]) + pl.col(f"{STAT_OPPORTUNITY[s]}_last"))
                )
                .fill_nan(0.0)
                .alias(f"pos_{s}")
                for s in STATS
            ],
            *[
                ((pl.col(s) + pl.col(f"{s}_last")) / (pl.col("attempts") + pl.col("attempts_last")))
                .fill_nan(0.0)
                .alias(f"pos_{s}_per_att")
                for s in QB_RUSH_STATS
            ],
        )
    )
    c = config.carryover
    frame = frame.join(pos, on=["season", "week", "position"], how="left").with_columns(
        pl.col([f"{s}_last" for s in sums]).fill_null(0.0)
    )

    def rate(stat: str, opp: Opportunity, prior: str) -> pl.Expr:
        k = config.rate_prior[opp]
        return (
            (pl.col(stat) + c * pl.col(f"{stat}_last") + k * pl.col(prior).fill_null(0.0))
            / (pl.col(opp) + c * pl.col(f"{opp}_last") + k)
        ).fill_nan(0.0)

    rates = [rate(s, STAT_OPPORTUNITY[s], f"pos_{s}").alias(f"rate_{s}") for s in STATS]
    rates += [
        rate(s, "attempts", f"pos_{s}_per_att").alias(f"rate_{s}_per_att") for s in QB_RUSH_STATS
    ]
    return frame.select("season", "week", "gsis_id", "position", *rates)


def expected_stats(shares: pl.DataFrame, volume: pl.DataFrame, rates: pl.DataFrame) -> pl.DataFrame:
    """Expected stats per player-game.

    ``shares``: ``season, week, team, gsis_id, role, share`` (expected share of the team's
    role plays). Returns ``season, week, team, gsis_id`` and ``exp_<stat>``.
    """
    role_to_opp = {r: o for o, r in OPPORTUNITY_ROLE.items()}
    wide = (
        shares.filter(pl.col("role").is_in(list(role_to_opp)))
        .with_columns(pl.col("role").replace_strict(role_to_opp).alias("opp"))
        .group_by("season", "week", "team", "gsis_id", "opp")
        .agg(pl.col("share").sum())
        .pivot(on="opp", index=["season", "week", "team", "gsis_id"], values="share")
    )
    for o in OPPORTUNITIES:
        if o not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(o))
    frame = (
        wide.with_columns(pl.col(OPPORTUNITIES).fill_null(0.0))
        .join(volume, on=_KEYS, how="left")
        .join(rates, on=["season", "week", "gsis_id"], how="left")
    )

    def per_opportunity(stat: str) -> pl.Expr:
        o = STAT_OPPORTUNITY[stat]
        return pl.col(o) * pl.col(f"vol_{o}") * pl.col(f"rate_{stat}").fill_null(0.0)

    def qb_rush(stat: str) -> pl.Expr:
        return (
            pl.col("attempts")
            * pl.col("vol_attempts")
            * pl.col(f"rate_{stat}_per_att").fill_null(0.0)
        )

    is_qb = pl.col("position") == "QB"
    return frame.select(
        "season",
        "week",
        "team",
        "gsis_id",
        *[
            (
                pl.when(is_qb).then(qb_rush(s)).otherwise(per_opportunity(s))
                if s in QB_RUSH_STATS
                else per_opportunity(s)
            )
            .fill_nan(0.0)
            .alias(f"exp_{s}")
            for s in STATS
        ],
    )


@dataclass(frozen=True)
class Ranges:
    """Quantiles of scaled errors ``(actual - expected) / sqrt(expected)`` per stat."""

    low_q: float
    high_q: float
    low: dict[str, float]
    high: dict[str, float]

    def interval(self, stat: str, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = np.sqrt(np.maximum(expected, 0.0))
        lo = np.maximum(expected + self.low[stat] * scale, 0.0)
        hi = np.maximum(expected + self.high[stat] * scale, 0.0)
        return lo, hi


def fit_ranges(pairs: pl.DataFrame, low_q: float, high_q: float, min_expected: float) -> Ranges:
    """Scaled-error quantiles from ``pairs`` (``exp_<stat>`` and actual ``<stat>``), using
    player-games with an expected value of at least ``min_expected``."""
    low, high = {}, {}
    for s in STATS:
        sub = pairs.filter(pl.col(f"exp_{s}") >= min_expected)
        if sub.is_empty():
            low[s], high[s] = 0.0, 0.0
            continue
        z = (sub[s] - sub[f"exp_{s}"]) / sub[f"exp_{s}"].sqrt()
        low[s] = float(z.quantile(low_q) or 0.0)
        high[s] = float(z.quantile(high_q) or 0.0)
    return Ranges(low_q, high_q, low, high)
