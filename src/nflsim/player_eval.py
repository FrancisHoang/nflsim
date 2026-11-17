"""Out-of-sample checks for player ratings and projected usage (Phase 2).

Ratings: for every player-game in the test seasons, the rating as of that week (built
from earlier games only) predicts the player's EPA per play in that game. Errors are
weighted by plays. Single-game EPA is mostly noise, so absolute errors are large and the
differences between methods are small; ``skill`` (1 - MSE / MSE of the position average)
is the number to compare.

Usage: for every team-week, the projected role shares are compared with the shares that
actually happened, as the total-variation distance (half the summed absolute
differences: 0 = exact, 1 = no overlap).
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from nflsim.config import PlayerRatingsConfig
from nflsim.model.players import player_ratings_as_of, projected_usage

_METHODS = ("position_mean", "season_to_date", "rating")


def evaluate_ratings(
    player_games: pl.DataFrame, seasons: Iterable[int], config: PlayerRatingsConfig
) -> pl.DataFrame:
    """Per-role weighted MSE of each method predicting a player's per-play EPA in a game.

    Methods: ``position_mean`` (the group average alone), ``season_to_date`` (the
    player's raw average this season before the game, or the group average if he has no
    plays yet) and ``rating`` (the shrunk rating).
    """
    test = (
        player_games.filter(pl.col("season").is_in(list(seasons)))
        .group_by("season", "week", "gsis_id", "role", "group")
        .agg(pl.col("n").sum(), pl.col("epa").sum())
    )
    if test.is_empty():
        raise ValueError("no player games in the requested seasons")
    rated = player_ratings_as_of(player_games, test, config)
    frame = (
        test.join(rated, on=["season", "week", "gsis_id", "role", "group"], how="inner")
        .filter(pl.col("rating").is_not_null())
        .with_columns(
            actual=pl.col("epa") / pl.col("n"),
            position_mean=pl.col("group_mean"),
            season_to_date=pl.when(pl.col("cur_n") > 0)
            .then(pl.col("cur_epa") / pl.col("cur_n"))
            .otherwise(pl.col("group_mean")),
        )
    )
    w = pl.col("n")
    mse = {
        m: ((w * (pl.col(m) - pl.col("actual")) ** 2).sum() / w.sum()).alias(f"mse_{m}")
        for m in _METHODS
    }
    return (
        frame.group_by("role")
        .agg(
            pl.len().alias("player_games"),
            w.sum().alias("plays"),
            (pl.col("prior_source") == "unknown").mean().alias("unknown_share"),
            *mse.values(),
        )
        .with_columns(
            *[
                (1 - pl.col(f"mse_{m}") / pl.col("mse_position_mean")).alias(f"skill_{m}")
                for m in ("season_to_date", "rating")
            ]
        )
        .sort("role")
    )


def evaluate_usage(
    history: pl.DataFrame,
    team_games: pl.DataFrame,
    rosters: pl.DataFrame,
    seasons: Iterable[int],
    config: PlayerRatingsConfig,
) -> pl.DataFrame:
    """Per-role mean total-variation distance between projected and actual shares.

    ``history`` and ``team_games`` come from ``usage_history`` and ``player_game_stats``.
    ``no_projection`` is the share of team-weeks where no rostered candidate had history
    in the role (shares null); those are excluded from ``tv_distance``.
    """
    wanted = list(seasons)
    targets = (
        team_games.filter(pl.col("season").is_in(wanted)).select("season", "week", "team").unique()
    )
    if targets.is_empty():
        raise ValueError("no team games in the requested seasons")
    proj = projected_usage(history, rosters, targets, config).select(
        "season", "week", "team", "gsis_id", "role", pl.col("share").alias("projected")
    )
    actual = (
        history.filter(pl.col("season").is_in(wanted))
        .group_by("season", "week", "team", "gsis_id", "role")
        .agg(pl.col("share").sum().alias("actual"))
    )
    key = ["season", "week", "team", "role"]
    projected_ok = proj.group_by(key).agg(pl.col("projected").is_not_null().all().alias("ok"))
    both = (
        proj.join(actual, on=[*key, "gsis_id"], how="full", coalesce=True)
        .join(projected_ok, on=key, how="left")
        .with_columns(pl.col("ok").fill_null(False))
    )
    per_team = both.group_by(key).agg(
        pl.col("ok").first(),
        (
            0.5 * (pl.col("projected").fill_null(0.0) - pl.col("actual").fill_null(0.0)).abs().sum()
        ).alias("tv"),
    )
    return (
        per_team.group_by("role")
        .agg(
            pl.len().alias("team_weeks"),
            (~pl.col("ok")).mean().alias("no_projection"),
            pl.col("tv").filter(pl.col("ok")).mean().alias("tv_distance"),
        )
        .sort("role")
    )
