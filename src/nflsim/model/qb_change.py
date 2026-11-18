"""Quarterback change since the games the team's ratings were built on (opt-in features).

Team ratings (EPA, market strength) describe the quarterbacks who played in the team's
earlier games. When a different quarterback starts, they are off by the difference:

    qb_change = dev(this week's QB) - baseline

where ``dev`` is a QB's dropback rating minus the league dropback average (both as of week
W, ``nflsim.model.players``), and ``baseline`` is the dropback-weighted ``dev`` of the
QBs who played for the team this season before week W (last season's when it has no games
yet). Every rating is as of week W, so only *who* plays differs, not when they were rated.

Two sources for this week's QB:

- ``qb_change``: the starter nflverse lists in the schedule (``home_qb_id``/``away_qb_id``;
  for past games the actual starter, for upcoming games nflverse's expected one). Starters
  are usually announced before kickoff, but a surprise start makes a backtest slightly
  optimistic. Falls back to ``qb_change_proj`` when no starter is listed.
- ``qb_change_proj``: the projected QB value ``off_qb`` (expected dropback share x
  ``dev``; injury-aware when ``model.injury_aware``) minus the baseline. Uses only
  information from before the game.
"""

from __future__ import annotations

import polars as pl

from nflsim.config import PlayerRatingsConfig
from nflsim.data.teams import normalize_teams
from nflsim.model.players import player_ratings_as_of, role_means_as_of

_KEYS = ["season", "week", "team"]


def _qb_devs(
    player_games: pl.DataFrame, rows: pl.DataFrame, config: PlayerRatingsConfig
) -> pl.DataFrame:
    """``rows`` (season, week, gsis_id, ...) with ``dev``: dropback rating minus the league
    dropback average as of that week (null when unrated)."""
    targets = (
        rows.select("season", "week", "gsis_id")
        .unique()
        .with_columns(role=pl.lit("dropback"), group=pl.lit("QB"))
    )
    rated = player_ratings_as_of(player_games, targets, config).select(
        "season", "week", "gsis_id", "rating"
    )
    means = role_means_as_of(player_games, targets).filter(pl.col("role") == "dropback")
    return (
        rows.join(rated, on=["season", "week", "gsis_id"], how="left")
        .join(means.select("season", "week", "role_mean"), on=["season", "week"], how="left")
        .with_columns(dev=pl.col("rating") - pl.col("role_mean"))
        .drop("rating", "role_mean")
    )


def qb_baseline(
    player_games: pl.DataFrame, targets: pl.DataFrame, config: PlayerRatingsConfig
) -> pl.DataFrame:
    """``season, week, team, qb_base, qb_base_source`` for each team-week in ``targets``.

    ``qb_base`` is the dropback-weighted ``dev`` (as of week W) of the team's QBs in its
    games this season before W (``qb_base_source = "season"``), else in all of last
    season (``"prior_season"``); 0 with neither (``"none"``).
    """
    targets = targets.select(*_KEYS).unique().cast({"season": pl.Int32, "week": pl.Int32})
    dropbacks = (
        player_games.filter((pl.col("role") == "dropback") & (pl.col("group") == "QB"))
        .group_by("season", "week", "team", "gsis_id")
        .agg(pl.col("n").sum())
        .cast({"season": pl.Int32, "week": pl.Int32})
    )
    current = (
        targets.join(dropbacks.rename({"week": "game_week"}), on=["season", "team"], how="inner")
        .filter(pl.col("game_week") < pl.col("week"))
        .group_by(*_KEYS, "gsis_id")
        .agg(pl.col("n").sum())
        .with_columns(source=pl.lit("season"))
    )
    missing = targets.join(current.select(*_KEYS).unique(), on=_KEYS, how="anti")
    previous = (
        missing.join(
            dropbacks.with_columns(pl.col("season") + 1).drop("week"),
            on=["season", "team"],
            how="inner",
        )
        .group_by(*_KEYS, "gsis_id")
        .agg(pl.col("n").sum())
        .with_columns(source=pl.lit("prior_season"))
    )
    shares = pl.concat([current, previous]).with_columns(
        share=pl.col("n") / pl.col("n").sum().over(_KEYS)
    )
    rated = _qb_devs(player_games, shares, config)
    base = rated.group_by(*_KEYS).agg(
        (pl.col("share") * pl.col("dev").fill_null(0.0)).sum().alias("qb_base"),
        pl.col("source").first().alias("qb_base_source"),
    )
    return targets.join(base, on=_KEYS, how="left").with_columns(
        pl.col("qb_base").fill_null(0.0), pl.col("qb_base_source").fill_null("none")
    )


def listed_starters(schedules: pl.DataFrame) -> pl.DataFrame:
    """``season, week, team, gsis_id``: the starting QB nflverse lists for each team-game."""
    sched = normalize_teams(schedules, ["home_team", "away_team"])
    sides = [
        sched.select(
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col(f"{side}_team").alias("team"),
            pl.col(f"{side}_qb_id").alias("gsis_id"),
        )
        for side in ("home", "away")
        if f"{side}_qb_id" in sched.columns
    ]
    if not sides:
        return pl.DataFrame(
            schema={"season": pl.Int32, "week": pl.Int32, "team": pl.String, "gsis_id": pl.String}
        )
    return pl.concat(sides).filter(pl.col("gsis_id").is_not_null()).unique(_KEYS, keep="first")


def qb_change_features(
    player_games: pl.DataFrame,
    schedules: pl.DataFrame,
    targets: pl.DataFrame,
    off_qb: pl.DataFrame,
    config: PlayerRatingsConfig,
) -> pl.DataFrame:
    """``season, week, team, qb_change, qb_change_proj, qb_starter_listed, qb_base_source``.

    ``off_qb`` is a team-week frame with the projected QB value ``off_qb`` (from
    ``team_player_features`` or the injury-aware features).
    """
    base = qb_baseline(player_games, targets, config)
    starters = _qb_devs(player_games, listed_starters(schedules), config).rename(
        {"dev": "starter_dev"}
    )
    proj = off_qb.select(*_KEYS, "off_qb").cast({"season": pl.Int32, "week": pl.Int32})
    frame = (
        base.join(proj, on=_KEYS, how="left")
        .join(starters.select(*_KEYS, "starter_dev"), on=_KEYS, how="left")
        .with_columns(
            qb_change_proj=pl.col("off_qb").fill_null(0.0) - pl.col("qb_base"),
            qb_starter_listed=pl.col("starter_dev").is_not_null(),
        )
        .with_columns(
            qb_change=pl.when(pl.col("qb_starter_listed"))
            .then(pl.col("starter_dev") - pl.col("qb_base"))
            .otherwise(pl.col("qb_change_proj"))
        )
    )
    return frame.select(
        *_KEYS, "qb_change", "qb_change_proj", "qb_starter_listed", "qb_base_source"
    )


__all__ = ["listed_starters", "qb_baseline", "qb_change_features"]
