"""Player identity: one ID (GSIS) across play-by-play, rosters and snap counts.

Play-by-play and weekly rosters key players by GSIS ID (``00-00xxxxx``). Snap counts
come from Pro Football Reference and key players by PFR ID, so they are mapped through
the ``players`` crosswalk. Rows that cannot be mapped are kept and flagged, never
dropped silently.

Positions come from the weekly roster, because a player's position can change between
seasons (QB to WR, DL to LB). The roster positions are already grouped (QB, RB, WR, TE,
OL, DL, LB, DB, K, P, LS).
"""

from __future__ import annotations

import polars as pl

from nflsim.data.teams import normalize_teams

# When a player is listed by two teams in the same week (e.g. claimed off another team's
# reserve list), the row with the earliest status here wins.
_STATUS_ORDER = ["ACT", "INA", "RES", "PUP", "SUS", "DEV", "NWT", "CUT", "RET"]

# Columns of play-by-play that name a player, and the role each one plays.
PBP_PLAYER_COLUMNS = {
    "passer_player_id": "passer",
    "rusher_player_id": "rusher",
    "receiver_player_id": "receiver",
}


def pfr_crosswalk(players: pl.DataFrame) -> pl.DataFrame:
    """Map PFR player IDs to GSIS IDs from the ``players`` table.

    Returns columns ``pfr_id`` and ``gsis_id``. Raises ``ValueError`` if either ID maps
    to more than one of the other, since a join would then duplicate rows.
    """
    xwalk = players.select("pfr_id", "gsis_id").drop_nulls().unique()
    for col in ("pfr_id", "gsis_id"):
        dupes = xwalk.filter(pl.len().over(col) > 1)[col].unique().sort().to_list()
        if dupes:
            raise ValueError(f"{col} maps to more than one player: {dupes[:5]}")
    return xwalk


def roster_weeks(rosters_weekly: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """One row per (season, week, player) with team and position.

    Columns: ``season``, ``week``, ``gsis_id``, ``name``, ``team`` (current abbreviation),
    ``position``, ``position_source`` ("roster", or "players" when the roster has no
    position and the crosswalk's position group is used), and ``status``.
    Rows without a GSIS ID cannot be linked to anything and are dropped.
    """
    status_rank = pl.col("status").replace_strict(
        _STATUS_ORDER, list(range(len(_STATUS_ORDER))), default=len(_STATUS_ORDER)
    )
    fallback = players.select(
        "gsis_id", pl.col("position_group").alias("fallback_position")
    ).drop_nulls("gsis_id")
    frame = (
        normalize_teams(rosters_weekly.filter(pl.col("gsis_id").is_not_null()), ["team"])
        .with_columns(_rank=status_rank)
        .sort("season", "week", "gsis_id", "_rank", "team")
        .unique(["season", "week", "gsis_id"], keep="first", maintain_order=True)
        .join(fallback, on="gsis_id", how="left")
    )
    return frame.select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "gsis_id",
        pl.col("full_name").alias("name"),
        "team",
        pl.coalesce("position", "fallback_position").alias("position"),
        pl.when(pl.col("position").is_not_null())
        .then(pl.lit("roster"))
        .when(pl.col("fallback_position").is_not_null())
        .then(pl.lit("players"))
        .otherwise(None)
        .alias("position_source"),
        "status",
    )


def player_snaps(snap_counts: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Snap counts keyed by GSIS ID, with current team abbreviations.

    Adds ``gsis_id`` (null when the PFR ID is not in the crosswalk) and ``id_matched``.
    ``snap_position`` is PFR's position label, kept for reference; use ``roster_weeks``
    for the position used in modelling.
    """
    xwalk = pfr_crosswalk(players)
    frame = normalize_teams(snap_counts, ["team", "opponent"]).join(
        xwalk, left_on="pfr_player_id", right_on="pfr_id", how="left"
    )
    return frame.select(
        "game_id",
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "game_type",
        "team",
        "opponent",
        "pfr_player_id",
        "gsis_id",
        pl.col("gsis_id").is_not_null().alias("id_matched"),
        pl.col("player").alias("name"),
        pl.col("position").alias("snap_position"),
        *[
            pl.col(c).cast(pl.Float64)
            for c in (
                "offense_snaps",
                "offense_pct",
                "defense_snaps",
                "defense_pct",
                "st_snaps",
                "st_pct",
            )
        ],
    )


def id_coverage(snaps: pl.DataFrame, pbp_ids: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    """Per-season share of snaps and play-by-play player IDs that link to a known player.

    ``snaps`` comes from ``player_snaps``; ``pbp_ids`` has ``season`` plus the columns in
    ``PBP_PLAYER_COLUMNS``; ``rosters`` comes from ``roster_weeks``. A play-by-play ID
    counts as linked when the player is on a weekly roster that season with a position.
    Shares are weighted by snaps and by plays, so they measure how much of the game is
    covered rather than how many fringe players are.
    """
    snap_cov = snaps.group_by("season").agg(
        pl.len().alias("snap_rows"),
        pl.col("id_matched").mean().alias("snap_rows_matched"),
        (
            pl.col("offense_snaps").filter(pl.col("id_matched")).sum()
            / pl.col("offense_snaps").sum()
        ).alias("off_snaps_matched"),
        (
            pl.col("defense_snaps").filter(pl.col("id_matched")).sum()
            / pl.col("defense_snaps").sum()
        ).alias("def_snaps_matched"),
    )
    known = (
        rosters.filter(pl.col("position").is_not_null())
        .select("season", "gsis_id")
        .unique()
        .with_columns(_known=pl.lit(True))
    )
    out = snap_cov
    for col, role in PBP_PLAYER_COLUMNS.items():
        ids = pbp_ids.select("season", pl.col(col).alias("gsis_id")).drop_nulls()
        share = (
            ids.join(known, on=["season", "gsis_id"], how="left")
            .group_by("season")
            .agg(pl.col("_known").is_not_null().mean().alias(f"{role}_plays_linked"))
        )
        out = out.join(share, on="season", how="full", coalesce=True)
    return out.sort("season")


def unmatched_snaps(snaps: pl.DataFrame, top: int) -> pl.DataFrame:
    """The ``top`` players with the most snaps whose PFR ID has no GSIS match."""
    return (
        snaps.filter(~pl.col("id_matched"))
        .sort("season", "week")
        .group_by("pfr_player_id")
        .agg(
            pl.col("name").last(),
            pl.col("snap_position").unique(maintain_order=True).str.join("/"),
            pl.col("offense_snaps").sum(),
            pl.col("defense_snaps").sum(),
            pl.col("season").min().alias("first_season"),
            pl.col("season").max().alias("last_season"),
        )
        .with_columns(total=pl.col("offense_snaps") + pl.col("defense_snaps"))
        .sort("total", "pfr_player_id", descending=[True, False])
        .head(top)
        .drop("total")
    )
