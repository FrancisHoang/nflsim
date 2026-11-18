"""Dropbacks with pressure and coverage from participation data (Phase 4).

``dropbacks`` joins each QB dropback in play-by-play with the nflverse participation
record for the play: whether the QB was pressured, the number of pass rushers, man or
zone coverage, the coverage shell and defenders in the box. Unknown values are null
(participation marks some unknowns with empty strings).

The raw pressure flag (``pressure_flag``) changes definition in 2023: before, sacks have
no flag at all; from 2023 every sack is flagged. ``pressure`` therefore counts every sack
as a pressure in all seasons. Scrambles are mostly unflagged before 2023, so ratings that
must be comparable across seasons should leave them out (``scramble``).

``pfr_team_pressure`` sums PFR's per-QB pressure counts to one row per team-game. It is a
second, independent source used to check the participation pressure flag, whose
definition changes between seasons.

Teams use current codes (``nflsim.data.normalize_teams``).
"""

from __future__ import annotations

import polars as pl

from nflsim.data.teams import normalize_teams

_MAN_ZONE = {"MAN_COVERAGE": "man", "ZONE_COVERAGE": "zone"}

DROPBACK_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "game_type",
    "offense",
    "defense",
    "passer_id",
    "epa",
    "sack",
    "qb_hit",
    "scramble",
    "matched",
    "pressure_flag",
    "pressure",
    "pass_rushers",
    "box",
    "man_zone",
    "coverage",
]


def _blank_to_null(col: str) -> pl.Expr:
    value = pl.col(col).cast(pl.String).str.strip_chars()
    return pl.when(value == "").then(None).otherwise(value)


def dropbacks(pbp: pl.LazyFrame, participation: pl.DataFrame) -> pl.DataFrame:
    """One row per QB dropback (sacks and scrambles included, two-point tries excluded).

    Columns: ``DROPBACK_COLUMNS``. ``matched`` is True when the play has a participation
    record. ``pressure_flag`` is the raw participation flag and ``pressure`` the same with
    every sack counted as pressure. ``pressure``, ``pass_rushers``, ``box``, ``man_zone``
    (``man``/``zone``) and ``coverage`` (e.g. ``COVER_3``) are null when unknown.
    """
    plays = (
        pbp.filter(
            (pl.col("qb_dropback") == 1)
            & (pl.col("two_point_attempt").fill_null(0) != 1)
            & pl.col("epa").is_not_null()
            & pl.col("posteam").is_not_null()
        )
        .select(
            "game_id",
            pl.col("play_id").cast(pl.Float64),
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("season_type").alias("game_type"),
            pl.col("posteam").alias("offense"),
            pl.col("defteam").alias("defense"),
            pl.col("passer_player_id").alias("passer_id"),
            "epa",
            pl.col("sack").fill_null(0).cast(pl.Boolean),
            pl.col("qb_hit").fill_null(0).cast(pl.Boolean),
            pl.col("qb_scramble").fill_null(0).cast(pl.Boolean).alias("scramble"),
        )
        .collect()
    )
    part = (
        participation.select(
            pl.col("nflverse_game_id").alias("game_id"),
            pl.col("play_id").cast(pl.Float64),
            pl.col("was_pressure").alias("pressure_flag"),
            pl.col("number_of_pass_rushers").cast(pl.Int32).alias("pass_rushers"),
            pl.col("defenders_in_box").cast(pl.Int32).alias("box"),
            _blank_to_null("defense_man_zone_type")
            .replace_strict(_MAN_ZONE, default=None)
            .alias("man_zone"),
            _blank_to_null("defense_coverage_type").alias("coverage"),
        )
        .unique(["game_id", "play_id"], keep="first")
        .with_columns(matched=pl.lit(True))
    )
    frame = plays.join(part, on=["game_id", "play_id"], how="left").with_columns(
        pl.col("matched").fill_null(False),
        pl.when(pl.col("sack")).then(True).otherwise(pl.col("pressure_flag")).alias("pressure"),
    )
    return normalize_teams(frame, ["offense", "defense"]).select(DROPBACK_COLUMNS)


def pfr_team_pressure(pfr_pass: pl.DataFrame) -> pl.DataFrame:
    """PFR pressures allowed per team-game: ``game_id, season, week, offense, defense,
    pfr_pressures, pfr_sacks`` (summed over the team's passers)."""
    frame = normalize_teams(pfr_pass, ["team", "opponent"])
    return (
        frame.group_by("game_id", "season", "week", "team", "opponent")
        .agg(
            pl.col("times_pressured").sum().alias("pfr_pressures"),
            pl.col("times_sacked").sum().alias("pfr_sacks"),
        )
        .rename({"team": "offense", "opponent": "defense"})
        .with_columns(pl.col("season", "week").cast(pl.Int32))
        .sort("season", "week", "game_id", "offense")
    )
