"""One row per drive from play-by-play (Phase 5, step 1).

nflverse numbers possessions with ``fixed_drive``. A kickoff belongs to the receiving
team's drive and an extra point to the scoring drive. A drive's points are the change in
each team's running score (``total_home_score``/``total_away_score``) since the previous
drive ended. Per-play ``posteam`` scores cannot be used: after a defensive touchdown the
extra point is credited to the other team inside the same drive. Start field position and clock come
from the first play that is not a kickoff; the drive's length from
``drive_time_of_possession`` (``M:SS``).

Teams use current codes (``nflsim.data.normalize_teams``).
"""

from __future__ import annotations

import polars as pl

from nflsim.data.teams import normalize_teams

DRIVE_RESULTS = (
    "Touchdown",
    "Field goal",
    "Missed field goal",
    "Punt",
    "Turnover",
    "Turnover on downs",
    "Safety",
    "Opp touchdown",
    "End of half",
)

DRIVE_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "drive",
    "offense",
    "defense",
    "half",
    "start_transition",
    "start_yardline",
    "start_seconds",
    "start_half_seconds",
    "start_diff",
    "result",
    "off_points",
    "def_points",
    "plays",
    "seconds",
]

_PBP_COLUMNS = [
    "game_id",
    "season",
    "week",
    "season_type",
    "play_id",
    "fixed_drive",
    "posteam",
    "defteam",
    "play_type",
    "yardline_100",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "game_half",
    "score_differential",
    "home_team",
    "total_home_score",
    "total_away_score",
    "fixed_drive_result",
    "drive_play_count",
    "drive_time_of_possession",
    "drive_start_transition",
]


def _clock_seconds(text: pl.Expr) -> pl.Expr:
    """``M:SS`` -> seconds (null when missing or malformed)."""
    parts = text.str.split(":")
    return (
        parts.list.get(0, null_on_oob=True).cast(pl.Int64, strict=False) * 60
        + parts.list.get(1, null_on_oob=True).cast(pl.Int64, strict=False)
    ).cast(pl.Float64)


def drives(pbp: pl.LazyFrame) -> pl.DataFrame:
    """One row per possession, with the columns in ``DRIVE_COLUMNS``.

    ``off_points``/``def_points`` are points scored by the offense and the defense during
    the drive (extra points and two-point tries included; defensive and return scores go
    to the defense). ``start_yardline`` is yards from the opponent's end zone,
    ``start_seconds``/``start_half_seconds`` the game and half clock, ``start_diff`` the
    offense's lead, all at the drive's first non-kickoff play. ``result`` is nflverse's
    ``fixed_drive_result``. Drives with no team on offense are dropped.
    """
    all_plays = (
        pbp.select(_PBP_COLUMNS)
        .filter(pl.col("fixed_drive").is_not_null())
        .collect()
        .sort("game_id", "play_id")
    )
    # Running score at the end of each drive, and its change since the previous drive.
    scores = (
        all_plays.group_by("game_id", "fixed_drive")
        .agg(
            pl.col("total_home_score").max().alias("_home"),
            pl.col("total_away_score").max().alias("_away"),
        )
        .sort("game_id", "fixed_drive")
        .with_columns(
            pl.col("_home", "_away").fill_null(strategy="forward").over("game_id").fill_null(0.0)
        )
        .with_columns(
            (pl.col("_home") - pl.col("_home").shift(1).over("game_id").fill_null(0.0)).alias(
                "_home_pts"
            ),
            (pl.col("_away") - pl.col("_away").shift(1).over("game_id").fill_null(0.0)).alias(
                "_away_pts"
            ),
        )
        .select("game_id", "fixed_drive", "_home_pts", "_away_pts")
    )
    plays = all_plays.filter(pl.col("posteam").is_not_null())
    scrimmage = pl.col("play_type").is_not_null() & (pl.col("play_type") != "kickoff")
    first = (
        plays.filter(scrimmage & pl.col("yardline_100").is_not_null())
        .group_by("game_id", "fixed_drive")
        .agg(
            pl.col("yardline_100").first().alias("start_yardline"),
            pl.col("game_seconds_remaining").first().alias("start_seconds"),
            pl.col("half_seconds_remaining").first().alias("start_half_seconds"),
            pl.col("score_differential").first().alias("start_diff"),
        )
    )
    per_drive = plays.group_by("game_id", "fixed_drive").agg(
        pl.col("season").first().cast(pl.Int32),
        pl.col("week").first().cast(pl.Int32),
        pl.col("season_type").first().alias("game_type"),
        pl.col("posteam").first().alias("offense"),
        pl.col("defteam").first().alias("defense"),
        pl.col("game_half").first().alias("half"),
        pl.col("drive_start_transition").first().alias("start_transition"),
        pl.col("fixed_drive_result").first().alias("result"),
        (pl.col("posteam").first() == pl.col("home_team").first()).alias("_off_home"),
        pl.col("drive_play_count").first().cast(pl.Float64).alias("plays"),
        _clock_seconds(pl.col("drive_time_of_possession").first()).alias("seconds"),
    )
    off_home = pl.col("_off_home")
    frame = (
        per_drive.join(first, on=["game_id", "fixed_drive"], how="left")
        .join(scores, on=["game_id", "fixed_drive"], how="left")
        .with_columns(
            pl.when(off_home).then("_home_pts").otherwise("_away_pts").alias("off_points"),
            pl.when(off_home).then("_away_pts").otherwise("_home_pts").alias("def_points"),
        )
        .rename({"fixed_drive": "drive"})
    )
    return (
        normalize_teams(frame, ["offense", "defense"])
        .select(DRIVE_COLUMNS)
        .sort("game_id", "drive")
    )


def drive_coverage(drive_frame: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """Per season: drives per game, result shares, averages, and whether drive points add
    up to the final score.

    ``score_match``: share of games where each team's drive points (own offense plus its
    defense's scores) equal its final score in ``schedules``.
    """
    d = drive_frame
    team_points = (
        pl.concat(
            [
                d.group_by("game_id", pl.col("offense").alias("team")).agg(
                    pl.col("off_points").sum().alias("p")
                ),
                d.group_by("game_id", pl.col("defense").alias("team")).agg(
                    pl.col("def_points").sum().alias("p")
                ),
            ]
        )
        .group_by("game_id", "team")
        .agg(pl.col("p").sum())
    )
    sched = normalize_teams(schedules, ["home_team", "away_team"]).filter(
        pl.col("home_score").is_not_null()
    )
    finals = pl.concat(
        [
            sched.select(
                "game_id",
                pl.col("home_team").alias("team"),
                pl.col("home_score").cast(pl.Float64).alias("final"),
            ),
            sched.select(
                "game_id",
                pl.col("away_team").alias("team"),
                pl.col("away_score").cast(pl.Float64).alias("final"),
            ),
        ]
    )
    matched = (
        finals.join(team_points, on=["game_id", "team"], how="inner")
        .group_by("game_id")
        .agg((pl.col("p") == pl.col("final")).all().alias("ok"))
        .join(d.select("game_id", "season").unique(), on="game_id")
        .group_by("season")
        .agg(pl.col("ok").mean().alias("score_match"))
    )
    result_shares = [
        (pl.col("result") == r).mean().alias(r.lower().replace(" ", "_")) for r in DRIVE_RESULTS
    ]
    per_season = d.group_by("season").agg(
        (pl.len() / pl.col("game_id").n_unique()).alias("drives_per_game"),
        pl.col("start_yardline").mean().alias("start_yardline"),
        pl.col("seconds").mean().alias("seconds"),
        (pl.col("off_points").sum() / pl.len()).alias("points_per_drive"),
        pl.col("start_yardline").is_null().mean().alias("no_start"),
        pl.col("result").is_null().mean().alias("no_result"),
        *result_shares,
    )
    return per_season.join(matched, on="season", how="left").sort("season")
