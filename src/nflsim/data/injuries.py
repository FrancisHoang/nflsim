"""Weekly injury reports and depth charts, keyed by GSIS ID and current team codes.

Injury reports: one row per player per team-week with the final game status (``out``,
``doubtful``, ``questionable``, or ``none`` when the player is on the practice report
without a game status) and the last practice participation (``dnp``, ``limited``,
``full``, or ``none`` when not recorded).

Depth charts come in two formats. Through 2024 nflverse publishes one chart per team-week.
From 2025 it publishes daily snapshots with timestamps and no week; for those, each team's
chart for a game is the last snapshot taken before kickoff, so no post-kickoff
information leaks in.
"""

from __future__ import annotations

import polars as pl

from nflsim.data.teams import normalize_teams

_STATUS_MAP = {"Out": "out", "Doubtful": "doubtful", "Questionable": "questionable"}
_PRACTICE_PREFIX = {"Did Not": "dnp", "Limited": "limited", "Full": "full"}

# Kickoff times in schedules are US Eastern.
_SCHEDULE_TZ = "America/New_York"


def injury_reports(injuries: pl.DataFrame) -> pl.DataFrame:
    """Final injury report per (season, week, team, player).

    Columns: ``season``, ``week``, ``game_type``, ``team``, ``gsis_id``, ``name``,
    ``position``, ``status`` and ``practice`` (see ``nflsim.config.REPORT_STATUSES`` and
    ``PRACTICES``)
    and ``injury`` (primary injury from the game report, else from the practice report).
    Statuses outside the known set (e.g. "Note") map to ``none``. When a player has
    several rows for the same team-week, the most recently modified one wins.
    """
    status = pl.col("report_status")
    practice = pl.col("practice_status").fill_null("")
    practice_code: pl.Expr = pl.lit("none")
    for prefix, code in _PRACTICE_PREFIX.items():
        practice_code = (
            pl.when(practice.str.starts_with(prefix)).then(pl.lit(code)).otherwise(practice_code)
        )
    frame = (
        normalize_teams(injuries.filter(pl.col("gsis_id").is_not_null()), ["team"])
        .with_columns(pl.col("season", "week").cast(pl.Int32))
        .sort("season", "week", "team", "gsis_id", "date_modified", nulls_last=False)
        .unique(["season", "week", "team", "gsis_id"], keep="last", maintain_order=True)
    )
    return frame.select(
        "season",
        "week",
        "game_type",
        "team",
        "gsis_id",
        pl.col("full_name").alias("name"),
        "position",
        status.replace_strict(_STATUS_MAP, default="none").alias("status"),
        practice_code.alias("practice"),
        pl.coalesce("report_primary_injury", "practice_primary_injury").alias("injury"),
    )


def kickoffs(schedules: pl.DataFrame) -> pl.DataFrame:
    """Kickoff time (UTC) for each team's game: ``season``, ``week``, ``team``, ``kickoff``."""
    games = normalize_teams(schedules, ["home_team", "away_team"]).with_columns(
        kickoff=pl.concat_str("gameday", pl.lit(" "), pl.col("gametime").fill_null("13:00"))
        .str.to_datetime("%Y-%m-%d %H:%M", time_unit="us")
        .dt.replace_time_zone(_SCHEDULE_TZ)
        .dt.convert_time_zone("UTC")
    )
    return pl.concat(
        [
            games.select(
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
                pl.col(side).alias("team"),
                "kickoff",
            )
            for side in ("home_team", "away_team")
        ]
    )


_DEPTH_COLUMNS = ["season", "week", "team", "gsis_id", "name", "side", "slot", "depth"]


def _weekly_depth(frame: pl.DataFrame) -> pl.DataFrame:
    """Charts in the pre-2025 format: one per team-week, ``depth_team`` is the rank."""
    side = (
        pl.col("formation")
        .replace_strict(
            {"Offense": "offense", "Defense": "defense", "Special Teams": "special"},
            default=None,
        )
        .alias("side")
    )
    slot = pl.col("depth_position").str.strip_chars()
    blank = slot.is_null() | (slot == "")
    # When both formats are stacked, the snapshot format's ``team`` column is all null here.
    weekly = frame.filter(pl.col("gsis_id").is_not_null()).drop("team", strict=False)
    return normalize_teams(weekly.rename({"club_code": "team"}), ["team"]).select(
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
        "team",
        "gsis_id",
        pl.col("full_name").alias("name"),
        side,
        pl.when(blank).then(pl.col("position")).otherwise(slot).alias("slot"),
        pl.col("depth_team").cast(pl.Int32, strict=False).alias("depth"),
    )


def _snapshot_depth(frame: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Charts in the 2025+ format: the last snapshot before each team's kickoff."""
    snaps = normalize_teams(frame.filter(pl.col("gsis_id").is_not_null()), ["team"]).with_columns(
        taken=pl.col("dt").str.to_datetime(time_zone="UTC", time_unit="us")
    )
    taken = snaps.select("team", "taken").unique().sort("taken")
    chosen = (
        games.sort("kickoff")
        .join_asof(
            taken,
            left_on="kickoff",
            right_on="taken",
            by="team",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,  # sorted above; polars cannot verify within groups
        )
        .filter(pl.col("taken").is_not_null())
    )
    group = pl.col("pos_grp")
    side = (
        pl.when(group == "Special Teams")
        .then(pl.lit("special"))
        .when(group.str.ends_with(" D"))
        .then(pl.lit("defense"))
        .otherwise(pl.lit("offense"))
    )
    return (
        chosen.join(snaps, on=["team", "taken"], how="inner")
        .select(
            "season",
            "week",
            "team",
            "gsis_id",
            pl.col("player_name").alias("name"),
            side.alias("side"),
            pl.col("pos_abb").alias("slot"),
            pl.col("pos_rank").cast(pl.Int32).alias("depth"),
        )
        # A player can appear in several formations (e.g. 3-4 and 4-3 groups); keep his
        # best rank per slot.
        .sort("depth")
        .unique(["season", "week", "team", "gsis_id", "side", "slot"], keep="first")
    )


def depth_chart_weeks(depth_charts: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """Depth chart per team-game in one format, whatever the source format.

    Columns: ``season``, ``week``, ``team``, ``gsis_id``, ``name``, ``side``
    ("offense", "defense" or "special"), ``slot`` (e.g. QB, RB, WR, LT, LCB, PK) and
    ``depth`` (1 = starter). ``schedules`` supplies kickoff times for snapshot charts.
    Rows without a GSIS ID are dropped.
    """
    parts = []
    if "club_code" in depth_charts.columns:
        parts.append(_weekly_depth(depth_charts.filter(pl.col("club_code").is_not_null())))
    if "dt" in depth_charts.columns:
        snapshot = depth_charts.filter(pl.col("dt").is_not_null())
        if not snapshot.is_empty():
            parts.append(_snapshot_depth(snapshot, kickoffs(schedules)))
    if not parts:
        raise ValueError("depth_charts has neither the weekly nor the snapshot format")
    out = pl.concat([p.select(_DEPTH_COLUMNS) for p in parts], how="vertical_relaxed")
    return out.sort("season", "week", "team", "side", "slot", "depth", "gsis_id")
