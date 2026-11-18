"""How often players on the injury report actually play (Phase 3, step 1).

For every team-game with snap counts, the population is every player on that team's
final injury report plus every regular who is not on it. A regular had at least
``regular_min_share`` of the offensive or defensive snaps in the team's previous game,
which is known before kickoff. A player "played" if he had any snap (offense, defense or
special teams) in the game, for any team.

The previous game is the team's previous game with snap counts, which for week 1 is
the last game of the season before; players new to the team count as non-regulars then.
"""

from __future__ import annotations

import polars as pl

from nflsim.config import InjuriesConfig
from nflsim.model.availability import previous_games, regulars_as_of, team_usage
from nflsim.model.injuries import UNLISTED, attach_play_probability

_KEY = ["season", "week", "team"]


def team_games(snaps: pl.DataFrame) -> pl.DataFrame:
    """Team-games with snap counts and the team's previous one (``prev_season/_week``)."""
    return previous_games(snaps, snaps.select(*_KEY).unique()).sort("team", "season", "week")


def report_outcomes(
    reports: pl.DataFrame,
    snaps: pl.DataFrame,
    regular_min_share: float,
    rosters: pl.DataFrame | None = None,
    unavailable_statuses: tuple[str, ...] = (),
) -> pl.DataFrame:
    """One row per player in the population with ``regular``, ``status`` and ``played``.

    ``reports`` comes from ``nflsim.data.injury_reports`` and ``snaps`` from
    ``nflsim.data.player_snaps``. Report rows for team-weeks without snap counts are
    dropped. Unlisted regulars get ``status = "unlisted"`` and ``practice = "none"``.

    With ``rosters`` (from ``nflsim.data.roster_weeks``), players whose weekly roster
    status that week is in ``unavailable_statuses`` (injured reserve, cut, suspended...)
    are left out: that is known before kickoff and already rules them out of projected
    usage, so the report adds nothing for them.
    """
    games = team_games(snaps)
    usage = team_usage(snaps)
    regulars = regulars_as_of(snaps, games, regular_min_share).with_columns(regular=pl.lit(True))
    listed = reports.join(games.select(_KEY), on=_KEY, how="semi").select(
        *_KEY, "gsis_id", "name", "position", "status", "practice"
    )
    unlisted = regulars.join(listed, on=[*_KEY, "gsis_id"], how="anti").select(
        *_KEY,
        "gsis_id",
        pl.lit(None, pl.String).alias("name"),
        pl.col("prev_position").alias("position"),
        pl.lit(UNLISTED).alias("status"),
        pl.lit("none").alias("practice"),
    )
    played = usage.group_by("season", "week", "gsis_id").agg(
        (pl.col("snaps").sum() > 0).alias("played")
    )
    population = pl.concat([listed, unlisted])
    if rosters is not None and unavailable_statuses:
        ruled_out = rosters.filter(pl.col("status").is_in(list(unavailable_statuses))).select(
            "season", "week", "gsis_id"
        )
        population = population.join(ruled_out, on=["season", "week", "gsis_id"], how="anti")
    return (
        population.join(
            regulars.select(*_KEY, "gsis_id", "regular"), on=[*_KEY, "gsis_id"], how="left"
        )
        .join(played, on=["season", "week", "gsis_id"], how="left")
        .with_columns(pl.col("regular", "played").fill_null(False))
        .sort(*_KEY, "gsis_id")
    )


def coverage(reports: pl.DataFrame, snaps: pl.DataFrame) -> pl.DataFrame:
    """Per season: team-games with snap counts, how many have no injury report rows, and
    report rows dropped because their team-week has no snap counts."""
    games = team_games(snaps).select(_KEY)
    reported = reports.select(_KEY).unique()
    no_report = games.join(reported, on=_KEY, how="anti").group_by("season").len("no_report")
    orphan = reports.join(games, on=_KEY, how="anti").group_by("season").len("orphan_rows")
    return (
        games.group_by("season")
        .len("team_games")
        .join(no_report, on="season", how="left")
        .join(orphan, on="season", how="left")
        .with_columns(pl.col("no_report", "orphan_rows").fill_null(0))
        .sort("season")
    )


def depth_coverage(depth: pl.DataFrame, snaps: pl.DataFrame) -> pl.DataFrame:
    """Per season: share of team-games with a depth chart, and of listed starters (depth 1
    on offense or defense) the share who played, and who led their slot group in snaps.

    A sanity check on the charts: stale or wrong charts show up as starters who did not
    play while healthy. ``qb_starter_led`` is the share of games where the depth-1 QB
    took the most offensive snaps among the team's QBs (when exactly one is listed).
    """
    games = team_games(snaps).select(_KEY)
    usage = team_usage(snaps)
    charted = depth.select(_KEY).unique().with_columns(charted=pl.lit(True))
    starters = depth.filter(
        (pl.col("depth") == 1) & pl.col("side").is_in(["offense", "defense"])
    ).join(games, on=_KEY, how="semi")
    played = starters.join(usage, on=[*_KEY, "gsis_id"], how="left").with_columns(
        played=pl.col("snaps").fill_null(0) > 0
    )
    qb_usage = (
        snaps.filter(pl.col("id_matched") & (pl.col("snap_position") == "QB"))
        .sort("offense_snaps", descending=True)
        .group_by(_KEY)
        .agg(pl.col("gsis_id").first().alias("top_qb"))
    )
    qb = (
        starters.filter(pl.col("slot") == "QB")
        .filter(pl.len().over(_KEY) == 1)
        .join(qb_usage, on=_KEY, how="inner")
        .group_by("season")
        .agg((pl.col("gsis_id") == pl.col("top_qb")).mean().alias("qb_starter_led"))
    )
    return (
        games.join(charted, on=_KEY, how="left")
        .group_by("season")
        .agg(pl.len().alias("team_games"), pl.col("charted").is_not_null().mean())
        .join(
            played.group_by("season").agg(pl.col("played").mean().alias("starters_played")),
            on="season",
            how="left",
        )
        .join(qb, on="season", how="left")
        .sort("season")
    )


def rate_table(outcomes: pl.DataFrame, config: InjuriesConfig) -> pl.DataFrame:
    """Observed play rate per (regular, status, practice) next to the configured one
    (averaged over the cell, since position adjustments vary within it)."""
    return (
        attach_play_probability(outcomes, config)
        .group_by("regular", "status", "practice")
        .agg(
            pl.len().alias("players"),
            pl.col("played").mean().alias("observed"),
            pl.col("play_prob").mean().alias("config"),
        )
        .with_columns(diff=pl.col("observed") - pl.col("config"))
        .sort(
            pl.col("regular"),
            pl.col("status").replace_strict(
                ["out", "doubtful", "questionable", "none", UNLISTED], [0, 1, 2, 3, 4]
            ),
            "practice",
            descending=[True, False, False],
        )
    )


def position_table(
    outcomes: pl.DataFrame, config: InjuriesConfig, status: str = "questionable"
) -> pl.DataFrame:
    """Observed vs. configured play rate by position for regulars with ``status``."""
    return (
        attach_play_probability(
            outcomes.filter(pl.col("regular") & (pl.col("status") == status)), config
        )
        .group_by("position")
        .agg(
            pl.len().alias("players"),
            pl.col("played").mean().alias("observed"),
            pl.col("play_prob").mean().alias("config"),
        )
        .with_columns(diff=pl.col("observed") - pl.col("config"))
        .sort("players", descending=True)
    )


def brier_scores(outcomes: pl.DataFrame, config: InjuriesConfig) -> pl.DataFrame:
    """Brier score of three ways to predict ``played``, per group of players.

    ``config``: the configured probabilities. ``cell_rate``: each cell's observed rate in
    these same seasons (an in-sample floor; ``config`` cannot beat it on the seasons it
    was measured on). ``status_rate``: the observed rate by status alone, ignoring
    practice and regular, as a coarser benchmark.
    """
    played = pl.col("played").cast(pl.Float64)
    frame = attach_play_probability(outcomes, config).with_columns(
        cell_rate=played.mean().over("regular", "status", "practice"),
        status_rate=played.mean().over("status"),
        group=pl.when(pl.col("status") == UNLISTED)
        .then(pl.lit("unlisted regulars"))
        .otherwise(pl.lit("on report")),
    )
    methods = ("config", "cell_rate", "status_rate")
    frame = frame.rename({"play_prob": "config"})

    def summarize(f: pl.DataFrame, label: str) -> pl.DataFrame:
        return f.select(
            pl.lit(label).alias("group"),
            pl.len().alias("players"),
            played.mean().alias("played"),
            *[((pl.col(m) - played) ** 2).mean().alias(f"brier_{m}") for m in methods],
        )

    parts = [
        summarize(frame.filter(pl.col("group") == g), g) for g in ("on report", "unlisted regulars")
    ]
    return pl.concat([*parts, summarize(frame, "all")])
