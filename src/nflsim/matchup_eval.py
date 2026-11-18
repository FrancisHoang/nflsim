"""Checks for the Phase 4 matchup data and models (``nflsim matchup-eval``)."""

from __future__ import annotations

import polars as pl

from nflsim.model.matchups import METRIC_TOTALS


def coverage(dropbacks: pl.DataFrame, pfr: pl.DataFrame | None) -> pl.DataFrame:
    """Per season: how much participation data exists for dropbacks and what it says.

    ``dropbacks`` comes from ``nflsim.data.participation.dropbacks`` and ``pfr`` from
    ``pfr_team_pressure``. Columns:

    - ``dropbacks``, ``matched``: dropbacks and share with a participation record.
    - ``sacks_flagged``: share of sacks with a raw pressure flag (0 before 2023, 1 after:
      the definition change that ``pressure`` corrects by counting sacks as pressure).
    - ``pressure_known``, ``pressure_rate``: share of non-scramble dropbacks with a known
      ``pressure``, and the pressured share of those (scrambles are left out because they
      are mostly unflagged before 2023).
    - ``scramble_known``: share of scrambles with a known pressure flag.
    - ``epa_pressured``, ``epa_clean``: EPA per non-scramble dropback with and without
      pressure.
    - ``rushers_known``, ``man_zone_known``, ``man_share``: pass rusher count and man/zone
      coverage availability, and the share of known coverage that was man.
    - ``pfr_ratio``, ``pfr_corr``: participation pressures (non-scramble) per PFR pressure over the
      team-games in both sources, and the correlation of the two counts across team-games
      (null without PFR data, i.e. before 2018).
    """
    known = pl.col("pressure").is_not_null()
    pocket = ~pl.col("scramble")
    pressure = pl.col("pressure").filter(pocket)
    per_season = dropbacks.group_by("season").agg(
        pl.len().alias("dropbacks"),
        pl.col("matched").mean().alias("matched"),
        pl.col("pressure_flag").filter(pl.col("sack")).is_not_null().mean().alias("sacks_flagged"),
        pressure.is_not_null().mean().alias("pressure_known"),
        pressure.cast(pl.Float64).mean().alias("pressure_rate"),
        pl.col("pressure").filter(pl.col("scramble")).is_not_null().mean().alias("scramble_known"),
        # Null pressure is dropped by both filters.
        pl.col("epa").filter(pocket & pl.col("pressure")).mean().alias("epa_pressured"),
        pl.col("epa").filter(pocket & ~pl.col("pressure")).mean().alias("epa_clean"),
        pl.col("pass_rushers").is_not_null().mean().alias("rushers_known"),
        pl.col("man_zone").is_not_null().mean().alias("man_zone_known"),
        (pl.col("man_zone") == "man").cast(pl.Float64).mean().alias("man_share"),
    )
    if pfr is None:
        return per_season.with_columns(
            pfr_ratio=pl.lit(None, pl.Float64), pfr_corr=pl.lit(None, pl.Float64)
        ).sort("season")
    games = (
        dropbacks.filter(known & pocket)
        .group_by("game_id", "season", "offense")
        .agg(pl.col("pressure").cast(pl.Float64).sum().alias("pressures"))
        .join(
            pfr.select("game_id", "offense", "pfr_pressures"),
            on=["game_id", "offense"],
            how="inner",
        )
        .group_by("season")
        .agg(
            (pl.col("pressures").sum() / pl.col("pfr_pressures").sum()).alias("pfr_ratio"),
            pl.corr("pressures", "pfr_pressures").alias("pfr_corr"),
        )
    )
    return per_season.join(games, on="season", how="left").sort("season")


def rating_skill(stats: pl.DataFrame, ratings: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """Out-of-sample skill of each matchup rating at predicting the next game.

    For each team-game in ``seasons`` and each side and metric, the target is the game's
    deviation from the league average (``league_<metric>`` as of that week) and the
    prediction is the rating as of that week (``matchup_ratings_as_of``). Errors are
    weighted by the game's count (dropbacks, pressured or clean dropbacks).
    ``skill = 1 - mse / mse_league``, where ``mse_league`` predicts 0 (league average);
    higher is better, and 0 means no better than the league average. ``skill`` is null
    when there are no games or the league average has no error.
    """
    games = stats.filter(pl.col("season").is_in(seasons)).join(
        ratings, on=["season", "week", "team"], how="inner"
    )
    rows = []
    for side in ("off", "def"):
        sided = games.filter(pl.col("side") == side)
        for metric, (total, count) in METRIC_TOTALS.items():
            frame = sided.filter(pl.col(count) > 0).select(
                (pl.col(total) / pl.col(count) - pl.col(f"league_{metric}")).alias("actual"),
                pl.col(f"{side}_{metric}").alias("pred"),
                pl.col(count).alias("w"),
            )
            frame = frame.drop_nulls()
            w = pl.col("w")
            mse, mse0 = frame.select(
                (((pl.col("actual") - pl.col("pred")) ** 2 * w).sum() / w.sum()).alias("mse"),
                ((pl.col("actual") ** 2 * w).sum() / w.sum()).alias("mse0"),
            ).row(0)
            rows.append(
                {
                    "side": side,
                    "metric": metric,
                    "games": frame.height,
                    "mse": mse,
                    "mse_league": mse0,
                    # Undefined without games, or when the league average is exact.
                    "skill": 1 - mse / mse0 if mse is not None and mse0 else None,
                }
            )
    return pl.DataFrame(rows)
