"""Team offense and defense ratings from play-by-play, shrunk toward a prior.

Two steps:

1. ``team_game_stats`` turns raw play-by-play into one row per team per game, holding
   play counts and sums (EPA, successes) for the team's offense and defense.
2. ``ratings_as_of`` rates any (season, week, team) using only games from *earlier weeks
   of the same season*, plus the previous season as a prior. The cut-off is strict
   (week < W), so a week-W rating never sees week-W results.

Shrinkage: each per-play metric is ``(sum + k * prior) / (plays + k)``. The prior is the
league average plus ``carryover`` times the team's previous-season deviation from the
league average. So a team starts the season near a regressed version of last year and the
prior fades as plays accumulate. With no previous season available, the prior is the
league average alone (``prior_source == "league"``). With no data at all (week 1 of the
first cached season) ratings are null (``prior_source == "none"``).

Defense metrics are EPA and success *allowed*: lower is better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nflsim.config import RatingsConfig

KEYS = ["season", "week", "game_id", "team"]

# Per-unit sum columns in the team-game table, prefixed with "off_" or "def_".
_UNIT_SUMS = ["n", "epa", "success", "pass_n", "pass_epa", "rush_n", "rush_epa"]
# Pace columns (offense only, every scrimmage play, no win-probability filter).
_PACE_SUMS = ["plays", "games"]
STAT_COLUMNS = [f"{u}_{c}" for u in ("off", "def") for c in _UNIT_SUMS] + _PACE_SUMS


@dataclass(frozen=True)
class _Metric:
    name: str  # output name, e.g. "pass_epa"
    total: str  # sum column, e.g. "pass_epa"
    count: str  # count column, e.g. "pass_n"
    strength: str  # RatingsConfig attribute holding the prior strength


_METRICS = (
    _Metric("epa", "epa", "n", "prior_plays_epa"),
    _Metric("success", "success", "n", "prior_plays_success"),
    _Metric("pass_epa", "pass_epa", "pass_n", "prior_plays_pass_epa"),
    _Metric("rush_epa", "rush_epa", "rush_n", "prior_plays_rush_epa"),
)

RATING_COLUMNS = [f"{u}_{m.name}" for u in ("off", "def") for m in _METRICS] + ["pace"]


def scrimmage_plays(pbp: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Keep real pass and run plays with an EPA value.

    ``play_type`` already separates kneels, spikes, and penalty-only plays ("no_play"),
    so filtering to pass/run removes them. Two-point tries are dropped too.
    """
    return (
        pbp.lazy()
        .filter(
            pl.col("play_type").is_in(["pass", "run"]),
            pl.col("epa").is_not_null(),
            pl.col("posteam").is_not_null(),
            pl.col("defteam").is_not_null(),
            pl.col("two_point_attempt").fill_null(0) == 0,
        )
        .select(
            "season",
            "week",
            "game_id",
            "posteam",
            "defteam",
            pl.col("play_type") == "pass",
            pl.col("epa").cast(pl.Float64),
            pl.col("success").fill_null(0).cast(pl.Float64),
            pl.col("wp").cast(pl.Float64),
        )
        .rename({"play_type": "is_pass"})
    )


def _unit_sums(rated: pl.LazyFrame, team_col: str, prefix: str) -> pl.LazyFrame:
    is_pass = pl.col("is_pass")
    epa = pl.col("epa")
    return (
        rated.group_by("season", "week", "game_id", pl.col(team_col).alias("team"))
        .agg(
            pl.len().cast(pl.Float64).alias("n"),
            epa.sum().alias("epa"),
            pl.col("success").sum().alias("success"),
            is_pass.sum().cast(pl.Float64).alias("pass_n"),
            epa.filter(is_pass).sum().alias("pass_epa"),
            (~is_pass).sum().cast(pl.Float64).alias("rush_n"),
            epa.filter(~is_pass).sum().alias("rush_epa"),
        )
        .rename({c: f"{prefix}_{c}" for c in _UNIT_SUMS})
    )


def team_game_stats(pbp: pl.DataFrame | pl.LazyFrame, config: RatingsConfig) -> pl.DataFrame:
    """One row per team per game: offense and defense sums, plus offensive play count.

    Offense/defense sums use only plays inside the config's win-probability window. Pace
    (``plays``) counts every scrimmage play, since garbage time still uses up the clock.
    """
    plays = scrimmage_plays(pbp)
    pace = plays.group_by("season", "week", "game_id", pl.col("posteam").alias("team")).agg(
        pl.len().cast(pl.Float64).alias("plays"), pl.lit(1.0).alias("games")
    )
    rated = plays.filter(
        pl.col("wp").fill_null(0.5).is_between(config.min_wp, config.max_wp, closed="both")
    )
    offense = _unit_sums(rated, "posteam", "off")
    defense = _unit_sums(rated, "defteam", "def")
    return (
        pace.join(offense, on=KEYS, how="left")
        .join(defense, on=KEYS, how="left")
        .with_columns(pl.col(STAT_COLUMNS).fill_null(0.0))
        .select(*KEYS, *STAT_COLUMNS)
        .sort(KEYS)
        .collect()
    )


def _rate(total: pl.Expr, count: pl.Expr) -> pl.Expr:
    """``total / count``, or null when there is nothing to divide by."""
    return pl.when(count > 0).then(total / count).otherwise(None)


def ratings_as_of(
    team_games: pl.DataFrame, targets: pl.DataFrame, config: RatingsConfig
) -> pl.DataFrame:
    """Rate each (season, week, team) in ``targets`` from games before that week.

    ``team_games`` is the output of ``team_game_stats`` and may span many seasons; the
    previous season, if present, supplies the prior. Returns ``targets`` with
    ``RATING_COLUMNS`` plus ``league_epa``, ``league_success``, ``league_pace``,
    ``games_played`` (this season, before the target week) and ``prior_source``.
    """
    stats = STAT_COLUMNS
    targets = targets.select("season", "week", "team").unique().sort("week")

    # This season's totals through week W-1: cumulative sums, then a strict as-of join.
    team_cum = (
        team_games.sort("season", "team", "week")
        .select("season", "team", "week", pl.col(stats).cum_sum().over("season", "team"))
        .sort("week")
    )
    current = targets.join_asof(
        team_cum,
        on="week",
        by=["season", "team"],
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    ).with_columns(pl.col(stats).fill_null(0.0))

    # League totals use offense columns: every play is one team's offense, so the league's
    # offense and defense averages are identical.
    league_cols = [f"off_{c}" for c in _UNIT_SUMS] + _PACE_SUMS
    league_week = (
        team_games.group_by("season", "week")
        .agg(pl.col(league_cols).sum())
        .sort("season", "week")
        .with_columns(pl.col(league_cols).cum_sum().over("season"))
        .sort("week")
    )
    league_cur = (
        targets.select("season", "week")
        .unique()
        .sort("week")
        .join_asof(
            league_week,
            on="week",
            by="season",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .with_columns(pl.col(league_cols).fill_null(0.0))
        .rename({c: f"lgcur_{c}" for c in league_cols})
    )

    # Previous season, keyed to the season it serves as prior for.
    prev_team = (
        team_games.group_by("season", "team")
        .agg(pl.col(stats).sum())
        .with_columns(pl.col("season") + 1)
        .rename({c: f"prev_{c}" for c in stats})
    )
    prev_league = (
        team_games.group_by("season")
        .agg(pl.col(league_cols).sum())
        .with_columns(pl.col("season") + 1)
        .rename({c: f"lgprev_{c}" for c in league_cols})
    )

    frame = (
        current.join(league_cur, on=["season", "week"], how="left")
        .join(prev_team, on=["season", "team"], how="left")
        .join(prev_league, on="season", how="left")
        .with_columns(pl.col(r"^(prev|lgprev)_.*$").fill_null(0.0))
    )

    exprs: list[pl.Expr] = []
    for unit in ("off", "def"):
        for m in _METRICS:
            k = float(getattr(config, m.strength))
            exprs.append(
                _shrunk(
                    cur_total=pl.col(f"{unit}_{m.total}"),
                    cur_count=pl.col(f"{unit}_{m.count}"),
                    prev_total=pl.col(f"prev_{unit}_{m.total}"),
                    prev_count=pl.col(f"prev_{unit}_{m.count}"),
                    league_total=f"off_{m.total}",
                    league_count=f"off_{m.count}",
                    strength=k,
                    carryover=config.carryover,
                ).alias(f"{unit}_{m.name}")
            )
    exprs.append(
        _shrunk(
            cur_total=pl.col("plays"),
            cur_count=pl.col("games"),
            prev_total=pl.col("prev_plays"),
            prev_count=pl.col("prev_games"),
            league_total="plays",
            league_count="games",
            strength=config.prior_games_pace,
            carryover=config.carryover,
        ).alias("pace")
    )
    league_exprs = [
        _league_rate("off_epa", "off_n").alias("league_epa"),
        _league_rate("off_success", "off_n").alias("league_success"),
        _league_rate("plays", "games").alias("league_pace"),
    ]
    has_league = (pl.col("lgcur_off_n") + pl.col("lgprev_off_n")) > 0
    prior_source = (
        pl.when(pl.col("prev_games") > 0)
        .then(pl.lit("prior_season"))
        .when(has_league)
        .then(pl.lit("league"))
        .otherwise(pl.lit("none"))
    )
    rated = frame.with_columns(*exprs, *league_exprs)
    if config.opponent_adjust:
        rated = _with_opponent_adjustment(rated, team_games, targets, config)
    return (
        rated.with_columns(
            pl.col("games").cast(pl.Int64).alias("games_played"),
            prior_source.alias("prior_source"),
        )
        .select(
            "season",
            "week",
            "team",
            *RATING_COLUMNS,
            "league_epa",
            "league_success",
            "league_pace",
            "games_played",
            "prior_source",
        )
        .sort("season", "week", "team")
    )


def _league_rate(total: str, count: str) -> pl.Expr:
    """League average pooling last season with this season through week W-1."""
    return _rate(
        pl.col(f"lgcur_{total}") + pl.col(f"lgprev_{total}"),
        pl.col(f"lgcur_{count}") + pl.col(f"lgprev_{count}"),
    )


def _shrunk(
    *,
    cur_total: pl.Expr,
    cur_count: pl.Expr,
    prev_total: pl.Expr,
    prev_count: pl.Expr,
    league_total: str,
    league_count: str,
    strength: float,
    carryover: float,
) -> pl.Expr:
    """Empirical-Bayes rating: this season's data shrunk toward a regressed prior."""
    league = _league_rate(league_total, league_count)
    prev_league = _rate(pl.col(f"lgprev_{league_total}"), pl.col(f"lgprev_{league_count}"))
    prev_team = _rate(prev_total, prev_count)
    deviation = (prev_team - prev_league).fill_null(0.0)
    prior = league + carryover * deviation
    return (cur_total + strength * prior) / (cur_count + strength)


# Opponent adjustment (``RatingsConfig.opponent_adjust``) ---------------------------------

ADJUSTED_COLUMNS = [f"{u}_{m.name}" for u in ("off", "def") for m in _METRICS]


def _with_opponent_adjustment(
    rated: pl.DataFrame, team_games: pl.DataFrame, targets: pl.DataFrame, config: RatingsConfig
) -> pl.DataFrame:
    """Replace the per-play ratings with ``opponent_adjusted`` ones (nulls stay null)."""
    adj = opponent_adjusted(team_games, targets, config).rename(
        {c: f"_adj_{c}" for c in ADJUSTED_COLUMNS}
    )
    return (
        rated.cast({"season": pl.Int32, "week": pl.Int32})
        .join(adj, on=["season", "week", "team"], how="left")
        .with_columns(
            pl.when(pl.col(c).is_not_null()).then(pl.col(f"_adj_{c}")).alias(c)
            for c in ADJUSTED_COLUMNS
        )
        .drop([f"_adj_{c}" for c in ADJUSTED_COLUMNS])
    )


def opponent_adjusted(
    team_games: pl.DataFrame, targets: pl.DataFrame, config: RatingsConfig
) -> pl.DataFrame:
    """Opponent-adjusted offense and defense ratings for each (season, week, team).

    For each metric, a weighted ridge regression over the season's games before week W:
    each team-game's per-play value is ``league + o[offense] + d[defense]`` (weight =
    plays), with ``o`` and ``d`` shrunk toward ``carryover`` x the previous season's
    adjusted deviations, at the metric's prior strength ``k``. Ratings are ``league + o``
    and ``league + d`` (``league`` pools last season with this season so far, like the
    unadjusted ratings). If every opponent were average (all ``d = 0``), the offense
    rating would be exactly the unadjusted ``(sum + k * prior) / (plays + k)``.

    Returns ``season, week, team`` plus ``ADJUSTED_COLUMNS``; null without league data.
    """
    rows = _opponent_rows(team_games)
    targets = targets.select("season", "week", "team").unique()
    frames: list[pl.DataFrame] = []
    for season in sorted(targets["season"].unique().to_list()):
        season_targets = targets.filter(pl.col("season") == season)
        cur = rows.filter(pl.col("season") == season)
        prev = rows.filter(pl.col("season") == season - 1)
        teams = sorted(
            set(cur["team"].to_list())
            | set(prev["team"].to_list())
            | set(season_targets["team"].to_list())
        )
        index = {t: i for i, t in enumerate(teams)}
        weeks = sorted(season_targets["week"].unique().to_list())
        columns: dict[str, list[float | None]] = {}
        for m in _METRICS:
            k = float(getattr(config, m.strength))
            prev_o, prev_d = _season_deviations(prev, m, index, config.opponent_prior_ridge)
            off: list[float | None] = []
            dfn: list[float | None] = []
            for week in weeks:
                before = cur.filter(pl.col("week") < week)
                league = _pooled_league(before, prev, m)
                if league is None:
                    off.extend([None] * len(teams))
                    dfn.extend([None] * len(teams))
                    continue
                o, d = _joint_fit(
                    before,
                    m,
                    index,
                    league,
                    k,
                    config.carryover * prev_o,
                    config.carryover * prev_d,
                )
                off.extend((league + o).tolist())
                dfn.extend((league + d).tolist())
            columns[f"off_{m.name}"] = off
            columns[f"def_{m.name}"] = dfn
        frame = pl.DataFrame(
            {
                "season": [season] * (len(weeks) * len(teams)),
                "week": [w for w in weeks for _ in teams],
                "team": teams * len(weeks),
                **columns,
            },
            schema_overrides={"season": pl.Int32, "week": pl.Int32},
        )
        frames.append(
            season_targets.cast({"season": pl.Int32, "week": pl.Int32}).join(
                frame, on=["season", "week", "team"], how="left"
            )
        )
    return pl.concat(frames, how="vertical")


def _opponent_rows(team_games: pl.DataFrame) -> pl.DataFrame:
    """Team-game offense sums with the opposing team (the defense on those plays)."""
    sums = [f"off_{c}" for c in _UNIT_SUMS]
    own = team_games.select("season", "week", "game_id", "team", *sums)
    opponents = team_games.select("game_id", pl.col("team").alias("opp"))
    return (
        own.join(opponents, on="game_id", how="inner")
        .filter(pl.col("team") != pl.col("opp"))
        .cast({"season": pl.Int32, "week": pl.Int32})
    )


def _pooled_league(cur: pl.DataFrame, prev: pl.DataFrame, m: _Metric) -> float | None:
    """League per-play value of last season plus this season so far (``_league_rate``)."""
    total = float(cur[f"off_{m.total}"].sum()) + float(prev[f"off_{m.total}"].sum())
    count = float(cur[f"off_{m.count}"].sum()) + float(prev[f"off_{m.count}"].sum())
    return total / count if count > 0 else None


def _season_deviations(
    rows: pl.DataFrame, m: _Metric, index: dict[str, int], ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    """Adjusted offense/defense deviations over a whole season (zeros without data)."""
    zeros = np.zeros(len(index))
    count = float(rows[f"off_{m.count}"].sum()) if rows.height else 0.0
    if count <= 0:
        return zeros, zeros.copy()
    league = float(rows[f"off_{m.total}"].sum()) / count
    return _joint_fit(rows, m, index, league, ridge, zeros, zeros.copy())


def _joint_fit(
    rows: pl.DataFrame,
    m: _Metric,
    index: dict[str, int],
    league: float,
    strength: float,
    prior_o: np.ndarray,
    prior_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize ``sum n (rate - league - o_t - d_u)^2 + k |o - prior_o|^2 + k |d - prior_d|^2``.

    ``rows`` are team-games (offense ``team``, defense ``opp``); unknowns are offense
    deviations for every team, then defense deviations. Solved by the normal equations.
    """
    size = len(index)
    a = np.diag(np.full(2 * size, strength))
    b = np.concatenate([strength * prior_o, strength * prior_d])
    if rows.height:
        t = np.array([index[x] for x in rows["team"].to_list()], dtype=np.int64)
        u = np.array([index[x] for x in rows["opp"].to_list()], dtype=np.int64) + size
        n = rows[f"off_{m.count}"].to_numpy().astype(np.float64)
        excess = rows[f"off_{m.total}"].to_numpy().astype(np.float64) - n * league
        for i, j in ((t, t), (u, u), (t, u), (u, t)):
            np.add.at(a, (i, j), n)
        np.add.at(b, t, excess)
        np.add.at(b, u, excess)
    x = np.linalg.solve(a, b)
    return x[:size], x[size:]
