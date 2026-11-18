"""Game-level points model: team ratings -> expected points for each side.

Each team's expected points in a game are

    league_ppg + intercept + sum(coef_f * feature_f) + home * home_sign

where ``league_ppg`` is league points per team-game as of the game (previous season pooled
with this season through week W-1), the features are rating deviations from league average
(see ``ModelConfig``), and ``home_sign`` is +1 for the home team, -1 for the away team and
0 at neutral sites. Coefficients are fit by least squares; the negative-binomial dispersion
of actual points around the fit is estimated by the method of moments.

With ``margin_fit = "direct"`` the expected margin comes from its own least-squares fit on
home-minus-away features instead (see ``PointsModel``); the total is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
import polars as pl

from nflsim.config import (
    PLAYER_FEATURES,
    QB_CHANGE_FEATURES,
    ContextConfig,
    MarketConfig,
    ModelConfig,
    RatingsConfig,
)
from nflsim.data.teams import normalize_teams
from nflsim.model.market import market_ratings
from nflsim.model.ratings import RATING_COLUMNS, ratings_as_of

FloatArray = npt.NDArray[np.float64]

_RATING_FIELDS = [*RATING_COLUMNS, "league_epa", "league_success", "league_pace"]


def league_points_as_of(schedules: pl.DataFrame) -> pl.DataFrame:
    """League points per team-game for every (season, week) in ``schedules``.

    Pools the previous season with this season's completed games before week W, so the
    value for week W never includes week-W scores. Null when neither exists.
    """
    keys = schedules.select("season", "week").unique().sort("week")
    played = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    per_week = (
        played.group_by("season", "week")
        .agg(
            (pl.col("home_score") + pl.col("away_score")).sum().cast(pl.Float64).alias("pts"),
            (pl.len() * 2).cast(pl.Float64).alias("tg"),
        )
        .sort("season", "week")
        .with_columns(pl.col("pts", "tg").cum_sum().over("season"))
        .sort("week")
    )
    prev = (
        played.group_by("season")
        .agg(
            (pl.col("home_score") + pl.col("away_score")).sum().cast(pl.Float64).alias("prev_pts"),
            (pl.len() * 2).cast(pl.Float64).alias("prev_tg"),
        )
        .with_columns(pl.col("season") + 1)
    )
    return (
        keys.join_asof(
            per_week,
            on="week",
            by="season",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .join(prev, on="season", how="left")
        .with_columns(pl.col("pts", "tg", "prev_pts", "prev_tg").fill_null(0.0))
        .select(
            "season",
            "week",
            pl.when(pl.col("tg") + pl.col("prev_tg") > 0)
            .then((pl.col("pts") + pl.col("prev_pts")) / (pl.col("tg") + pl.col("prev_tg")))
            .otherwise(None)
            .alias("league_ppg"),
        )
    )


def build_games(
    schedules: pl.DataFrame,
    team_games: pl.DataFrame,
    ratings_config: RatingsConfig,
    player_features: pl.DataFrame | None = None,
    context: ContextConfig | None = None,
    market: MarketConfig | None = None,
) -> pl.DataFrame:
    """One row per scheduled game with both teams' pre-game ratings attached.

    Columns: game identifiers, scores (null if unplayed), Vegas spread/total/moneylines,
    ``home_sign``, ``league_ppg``, ``h_*``/``a_*`` ratings and prior flags, and
    ``has_ratings`` (False when any rating is missing, e.g. week 1 of the first cached
    season). ``player_features`` (from ``team_player_features``, keyed by season, week,
    team) adds ``h_*``/``a_*`` player-based offense columns and ``player_proj_missing``
    (and ``injury_report`` when the features are injury-aware).
    A team with no row gets 0 (league average) for each and every role counted as
    missing; this does not affect ``has_ratings``.

    Context columns (``context_columns``): ``h_rest``/``a_rest``, ``indoor``, ``wind``,
    ``rest_missing``, ``wind_missing``, ``roof`` and ``roof_inferred``.

    Market strength (``market_ratings``, earlier weeks' closing spreads): ``h_mkt``/``a_mkt``
    with ``*_mkt_games`` (lines this season before the week) and ``*_mkt_prior_games``.
    """
    sched = normalize_teams(schedules, ["home_team", "away_team"]).cast(
        {"season": pl.Int32, "week": pl.Int32}
    )
    targets = pl.concat(
        [
            sched.select("season", "week", pl.col(c).alias("team"))
            for c in ("home_team", "away_team")
        ]
    )
    ratings = ratings_as_of(team_games, targets, ratings_config).select(
        "season", "week", "team", *_RATING_FIELDS, "games_played", "prior_source"
    )
    if player_features is not None:
        extra = [
            c
            for c in ("injury_report", *QB_CHANGE_FEATURES, "qb_starter_listed", "qb_starter_id")
            if c in player_features.columns
        ]
        ratings = ratings.join(
            player_features.select(
                "season", "week", "team", *PLAYER_FEATURES, "player_proj_missing", *extra
            ).cast({"season": pl.Int32, "week": pl.Int32}),
            on=["season", "week", "team"],
            how="left",
        ).with_columns(
            pl.col(PLAYER_FEATURES).fill_null(0.0),
            *[pl.col(c).fill_null(0.0) for c in QB_CHANGE_FEATURES if c in extra],
            pl.col("player_proj_missing").fill_null(len(PLAYER_FEATURES)),
        )

    def side(prefix: str) -> pl.DataFrame:
        return ratings.rename(
            {c: f"{prefix}{c}" for c in ratings.columns if c not in ("season", "week")}
        )

    games = (
        sched.select(
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            pl.col("home_score").cast(pl.Float64),
            pl.col("away_score").cast(pl.Float64),
            pl.col("spread_line").cast(pl.Float64),
            pl.col("total_line").cast(pl.Float64),
            pl.col("home_moneyline").cast(pl.Float64),
            pl.col("away_moneyline").cast(pl.Float64),
            pl.when(pl.col("location") == "Neutral").then(0.0).otherwise(1.0).alias("home_sign"),
        )
        .join(
            side("h_"),
            left_on=["season", "week", "home_team"],
            right_on=["season", "week", "h_team"],
            how="left",
        )
        .join(
            side("a_"),
            left_on=["season", "week", "away_team"],
            right_on=["season", "week", "a_team"],
            how="left",
        )
        .join(league_points_as_of(sched), on=["season", "week"], how="left")
        .join(context_columns(sched, context or ContextConfig()), on="game_id", how="left")
    )
    mkt_cols = ["mkt", "mkt_games", "mkt_prior_games"]
    mkt = market_ratings(sched, market or MarketConfig()).select(
        "season", "week", "team", *mkt_cols
    )
    for prefix, team in (("h_", "home_team"), ("a_", "away_team")):
        games = games.join(
            mkt.rename({"team": team, **{c: f"{prefix}{c}" for c in mkt_cols}}),
            on=["season", "week", team],
            how="left",
        )
    needed = [f"{p}{c}" for p in ("h_", "a_") for c in _RATING_FIELDS] + ["league_ppg"]
    return games.with_columns(
        pl.all_horizontal(pl.col(needed).is_not_null()).alias("has_ratings")
    ).sort("season", "week", "game_id")


def _schedule_col(sched: pl.DataFrame, name: str, dtype: type[pl.DataType]) -> pl.Expr:
    """A schedule column, or nulls when the schedule does not have it."""
    if name in sched.columns:
        return pl.col(name).cast(dtype)
    return pl.lit(None, dtype)


def context_columns(sched: pl.DataFrame, config: ContextConfig) -> pl.DataFrame:
    """Per game: rest relative to normal for each side, indoor flag and outdoor wind.

    ``h_rest``/``a_rest``: days of rest minus ``normal_rest_days``, clipped to
    ``+/- rest_cap_days``; 0 when unknown (``rest_missing``). ``indoor``: 1 for dome or
    closed roof. ``wind``: mph at kickoff outdoors (capped at ``wind_cap_mph``), 0 indoors;
    missing outdoor wind uses
    the mean outdoor wind of earlier seasons (of all known games if there are none) and
    sets ``wind_missing``.
    """
    cap, normal = config.rest_cap_days, config.normal_rest_days
    roof = _schedule_col(sched, "roof", pl.String)
    frame = sched.select(
        "game_id",
        "season",
        *[
            ((_schedule_col(sched, f"{side}_rest", pl.Float64) - normal).clip(-cap, cap)).alias(
                f"{p}rest"
            )
            for side, p in (("home", "h_"), ("away", "a_"))
        ],
        roof.alias("roof"),
        _schedule_col(sched, "roof_inferred", pl.Boolean).fill_null(False).alias("roof_inferred"),
        roof.is_in(["dome", "closed"]).fill_null(False).alias("_indoor"),
        _schedule_col(sched, "wind", pl.Float64)
        .clip(upper_bound=config.wind_cap_mph)
        .alias("_wind"),
    )
    outdoor_known = frame.filter(~pl.col("_indoor") & pl.col("_wind").is_not_null())
    overall = outdoor_known["_wind"].mean()
    by_season = (
        outdoor_known.group_by("season")
        .agg(pl.col("_wind").sum().alias("s"), pl.len().alias("n"))
        .sort("season")
        .with_columns(
            (pl.col("s").cum_sum().shift(1) / pl.col("n").cum_sum().shift(1)).alias("_fill")
        )
        .select("season", "_fill")
    )
    missing_wind = ~pl.col("_indoor") & pl.col("_wind").is_null()
    return (
        frame.join(by_season, on="season", how="left")
        .with_columns(
            pl.col("_fill").fill_null(pl.lit(overall, pl.Float64)),
            (pl.col("h_rest").is_null() | pl.col("a_rest").is_null()).alias("rest_missing"),
        )
        .select(
            "game_id",
            pl.col("h_rest").fill_null(0.0),
            pl.col("a_rest").fill_null(0.0),
            pl.col("_indoor").cast(pl.Float64).alias("indoor"),
            pl.when(pl.col("_indoor"))
            .then(0.0)
            .otherwise(pl.coalesce("_wind", "_fill", pl.lit(0.0)))
            .alias("wind"),
            "rest_missing",
            missing_wind.alias("wind_missing"),
            "roof",
            "roof_inferred",
        )
    )


def _features(offense: str, defense: str) -> dict[str, pl.Expr]:
    """Feature expressions for the team with prefix ``offense`` scoring on ``defense``."""

    def dev(prefix: str, rating: str, league: str) -> pl.Expr:
        return pl.col(f"{prefix}{rating}") - pl.col(f"h_league_{league}")

    return {
        "off_epa": dev(offense, "off_epa", "epa"),
        "off_success": dev(offense, "off_success", "success"),
        "off_pace": dev(offense, "pace", "pace"),
        "def_epa": dev(defense, "def_epa", "epa"),
        "def_success": dev(defense, "def_success", "success"),
        "def_pace": dev(defense, "pace", "pace"),
        # Player-based ratings are already deviations from the league role average.
        **{f: pl.col(f"{offense}{f}") for f in (*PLAYER_FEATURES, *QB_CHANGE_FEATURES)},
        # Context (Phase 6): own and opponent rest; indoor and wind are the same for both.
        "rest": pl.col(f"{offense}rest"),
        "opp_rest": pl.col(f"{defense}rest"),
        "indoor": pl.col("indoor"),
        "wind": pl.col("wind"),
        # Market strength (opt-in): the scoring team's and the opponent's.
        "mkt": pl.col(f"{offense}mkt"),
        "opp_mkt": pl.col(f"{defense}mkt"),
    }


def design_matrices(games: pl.DataFrame, features: Sequence[str]) -> tuple[FloatArray, FloatArray]:
    """Feature matrices (games x features) for the home side and the away side.

    Missing ratings become 0 (league average); callers flag those rows via ``has_ratings``.
    """
    if not features:
        empty = np.zeros((games.height, 0))
        return empty, empty.copy()
    needs_players = (*PLAYER_FEATURES, *QB_CHANGE_FEATURES)
    missing = [f for f in features if f in needs_players and f"h_{f}" not in games.columns]
    if missing:
        raise ValueError(f"player features {missing} need build_games(player_features=...)")
    out = []
    for offense, defense in (("h_", "a_"), ("a_", "h_")):
        exprs = _features(offense, defense)
        frame = games.select([exprs[f].alias(f) for f in features]).fill_null(0.0)
        out.append(frame.to_numpy().astype(np.float64))
    return out[0], out[1]


@dataclass(frozen=True)
class PointsModel:
    """Fitted coefficients plus the score dispersion. ``home`` is per team (margin = 2x).

    If ``margin_coefs`` is set (``margin_fit = "direct"``), the expected margin comes from
    that separate fit, ``(x_home - x_away) . margin_coefs + margin_home * home_sign``. The
    expected total still comes from the points model, and each side gets
    ``(total +/- margin) / 2``.
    """

    features: tuple[str, ...]
    intercept: float
    coefs: tuple[float, ...]
    home: float
    dispersion: float
    min_expected_points: float
    train_games: int
    margin_coefs: tuple[float, ...] | None = None
    margin_home: float = 0.0

    def expected_points(
        self, games: pl.DataFrame, runs: Mapping[str, FloatArray] | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """Expected (home, away) points for each row of a ``build_games`` frame.

        ``runs`` optionally maps ``h_<feature>``/``a_<feature>`` columns to per-run values
        of shape ``(games, n)`` (e.g. sampled player availability). Those replace the
        column in each run and the result has shape ``(games, n)``; features not in
        ``runs`` keep their single value.
        """
        league = games["league_ppg"].cast(pl.Float64).to_numpy()
        if np.isnan(league).any() or games["league_ppg"].null_count():
            raise ValueError("league_ppg is missing for some games; no prior data exists")
        x_home, x_away = design_matrices(games, self.features)
        coefs = np.asarray(self.coefs, dtype=np.float64)
        sign = games["home_sign"].to_numpy().astype(np.float64)
        base = league + self.intercept
        used = {k: v for k, v in (runs or {}).items() if k[2:] in self.features}
        if used:
            x_home = _with_runs(x_home, self.features, used, "h_")
            x_away = _with_runs(x_away, self.features, used, "a_")
            base, sign = base[:, None], sign[:, None]
        mu_home = base + x_home @ coefs + self.home * sign
        mu_away = base + x_away @ coefs - self.home * sign
        if self.margin_coefs is not None:
            margin_coefs = np.asarray(self.margin_coefs, dtype=np.float64)
            margin = (x_home - x_away) @ margin_coefs + self.margin_home * sign
            total = mu_home + mu_away
            mu_home, mu_away = (total + margin) / 2, (total - margin) / 2
        floor = self.min_expected_points
        return np.maximum(mu_home, floor), np.maximum(mu_away, floor)


def _with_runs(
    x: FloatArray, features: Sequence[str], runs: Mapping[str, FloatArray], prefix: str
) -> FloatArray:
    """Design matrix (games x features) expanded to (games x runs x features), with the
    per-run values of ``runs`` columns for this side."""
    n = next(iter(runs.values())).shape[1]
    out = np.repeat(x[:, None, :], n, axis=1)
    for j, feature in enumerate(features):
        values = runs.get(f"{prefix}{feature}")
        if values is not None:
            if values.shape != out.shape[:2]:
                raise ValueError(f"{prefix}{feature} runs must have shape {out.shape[:2]}")
            out[:, :, j] = values
    return out


def _home_columns(train: pl.DataFrame, config: ModelConfig) -> tuple[FloatArray, FloatArray]:
    """Home-field regressors for the training games and the weights that combine their
    coefficients into one home value.

    By default a single ``home_sign`` column (one constant home value). With
    ``home_exclude_seasons`` or ``home_half_life_seasons`` each training season gets its
    own home column, and the home value is the weighted mean of the season coefficients:
    excluded seasons weigh 0, and with a half-life a season ``k`` seasons before the
    latest training season weighs ``0.5 ** (k / half_life)``.
    """
    sign = train["home_sign"].to_numpy().astype(np.float64)
    if not config.home_exclude_seasons and config.home_half_life_seasons is None:
        return sign[:, None], np.ones(1)
    seasons = train["season"].to_numpy()
    unique = np.unique(seasons)
    cols = np.column_stack([sign * (seasons == s) for s in unique])
    age = unique.max() - unique
    half_life = config.home_half_life_seasons
    weights = np.ones(unique.size) if half_life is None else 0.5 ** (age / half_life)
    weights[np.isin(unique, config.home_exclude_seasons)] = 0.0
    if weights.sum() == 0:
        raise ValueError("home_exclude_seasons leaves no training season for home field")
    return cols, weights / weights.sum()


def fit_points_model(games: pl.DataFrame, config: ModelConfig) -> PointsModel:
    """Least-squares fit of points on rating features, plus negative-binomial dispersion.

    Uses only rows with ratings and final scores. Each game contributes two rows: one per
    team. Raises ``ValueError`` if there are too few games or no overdispersion.
    """
    train = games.filter(
        pl.col("has_ratings")
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    features = tuple(config.features)
    if train.height < 10 * (len(features) + 2):
        raise ValueError(f"only {train.height} games to fit {len(features) + 2} coefficients")

    x_home, x_away = design_matrices(train, features)
    home_cols, home_weights = _home_columns(train, config)
    league = train["league_ppg"].to_numpy().astype(np.float64)
    ones = np.ones(train.height)
    x = np.vstack(
        [
            np.column_stack([ones, x_home, home_cols]),
            np.column_stack([ones, x_away, -home_cols]),
        ]
    )
    n_feat = len(features)
    points = np.concatenate(
        [train["home_score"].to_numpy(), train["away_score"].to_numpy()]
    ).astype(np.float64)
    offset = np.concatenate([league, league])
    beta = _least_squares(x, points - offset, range(1, 1 + n_feat), config.ridge)

    model = PointsModel(
        features=features,
        intercept=float(beta[0]),
        coefs=tuple(float(b) for b in beta[1 : 1 + n_feat]),
        home=float(beta[1 + n_feat :] @ home_weights),
        dispersion=1.0,  # placeholder until the final means are known
        min_expected_points=config.min_expected_points,
        train_games=train.height,
    )
    if config.margin_fit == "direct":
        # No intercept: swapping home and away must flip the sign of the margin.
        xm = np.column_stack([x_home - x_away, home_cols])
        margin = (train["home_score"] - train["away_score"]).to_numpy().astype(np.float64)
        gamma = _least_squares(xm, margin, range(n_feat), config.ridge)
        model = replace(
            model,
            margin_coefs=tuple(float(g) for g in gamma[:n_feat]),
            margin_home=float(gamma[n_feat:] @ home_weights),
        )

    # Dispersion of actual points around the model's final expected points.
    mu_home, mu_away = model.expected_points(train)
    mu = np.concatenate([mu_home, mu_away])
    excess = float(np.mean((points - mu) ** 2) - np.mean(mu))
    if excess <= 0:
        raise ValueError("points are not overdispersed; a negative binomial does not apply")
    return replace(model, dispersion=float(np.mean(mu**2) / excess))


def _least_squares(x: FloatArray, y: FloatArray, penalized: range, ridge: float) -> FloatArray:
    """Least squares of ``y`` on ``x``; with ``ridge`` > 0 the ``penalized`` columns get a
    ridge penalty ``ridge * rows * sd_j^2 * b_j^2`` (i.e. on standardized features)."""
    if ridge <= 0 or len(penalized) == 0:
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return np.asarray(beta, dtype=np.float64)
    cols = list(penalized)
    sd = x[:, cols].std(axis=0)
    extra = np.zeros((len(cols), x.shape[1]))
    extra[np.arange(len(cols)), cols] = np.sqrt(ridge * x.shape[0]) * sd
    beta, *_ = np.linalg.lstsq(
        np.vstack([x, extra]), np.concatenate([y, np.zeros(len(cols))]), rcond=None
    )
    return np.asarray(beta, dtype=np.float64)
