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

from nflsim.config import PLAYER_FEATURES, ModelConfig, RatingsConfig
from nflsim.data.teams import normalize_teams
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
        extra = [c for c in ("injury_report",) if c in player_features.columns]
        ratings = ratings.join(
            player_features.select(
                "season", "week", "team", *PLAYER_FEATURES, "player_proj_missing", *extra
            ).cast({"season": pl.Int32, "week": pl.Int32}),
            on=["season", "week", "team"],
            how="left",
        ).with_columns(
            pl.col(PLAYER_FEATURES).fill_null(0.0),
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
    )
    needed = [f"{p}{c}" for p in ("h_", "a_") for c in _RATING_FIELDS] + ["league_ppg"]
    return games.with_columns(
        pl.all_horizontal(pl.col(needed).is_not_null()).alias("has_ratings")
    ).sort("season", "week", "game_id")


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
        **{f: pl.col(f"{offense}{f}") for f in PLAYER_FEATURES},
    }


def design_matrices(games: pl.DataFrame, features: Sequence[str]) -> tuple[FloatArray, FloatArray]:
    """Feature matrices (games x features) for the home side and the away side.

    Missing ratings become 0 (league average); callers flag those rows via ``has_ratings``.
    """
    if not features:
        empty = np.zeros((games.height, 0))
        return empty, empty.copy()
    missing = [f for f in features if f in PLAYER_FEATURES and f"h_{f}" not in games.columns]
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
    sign = train["home_sign"].to_numpy().astype(np.float64)
    league = train["league_ppg"].to_numpy().astype(np.float64)
    ones = np.ones(train.height)
    x = np.vstack(
        [
            np.column_stack([ones, x_home, sign]),
            np.column_stack([ones, x_away, -sign]),
        ]
    )
    points = np.concatenate(
        [train["home_score"].to_numpy(), train["away_score"].to_numpy()]
    ).astype(np.float64)
    offset = np.concatenate([league, league])
    beta, *_ = np.linalg.lstsq(x, points - offset, rcond=None)

    model = PointsModel(
        features=features,
        intercept=float(beta[0]),
        coefs=tuple(float(b) for b in beta[1:-1]),
        home=float(beta[-1]),
        dispersion=1.0,  # placeholder until the final means are known
        min_expected_points=config.min_expected_points,
        train_games=train.height,
    )
    if config.margin_fit == "direct":
        # No intercept: swapping home and away must flip the sign of the margin.
        xm = np.column_stack([x_home - x_away, sign])
        margin = (train["home_score"] - train["away_score"]).to_numpy().astype(np.float64)
        gamma, *_ = np.linalg.lstsq(xm, margin, rcond=None)
        model = replace(
            model,
            margin_coefs=tuple(float(g) for g in gamma[:-1]),
            margin_home=float(gamma[-1]),
        )

    # Dispersion of actual points around the model's final expected points.
    mu_home, mu_away = model.expected_points(train)
    mu = np.concatenate([mu_home, mu_away])
    excess = float(np.mean((points - mu) ** 2) - np.mean(mu))
    if excess <= 0:
        raise ValueError("points are not overdispersed; a negative binomial does not apply")
    return replace(model, dispersion=float(np.mean(mu**2) / excess))
