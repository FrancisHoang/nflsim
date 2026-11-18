"""Walk-forward backtest of the points model against simple and market baselines.

For each test season S the points model is fit on seasons S-k .. S-1 (k =
``model.train_seasons``) and then predicts every game of S whose teams have ratings.
Ratings are already as-of each game's week, so nothing from a game's own week or later
reaches its prediction. Three methods are scored on the same games:

* ``model``: the Monte Carlo simulation (mean margin and total, share of home wins).
* ``vegas``: closing spread and total; win probability from the moneylines with the
  bookmaker's margin removed, or, if a moneyline is missing, from the spread with a
  normal error whose spread is estimated on the training seasons.
* ``home``: home-field only, from training-season averages (neutral sites get 0 and 0.5).
* ``pbp_only``: when the model uses market strength (``mkt``, ``opp_mkt``), the same model
  without it, so play-by-play ratings are still measured against Vegas independently.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from nflsim.config import MARKET_FEATURES, Settings
from nflsim.distribution import beat_line_probability, crps_normal, crps_samples, margin_shares
from nflsim.model import PointsModel, SimResult, fit_points_model, simulate_scores
from nflsim.model.drive_engine import (
    build_library,
    overtime_rules,
    simulate_drives,
    theta_for_points,
)
from nflsim.pipeline import availability_runs

METHODS = ("model", "pbp_only", "vegas", "home")


def pbp_only_settings(settings: Settings) -> Settings | None:
    """``settings`` without the market features, or None when it uses none."""
    features = settings.model.features
    if not set(features) & set(MARKET_FEATURES):
        return None
    kept = tuple(f for f in features if f not in MARKET_FEATURES)
    return settings.model_copy(
        update={"model": settings.model.model_copy(update={"features": kept})}
    )


def american_to_prob(odds: pl.Expr) -> pl.Expr:
    """Break-even probability of American odds (includes the bookmaker's margin)."""
    return pl.when(odds < 0).then(-odds / (-odds + 100.0)).otherwise(100.0 / (odds + 100.0))


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return np.asarray([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x], dtype=float)


@dataclass(frozen=True)
class SeasonFit:
    season: int
    model: PointsModel
    train_seasons: tuple[int, ...]
    test_games: int
    skipped_games: int  # completed games without ratings (flagged, not predicted)


def walk_forward(
    games: pl.DataFrame,
    settings: Settings,
    test_seasons: Iterable[int],
    n: int,
    seed: int,
    lineup: pl.DataFrame | None = None,
    drive_table: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, list[SeasonFit]]:
    """Predict each test season from a model fit on earlier seasons only.

    With ``sim.engine = "drive"``, games are simulated drive by drive from
    ``drive_table`` (``nflsim.data.drives``), using only the training seasons' drives.

    Returns one row per predicted game (actuals plus each method's margin, total and home
    win probability) and the fitted model for each season. Each season's simulation uses
    seed ``seed + season`` so seasons are reproducible independently. With ``lineup``
    (injury-aware lineups) and ``sim.sample_availability``, who plays is drawn per run
    (``nflsim.pipeline.availability_runs``).
    """

    k = settings.model.train_seasons
    completed = pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    frames: list[pl.DataFrame] = []
    fits: list[SeasonFit] = []
    for season in sorted(set(test_seasons)):
        train = games.filter(pl.col("season").is_between(season - k, season - 1) & completed)
        season_games = games.filter((pl.col("season") == season) & completed)
        test = season_games.filter(pl.col("has_ratings"))
        if test.is_empty():
            raise ValueError(f"no completed games with ratings in {season}")
        if train.is_empty():
            raise ValueError(f"no training seasons before {season}")

        model = fit_points_model(train, settings.model)
        runs = availability_runs(test, model, lineup, settings, n, seed + season)
        mu_home, mu_away = model.expected_points(test, runs)
        if settings.sim.engine == "drive":
            if drive_table is None:
                raise ValueError("the drive engine needs drive_table (nflsim.data.drives)")
            sim = drive_simulation(
                drive_table,
                train,
                test,
                _run_mean(mu_home),
                _run_mean(mu_away),
                settings,
                n,
                seed + season,
            )
        else:
            sim = simulate_scores(
                mu_home, mu_away, model.dispersion, n, seed + season, settings.sim.overtime_points
            )
        # A share of n runs can be exactly 0 or 1; keep it within half a run of the edges.
        p_model = np.clip(sim.home_win_prob(), 0.5 / n, 1 - 0.5 / n)

        base = _baselines(test, train)
        actual_margin = base["actual_margin"].to_numpy()
        actual_total = base["actual_total"].to_numpy()
        spread = base["vegas_margin"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        total_line = base["vegas_total"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        keys = margin_shares(sim.margin, settings.backtest.key_margins)
        pure = _pbp_only(train, test, settings, n, seed + season, lineup)
        frames.append(
            base.with_columns(
                **pure,
                model_margin=pl.Series(sim.mean_margin()),
                model_total=pl.Series(sim.mean_total()),
                model_p=pl.Series(p_model),
                model_mu_home=pl.Series(_run_mean(mu_home)),
                model_mu_away=pl.Series(_run_mean(mu_away)),
                model_crps_margin=pl.Series(crps_samples(sim.margin, actual_margin)),
                model_crps_total=pl.Series(crps_samples(sim.total, actual_total)),
                model_p_cover=pl.Series(beat_line_probability(sim.margin, spread)),
                model_p_over=pl.Series(beat_line_probability(sim.total, total_line)),
                **{f"model_key_{k}": pl.Series(v) for k, v in keys.items()},
            )
        )
        fits.append(
            SeasonFit(
                season=season,
                model=model,
                train_seasons=tuple(sorted(train["season"].unique().to_list())),
                test_games=test.height,
                skipped_games=season_games.height - test.height,
            )
        )
    return pl.concat(frames, how="vertical"), fits


def _pbp_only(
    train: pl.DataFrame,
    test: pl.DataFrame,
    settings: Settings,
    n: int,
    seed: int,
    lineup: pl.DataFrame | None,
) -> dict[str, pl.Series]:
    """``pbp_only_*`` margin, total and win probability (score engine), or nothing when
    the model has no market features."""
    pure = pbp_only_settings(settings)
    if pure is None:
        return {}
    model = fit_points_model(train, pure.model)
    runs = availability_runs(test, model, lineup, pure, n, seed)
    mu_home, mu_away = model.expected_points(test, runs)
    sim = simulate_scores(mu_home, mu_away, model.dispersion, n, seed, pure.sim.overtime_points)
    return {
        "pbp_only_margin": pl.Series(sim.mean_margin()),
        "pbp_only_total": pl.Series(sim.mean_total()),
        "pbp_only_p": pl.Series(np.clip(sim.home_win_prob(), 0.5 / n, 1 - 0.5 / n)),
    }


def drive_simulation(
    drive_table: pl.DataFrame,
    train: pl.DataFrame,
    games: pl.DataFrame,
    mu_home: np.ndarray,
    mu_away: np.ndarray,
    settings: Settings,
    n: int,
    seed: int,
) -> SimResult:
    """Simulate ``games`` drive by drive, tilted so each team's expected points per drive
    are its expected points (``mu_*``) over the expected number of drives.

    The library uses the drives of ``train``'s seasons only. Expected drives per team =
    the training games' average points per team / the library's points per drive.
    """
    cfg = settings.drives
    seasons = sorted(train["season"].unique().to_list())
    library, weights = build_library(drive_table, seasons, cfg)
    per_drive = float(library.expected_points(np.zeros(1), weights)[0])
    team_points = float(
        train.select(((pl.col("home_score") + pl.col("away_score")) / 2).mean()).item()
    )
    drives_per_team = team_points / per_drive
    grid = np.linspace(cfg.theta_min, cfg.theta_max, cfg.theta_steps)
    overtime = [
        overtime_rules(int(s), g != "REG", cfg)
        for s, g in zip(games["season"].to_list(), games["game_type"].to_list(), strict=True)
    ]
    # Targets in points per drive, corrected by pilot runs so each team's simulated mean
    # matches its expected points (field position and overtime shift it otherwise).
    target_home, target_away = mu_home / drives_per_team, mu_away / drives_per_team
    pilot_rng = np.random.SeedSequence(seed).spawn(cfg.pilot_rounds)
    for child in pilot_rng:
        pilot = simulate_drives(
            library,
            theta_for_points(library, weights, target_home, grid),
            theta_for_points(library, weights, target_away, grid),
            overtime,
            cfg.pilot_runs,
            int(child.generate_state(1)[0]),
            cfg.max_drives,
        )
        target_home = target_home * _ratio(mu_home, pilot.home.mean(axis=1))
        target_away = target_away * _ratio(mu_away, pilot.away.mean(axis=1))
    theta_home = theta_for_points(library, weights, target_home, grid)
    theta_away = theta_for_points(library, weights, target_away, grid)
    return simulate_drives(library, theta_home, theta_away, overtime, n, seed, cfg.max_drives)


def _ratio(want: np.ndarray, got: np.ndarray) -> np.ndarray:
    """Correction factor want / got, kept within [0.5, 2] so one noisy pilot can't explode."""
    return np.asarray(np.clip(want / np.maximum(got, 1e-9), 0.5, 2.0), dtype=np.float64)


def _run_mean(mu: np.ndarray) -> np.ndarray:
    """Expected points per game, averaged over runs when they vary per run."""
    return np.asarray(mu.mean(axis=1) if mu.ndim == 2 else mu, dtype=np.float64)


def _baselines(test: pl.DataFrame, train: pl.DataFrame) -> pl.DataFrame:
    """Actual outcomes plus the Vegas and home-field-only predictions for ``test``."""
    margin = pl.col("home_score") - pl.col("away_score")
    outcome = pl.when(margin > 0).then(1.0).when(margin < 0).then(0.0).otherwise(0.5)

    at_home = train.filter(pl.col("home_sign") == 1.0)
    home_margin = float(at_home.select(margin.mean()).item())
    home_p = float(at_home.select(outcome.mean()).item())
    mean_total = float(train.select((pl.col("home_score") + pl.col("away_score")).mean()).item())
    spread_sd = float(train.select((margin - pl.col("spread_line")).std()).item())
    total = pl.col("home_score") + pl.col("away_score")
    total_sd = float(train.select((total - pl.col("total_line")).std()).item())

    p_home_ml = american_to_prob(pl.col("home_moneyline"))
    p_away_ml = american_to_prob(pl.col("away_moneyline"))
    has_ml = pl.col("home_moneyline").is_not_null() & pl.col("away_moneyline").is_not_null()
    spread_p = pl.Series(_normal_cdf(test["spread_line"].fill_null(0.0).to_numpy() / spread_sd))

    return test.select(
        "game_id",
        "season",
        "week",
        "game_type",
        "home_team",
        "away_team",
        margin.alias("actual_margin"),
        (pl.col("home_score") + pl.col("away_score")).alias("actual_total"),
        outcome.alias("outcome"),
        pl.col("spread_line").alias("vegas_margin"),
        pl.col("total_line").alias("vegas_total"),
        pl.when(has_ml)
        .then(p_home_ml / (p_home_ml + p_away_ml))
        .otherwise(spread_p)
        .alias("vegas_p"),
        pl.when(has_ml)
        .then(pl.lit("moneyline"))
        .otherwise(pl.lit("spread"))
        .alias("vegas_p_source"),
        (pl.col("home_sign") * home_margin).alias("home_margin"),
        pl.lit(mean_total).alias("home_total"),
        pl.when(pl.col("home_sign") == 1.0).then(home_p).otherwise(0.5).alias("home_p"),
    ).with_columns(
        # Vegas as a distribution, for CRPS: normal around the line, with the spread of
        # results around the line in the training seasons.
        pl.Series(
            "vegas_crps_margin",
            crps_normal(
                test["spread_line"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
                spread_sd,
                test.select(margin).to_series().to_numpy(),
            ),
        ).fill_nan(None),
        pl.Series(
            "vegas_crps_total",
            crps_normal(
                test["total_line"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
                total_sd,
                test.select(total).to_series().to_numpy(),
            ),
        ).fill_nan(None),
    )


def _metrics(frame: pl.DataFrame, method: str) -> dict[str, float]:
    p = pl.col(f"{method}_p")
    y = pl.col("outcome")
    row = frame.select(
        (pl.col(f"{method}_margin") - pl.col("actual_margin")).abs().mean().alias("margin_mae"),
        (pl.col(f"{method}_total") - pl.col("actual_total")).abs().mean().alias("total_mae"),
        ((p - y) ** 2).mean().alias("brier"),
        (-(y * p.log() + (1 - y) * (1 - p).log())).mean().alias("log_loss"),
        ((pl.col(f"{method}_margin") > 0) == (pl.col("actual_margin") > 0))
        .filter(pl.col("actual_margin") != 0)
        .mean()
        .alias("pick_rate"),
    ).row(0, named=True)
    return {key: float(value) for key, value in row.items()}


def summarize(predictions: pl.DataFrame) -> pl.DataFrame:
    """Metrics per season and overall for every method, on the same set of games.

    Games missing any Vegas number are dropped for all methods so the comparison is fair.
    """
    scored = predictions.drop_nulls(["vegas_margin", "vegas_total", "vegas_p"])
    rows = []
    groups: list[tuple[str, pl.DataFrame]] = [
        (str(s), scored.filter(pl.col("season") == s))
        for s in sorted(scored["season"].unique().to_list())
    ]
    groups.append(("all", scored))
    for label, frame in groups:
        for method in METHODS:
            if f"{method}_p" not in frame.columns:
                continue
            rows.append(
                {
                    "seasons": label,
                    "method": method,
                    "games": frame.height,
                    **_metrics(frame, method),
                }
            )
    return pl.DataFrame(rows)


def calibration(predictions: pl.DataFrame, method: str, bins: int) -> pl.DataFrame:
    """Predicted vs. actual home win rate in equal-width probability bins."""
    p = pl.col(f"{method}_p")
    index = (p * bins).floor().clip(0, bins - 1).cast(pl.Int64)
    return (
        predictions.group_by(index.alias("bin"))
        .agg(
            pl.len().alias("games"),
            p.mean().alias("mean_pred"),
            pl.col("outcome").mean().alias("actual"),
        )
        .with_columns(
            (pl.col("bin") / bins).alias("lo"),
            ((pl.col("bin") + 1) / bins).alias("hi"),
        )
        .sort("bin")
        .select("lo", "hi", "games", "mean_pred", "actual")
    )


def format_table(frame: pl.DataFrame, digits: Sequence[tuple[str, int]] = ()) -> str:
    """Plain-ASCII table (safe on any console encoding)."""
    places = dict(digits)
    header = frame.columns
    body = [
        [
            f"{v:.{places.get(c, 3)}f}" if isinstance(v, float) else str(v)
            for c, v in zip(header, row, strict=True)
        ]
        for row in frame.iter_rows()
    ]
    widths = [max([len(h)] + [len(r[i]) for r in body]) for i, h in enumerate(header)]
    lines = ["  ".join(h.rjust(w) for h, w in zip(header, widths, strict=True))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(v.rjust(w) for v, w in zip(r, widths, strict=True)) for r in body]
    return "\n".join(lines)
