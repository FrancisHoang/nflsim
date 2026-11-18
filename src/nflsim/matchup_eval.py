"""Checks for the Phase 4 matchup data and models (``nflsim matchup-eval``)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nflsim.model.matchups import (
    LOG5,
    METRIC_TOTALS,
    FloatArray,
    PressureModel,
    expected_pass_epa,
    fit_pressure_model,
    pressure_terms,
)


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


def _game_bootstrap(
    games: pl.Series, diff: FloatArray, weight: FloatArray, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Weighted mean of ``diff`` and a 95% interval from resampling whole games."""
    frame = pl.DataFrame({"g": games, "d": diff * weight, "w": weight})
    # Sorted so the same seed resamples the same games (group_by order is arbitrary).
    per_game = frame.group_by("g").agg(pl.col("d").sum(), pl.col("w").sum()).sort("g")
    d = np.asarray(per_game["d"].to_numpy(), dtype=np.float64)
    w = np.asarray(per_game["w"].to_numpy(), dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (n_boot, d.size))
    boots = d[idx].sum(axis=1) / w[idx].sum(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(np.sum(d)) / float(np.sum(w)), float(lo), float(hi)


def pressure_gate(
    frame: pl.DataFrame,
    fit_seasons: list[int],
    test_seasons: list[int],
    n_boot: int,
    seed: int,
) -> tuple[pl.DataFrame, dict[str, PressureModel]]:
    """Step 3 of the Phase 4 plan, part 1: predicting pressure on each dropback.

    ``frame`` comes from ``dropback_matchups``. Models (fit on ``fit_seasons``, scored on
    ``test_seasons``): ``league`` (league rate only), ``log5`` (offense and defense
    combined on the logit scale, nothing fit), ``additive`` (log5 with fitted weights) and
    ``matchup`` (plus a fitted offense x defense interaction). Returns per model its
    coefficients, test log loss, and the difference from ``additive`` with a 95%
    interval from resampling games; and the fitted models.
    """
    terms = pressure_terms(frame)
    fit = terms.filter(pl.col("season").is_in(fit_seasons))
    test = terms.filter(pl.col("season").is_in(test_seasons))
    models = {
        "league": PressureModel(w_off=0.0, w_def=0.0),
        "log5": LOG5,
        "additive": fit_pressure_model(fit, matchup=False),
        "matchup": fit_pressure_model(fit, matchup=True),
    }
    y = test["pressure"].cast(pl.Float64).to_numpy()
    weight = np.ones_like(y)
    losses = {}
    for name, model in models.items():
        p = model.probability(test)
        losses[name] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    rows = []
    for name, model in models.items():
        diff, lo, hi = _game_bootstrap(
            test["game_id"], losses[name] - losses["additive"], weight, n_boot, seed
        )
        rows.append(
            {
                "model": name,
                "intercept": model.intercept,
                "w_off": model.w_off,
                "w_def": model.w_def,
                "w_matchup": model.w_matchup,
                "log_loss": float(losses[name].mean()),
                "vs_additive": diff,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pl.DataFrame(rows), models


def pass_epa_gate(
    frame: pl.DataFrame,
    models: dict[str, PressureModel],
    test_seasons: list[int],
    n_boot: int,
    seed: int,
) -> pl.DataFrame:
    """Step 3, part 2: predicting each team-game's EPA per dropback.

    Predictions for the offense in each test game: ``league`` (league EPA per dropback),
    ``additive`` (league + offense + defense EPA per dropback) and ``mixture/<model>``
    (pressure chance from that model times EPA when pressured, plus the rest times EPA
    when clean; ``expected_pass_epa``). Squared error is weighted by dropbacks; the
    difference from ``additive`` has a 95% interval from resampling games.
    """
    test = pressure_terms(frame.filter(pl.col("season").is_in(test_seasons)))
    predictions: dict[str, pl.Expr] = {
        "league": pl.col("league_epa").first(),
        "additive": pl.col("additive").first(),
    }
    columns: dict[str, pl.Series] = {}
    for name in ("log5", "additive", "matchup"):
        mixed = expected_pass_epa(test, models[name].probability(test))
        columns[f"mixture/{name}"] = mixed["mixture"]
    with_preds = expected_pass_epa(test, LOG5.probability(test)).with_columns(
        pl.Series(k, v) for k, v in columns.items()
    )
    for key in columns:
        predictions[key] = pl.col(key).first()
    games = (
        with_preds.group_by("game_id", "offense")
        .agg(
            pl.col("epa").mean().alias("actual"),
            pl.len().cast(pl.Float64).alias("w"),
            *[expr.alias(name) for name, expr in predictions.items()],
        )
        .sort("game_id", "offense")  # fixed order, so results repeat exactly
    )
    w = games["w"].to_numpy()
    actual = games["actual"].to_numpy()
    errors = {name: (games[name].to_numpy() - actual) ** 2 for name in predictions}
    rows = []
    for name, err in errors.items():
        diff, lo, hi = _game_bootstrap(games["game_id"], err - errors["additive"], w, n_boot, seed)
        rows.append(
            {
                "prediction": name,
                "team_games": games.height,
                "mse": float((err * w).sum() / w.sum()),
                "vs_additive": diff,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pl.DataFrame(rows)
