"""Grid search over rating-shrinkage settings, scored by the walk-forward backtest.

Every combination is evaluated on the same test seasons with the same seed, so the
simulation's random draws are shared and differences come from the settings. Choose
settings on tuning seasons, then confirm on a held-out season that was never used to pick
them (``evaluate_holdout``).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import polars as pl

from nflsim.backtest import summarize, walk_forward
from nflsim.config import RatingsConfig, Settings
from nflsim.model import build_games, team_game_stats

TUNED_FIELDS = ("prior_plays_epa", "prior_plays_success", "carryover", "prior_games_pace")
METRIC_COLUMNS = ("brier", "log_loss", "margin_mae", "total_mae")


def grid(settings: Settings) -> list[dict[str, float]]:
    """All combinations from ``[tuning]``, plus the current ``[ratings]`` settings."""
    t = settings.tuning
    combos = [
        {
            "prior_plays_epa": epa,
            "prior_plays_success": success,
            "carryover": carry,
            "prior_games_pace": pace,
            "min_wp": lo,
            "max_wp": hi,
        }
        for epa, success, carry, pace, (lo, hi) in itertools.product(
            t.prior_plays_epa, t.prior_plays_success, t.carryover, t.prior_games_pace, t.wp_windows
        )
    ]
    current = current_params(settings.ratings)
    if current not in combos:
        combos.append(current)
    return combos


def current_params(ratings: RatingsConfig) -> dict[str, float]:
    """The tunable subset of a ratings config."""
    fields = (*TUNED_FIELDS, "min_wp", "max_wp")
    return {f: float(getattr(ratings, f)) for f in fields}


def with_ratings(settings: Settings, params: Mapping[str, Any]) -> Settings:
    """Settings with ``params`` applied to ``[ratings]`` (validated)."""
    ratings = RatingsConfig.model_validate({**settings.ratings.model_dump(), **params})
    return settings.model_copy(update={"ratings": ratings})


def evaluate(
    pbp: pl.DataFrame | pl.LazyFrame,
    schedules: pl.DataFrame,
    settings: Settings,
    combos: Iterable[Mapping[str, Any]],
    seasons: Iterable[int],
    n: int,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> pl.DataFrame:
    """Backtest each combination; one row per combination with its model metrics.

    Team-game stats depend only on the win-probability window, so they are computed once
    per window and reused.
    """
    combos = list(combos)
    seasons = list(seasons)
    stats_cache: dict[tuple[float, float], pl.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for i, params in enumerate(combos, start=1):
        trial = with_ratings(settings, params)
        window = (trial.ratings.min_wp, trial.ratings.max_wp)
        if window not in stats_cache:
            stats_cache[window] = team_game_stats(pbp, trial.ratings)
        games = build_games(schedules, stats_cache[window], trial.ratings)
        predictions, _ = walk_forward(games, trial, seasons, n, seed)
        summary = summarize(predictions).filter(
            (pl.col("seasons") == "all") & (pl.col("method") == "model")
        )
        metrics = summary.select("games", *METRIC_COLUMNS).row(0, named=True)
        rows.append({**current_params(trial.ratings), **metrics})
        if progress is not None:
            progress(i, len(combos))
    return pl.DataFrame(rows).sort("brier", "log_loss")


def evaluate_holdout(
    pbp: pl.DataFrame | pl.LazyFrame,
    schedules: pl.DataFrame,
    settings: Settings,
    candidates: Mapping[str, Mapping[str, Any]],
    holdout: Iterable[int],
    n: int,
    seed: int,
) -> pl.DataFrame:
    """Full summary (model, Vegas, home) on the holdout season(s) for each named candidate."""
    frames = []
    for name, params in candidates.items():
        trial = with_ratings(settings, params)
        games = build_games(schedules, team_game_stats(pbp, trial.ratings), trial.ratings)
        predictions, _ = walk_forward(games, trial, holdout, n, seed)
        summary = summarize(predictions).filter(pl.col("seasons") == "all")
        frames.append(summary.with_columns(pl.lit(name).alias("settings")))
    out = pl.concat(frames)
    return out.select("settings", "method", "games", *METRIC_COLUMNS, "pick_rate")
