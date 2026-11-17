from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from nflsim.config import ModelConfig, Settings, TuningConfig
from nflsim.model import team_game_stats as real_team_game_stats
from nflsim.tuning import current_params, evaluate, evaluate_holdout, grid, with_ratings
from tests.test_points import schedule
from tests.test_ratings import pbp, play

STRENGTH = {"A": 0.3, "B": 0.1, "C": -0.1, "D": -0.3}


def league(seasons: range) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Four teams, two games a week, scores loosely tied to a fixed EPA strength."""
    rng = np.random.default_rng(0)
    plays: list[dict[str, Any]] = []
    games: list[tuple[Any, ...]] = []
    for season in seasons:
        for week in range(1, 19):
            pairs = [("A", "B"), ("C", "D")] if week % 2 else [("A", "C"), ("B", "D")]
            for home, away in pairs if week % 3 else [(b, a) for a, b in pairs]:
                for i in range(6):
                    kind = "pass" if i % 2 else "run"
                    for off, dfn in ((home, away), (away, home)):
                        epa = STRENGTH[off] - STRENGTH[dfn] + rng.normal(0, 0.5)
                        plays.append(
                            play(
                                season,
                                week,
                                off,
                                dfn,
                                epa,
                                play_type=kind,
                                wp=0.97 if i == 5 else 0.5,
                            )
                        )
                edge = (STRENGTH[home] - STRENGTH[away]) * 30
                games.append(
                    (
                        season,
                        week,
                        home,
                        away,
                        int(rng.negative_binomial(6, 6 / (6 + 23 + edge / 2))),
                        int(rng.negative_binomial(6, 6 / (6 + 21 - edge / 2))),
                    )
                )
    return pbp(plays), schedule(games)


SETTINGS = Settings(
    model=ModelConfig(features=("off_epa", "def_epa"), train_seasons=2),
    tuning=TuningConfig(
        prior_plays_epa=(50.0, 300.0),
        prior_plays_success=(200.0,),
        carryover=(0.5,),
        prior_games_pace=(4.0,),
        wp_windows=((0.0, 1.0), (0.05, 0.95)),
    ),
)


def test_grid_is_full_product_plus_current() -> None:
    combos = grid(SETTINGS)
    assert len(combos) == 4  # the current settings (300, 200, 0.5, 4, 0.05-0.95) are in the grid
    assert current_params(SETTINGS.ratings) in combos
    other = SETTINGS.model_copy(
        update={"tuning": SETTINGS.tuning.model_copy(update={"prior_plays_epa": (50.0,)})}
    )
    assert len(grid(other)) == 2 + 1


def test_with_ratings_validates() -> None:
    assert with_ratings(SETTINGS, {"carryover": 0.9}).ratings.carryover == 0.9
    with pytest.raises(ValidationError):
        with_ratings(SETTINGS, {"min_wp": 0.9, "max_wp": 0.1})
    with pytest.raises(ValidationError):
        TuningConfig(carryover=())


def test_evaluate_ranks_every_combo_and_reuses_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    plays, sched = league(range(2019, 2022))
    calls: list[tuple[float, float]] = []
    real = real_team_game_stats

    def counting(frame: Any, config: Any) -> pl.DataFrame:
        calls.append((config.min_wp, config.max_wp))
        return real(frame, config)

    monkeypatch.setattr("nflsim.tuning.team_game_stats", counting)
    ranking = evaluate(plays, sched, SETTINGS, grid(SETTINGS), [2021], n=200, seed=1)

    assert ranking.height == 4
    assert ranking["brier"].is_sorted()
    assert sorted(calls) == [(0.0, 1.0), (0.05, 0.95)]  # one stats pass per window
    assert set(ranking["prior_plays_epa"]) == {50.0, 300.0}
    assert (ranking["games"] == 36).all()

    again = evaluate(plays, sched, SETTINGS, grid(SETTINGS), [2021], n=200, seed=1)
    assert ranking.equals(again)


def test_evaluate_holdout_reports_each_candidate() -> None:
    plays, sched = league(range(2019, 2022))
    out = evaluate_holdout(
        plays,
        sched,
        SETTINGS,
        {"current": current_params(SETTINGS.ratings), "loose": {"prior_plays_epa": 50.0}},
        [2021],
        n=100,
        seed=0,
    )
    assert out.height == 6  # 2 candidates x (model, vegas, home)
    assert set(out["settings"]) == {"current", "loose"}
