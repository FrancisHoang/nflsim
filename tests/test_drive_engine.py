from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nflsim.config import DrivesConfig
from nflsim.model.drive_engine import (
    _fill_empty_cells,
    build_library,
    overtime_rules,
    simulate_drives,
    theta_for_points,
)

CFG = DrivesConfig(field_bin_yards=50.0, clock_edges_seconds=(120.0,))


def drive_table(n_games: int = 60, seed: int = 0) -> pl.DataFrame:
    """Synthetic drives: alternating possessions, results drawn from a fixed mix."""
    rng = np.random.default_rng(seed)
    results = [
        ("Touchdown", 7.0, 0.0, 0.25),
        ("Field goal", 3.0, 0.0, 0.2),
        ("Punt", 0.0, 0.0, 0.45),
        ("Turnover", 0.0, 0.0, 0.08),
        ("Opp touchdown", 0.0, 7.0, 0.02),
    ]
    probs = np.array([r[3] for r in results])
    rows = []
    for g in range(n_games):
        clock = 3600.0
        for d in range(22):
            k = rng.choice(len(results), p=probs / probs.sum())
            name, off, de, _ = results[k]
            offense, defense = ("AAA", "BBB") if d % 2 == 0 else ("BBB", "AAA")
            rows.append(
                {
                    "game_id": f"g{g}",
                    "season": 2020 + (g % 2),
                    "drive": float(d + 1),
                    "offense": offense,
                    "defense": defense,
                    "half": "Half1" if d < 11 else "Half2",
                    "start_transition": "KICKOFF" if d in (0, 11) or k <= 1 else "PUNT",
                    "start_yardline": float(rng.integers(20, 90)),
                    "start_half_seconds": float(max(clock % 1800, 30.0)),
                    "result": name,
                    "off_points": off,
                    "def_points": de,
                    "seconds": float(rng.integers(60, 240)),
                }
            )
            clock -= 160
    return pl.DataFrame(rows)


DRIVES = drive_table()


def test_library_groups_drives_and_maps_empty_cells() -> None:
    lib, weights = build_library(DRIVES, [2020, 2021], CFG)
    assert lib.counts.sum() == DRIVES.height
    assert weights.sum() == DRIVES.height
    assert set(zip(lib.cat_off, lib.cat_def, strict=True)) == {(0, 0), (3, 0), (7, 0), (0, 7)}
    # Kickoff starts come from the latest season only.
    latest = DRIVES.filter((pl.col("season") == 2021) & (pl.col("start_transition") == "KICKOFF"))
    assert lib.kickoff_starts.size == latest.height
    mapped = _fill_empty_cells(np.array([5.0, 0.0, 0.0, 2.0]), 2, 2)
    assert mapped.tolist() == [0, 3, 0, 3]  # empty cells borrow the nearest bin, same bucket


def test_theta_is_monotone_and_hits_targets() -> None:
    lib, weights = build_library(DRIVES, [2020, 2021], CFG)
    grid = np.linspace(-2, 2, 401)
    curve = lib.expected_points(grid, weights)
    assert np.all(np.diff(curve) > 0)
    base = float(lib.expected_points(np.zeros(1), weights)[0])
    theta = theta_for_points(lib, weights, np.array([base, base + 0.5]), grid)
    assert theta[0] == pytest.approx(0.0, abs=0.01)
    assert theta[1] > 0
    assert lib.expected_points(theta[1:], weights)[0] == pytest.approx(base + 0.5, abs=0.01)


def test_simulation_is_reproducible_and_stronger_offense_scores_more() -> None:
    lib, _ = build_library(DRIVES, [2020, 2021], CFG)
    ot = [overtime_rules(2024, False, CFG)] * 2
    args = (lib, np.array([0.0, 0.5]), np.array([0.0, -0.5]), ot, 3000)
    a = simulate_drives(*args, seed=3, max_drives=CFG.max_drives)
    b = simulate_drives(*args, seed=3, max_drives=CFG.max_drives)
    assert np.array_equal(a.home, b.home) and np.array_equal(a.away, b.away)
    assert abs(a.home[0].mean() - a.away[0].mean()) < 1.5  # equal teams
    assert a.home[1].mean() - a.away[1].mean() > 5  # tilted game
    # Scores are sums of real drive outcomes: no team ends on 1 point.
    assert not np.any(a.home == 1)


def test_overtime_rules_follow_the_season() -> None:
    old = overtime_rules(2016, False, CFG)
    assert old.seconds == 900 and not old.both_possess and old.can_tie
    assert overtime_rules(2017, False, CFG).seconds == 600
    assert not overtime_rules(2023, False, CFG).both_possess
    assert overtime_rules(2024, False, CFG).both_possess
    playoff = overtime_rules(2022, True, CFG)
    assert playoff.both_possess and not playoff.can_tie and playoff.seconds == 900
    assert not overtime_rules(2021, True, CFG).both_possess


def test_regular_season_can_tie_but_playoffs_cannot() -> None:
    lib, _ = build_library(DRIVES, [2020, 2021], CFG)
    zeros = np.zeros(1)
    reg = simulate_drives(lib, zeros, zeros, [overtime_rules(2024, False, CFG)], 4000, 1, 80)
    post = simulate_drives(lib, zeros, zeros, [overtime_rules(2024, True, CFG)], 4000, 1, 80)
    assert (reg.home == reg.away).any()
    assert not (post.home == post.away).any()


def test_walk_forward_with_drive_engine() -> None:
    from nflsim.backtest import walk_forward
    from nflsim.config import SimConfig
    from tests.test_backtest import SETTINGS, season_games

    games = season_games(range(2019, 2022), per_season=120)
    drives = drive_table(n_games=40).with_columns(
        season=pl.when(pl.col("season") == 2020).then(2019).otherwise(2020).cast(pl.Int32)
    )
    settings = SETTINGS.model_copy(
        update={
            "sim": SimConfig(engine="drive"),
            "drives": CFG.model_copy(update={"pilot_runs": 200, "pilot_rounds": 1}),
        }
    )
    with pytest.raises(ValueError, match="drive_table"):
        walk_forward(games, settings, [2021], n=200, seed=1)
    a, _ = walk_forward(games, settings, [2021], n=500, seed=1, drive_table=drives)
    b, _ = walk_forward(games, settings, [2021], n=500, seed=1, drive_table=drives)
    assert a.equals(b)

    # Pilot calibration moves the simulated totals toward the points model's.
    def gap(preds: pl.DataFrame) -> float:
        diff = pl.col("model_total") - pl.col("model_mu_home") - pl.col("model_mu_away")
        return float(preds.select(diff.abs().mean()).item())

    no_pilot = settings.model_copy(
        update={"drives": settings.drives.model_copy(update={"pilot_rounds": 0})}
    )
    raw, _ = walk_forward(games, no_pilot, [2021], n=500, seed=1, drive_table=drives)
    assert gap(a) < gap(raw)
