from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from nflsim.config import InjuriesConfig, PlayerRatingsConfig
from nflsim.model.availability import (
    depth_ranks,
    expected_lineups,
    expected_places,
    expected_role_values,
    lineups,
    player_availability,
    regulars_as_of,
    role_features,
    sample_role_values,
)
from nflsim.player_eval import evaluate_injury_usage

INJ = InjuriesConfig()
PR = PlayerRatingsConfig()


def test_expected_places_healthy_is_identity() -> None:
    s = np.array([0.6, 0.3, 0.1])
    expected, repl = expected_places(s, np.ones(3))
    assert expected == pytest.approx(s)
    assert repl == pytest.approx(0.0)


def test_expected_places_next_man_up() -> None:
    s = np.array([0.8, 0.2])
    # Starter out: backup takes the starter's place; the backup's place is left empty.
    expected, repl = expected_places(s, np.array([0.0, 1.0]))
    assert expected == pytest.approx([0.0, 0.8])
    assert repl == pytest.approx(0.2)
    # Starter 50/50: backup gets 0.8 or 0.2.
    expected, repl = expected_places(s, np.array([0.5, 1.0]))
    assert expected == pytest.approx([0.4, 0.5])
    assert repl == pytest.approx(0.1)
    assert expected.sum() + repl == pytest.approx(s.sum())


def lineup_rows(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """``lineups``-shaped rows: (team, role, position, gsis_id, order, slot, p, dev, repl)."""
    frame = pl.DataFrame(
        rows,
        schema={
            "team": pl.String,
            "role": pl.String,
            "position": pl.String,
            "gsis_id": pl.String,
            "order": pl.Int32,
            "slot_share": pl.Float64,
            "play_prob": pl.Float64,
            "dev": pl.Float64,
            "repl_dev": pl.Float64,
        },
        orient="row",
    )
    return frame.with_columns(
        season=pl.lit(2024, pl.Int32), week=pl.lit(5, pl.Int32), status=pl.lit("unlisted")
    ).sort("season", "week", "team", "role", "position", "order")


LINEUP = lineup_rows(
    [
        ("KC", "dropback", "QB", "qb1", 0, 1.0, 0.6, 0.20, -0.15),
        ("KC", "dropback", "QB", "qb2", 1, 0.0, 1.0, -0.05, -0.15),
        ("KC", "target", "WR", "wr1", 0, 0.5, 0.0, 0.10, -0.05),
        ("KC", "target", "WR", "wr2", 1, 0.3, 1.0, 0.02, -0.05),
        ("KC", "target", "TE", "te1", 0, 0.2, 1.0, 0.04, -0.05),
        ("BUF", "dropback", "QB", "qb3", 0, 1.0, 1.0, 0.10, -0.15),
    ]
)


def test_expected_role_values() -> None:
    values = expected_role_values(LINEUP)
    got = {(r["team"], r["role"]): r["value"] for r in values.iter_rows(named=True)}
    # KC dropback: qb1 0.6 * 0.20; qb2 takes place 0 when qb1 is out: 0.4 * -0.05.
    assert got[("KC", "dropback")] == pytest.approx(0.6 * 0.20 + 0.4 * -0.05)
    # KC target: wr1 out, wr2 takes 0.5, replacement takes wr2's 0.3; te unchanged.
    assert got[("KC", "target")] == pytest.approx(0.5 * 0.02 + 0.3 * -0.05 + 0.2 * 0.04)
    assert got[("BUF", "dropback")] == pytest.approx(0.10)


def test_sampled_values_match_expectation() -> None:
    keys, values = sample_role_values(LINEUP, 40_000, np.random.default_rng(7))
    sampled = keys.with_columns(mean=pl.Series(values.mean(axis=1)))
    joined = sampled.join(expected_role_values(LINEUP), on=["season", "week", "team", "role"])
    assert joined["mean"].to_numpy() == pytest.approx(joined["value"].to_numpy(), abs=0.003)
    # KC dropback takes only two values: qb1 plays (0.20) or qb2 does (-0.05).
    kc = values[keys["team"].to_list().index("KC")]
    assert set(np.round(kc, 6)) <= {0.2, -0.05}


def test_sampling_is_reproducible() -> None:
    _, a = sample_role_values(LINEUP, 100, np.random.default_rng(1))
    _, b = sample_role_values(LINEUP, 100, np.random.default_rng(1))
    assert np.array_equal(a, b)


def rated_rows(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """``rated_usage``-shaped rows: (gsis_id, role, position, share, rating, group_mean)."""
    frame = pl.DataFrame(
        rows,
        schema={
            "gsis_id": pl.String,
            "role": pl.String,
            "position": pl.String,
            "share": pl.Float64,
            "rating": pl.Float64,
            "group_mean": pl.Float64,
        },
        orient="row",
    )
    return frame.with_columns(
        season=pl.lit(2024, pl.Int32),
        week=pl.lit(5, pl.Int32),
        team=pl.lit("KC"),
        group=pl.col("position"),
        role_mean=pl.lit(0.05),
        dev=(pl.col("rating") - 0.05).fill_null(0.0),
    )


RATED = rated_rows(
    [
        ("qb1", "dropback", "QB", 0.95, 0.20, 0.08),
        ("qb2", "dropback", "QB", 0.05, 0.00, 0.08),
        ("qb3", "dropback", "QB", 0.00, -0.02, 0.08),
        ("rb1", "carry", "RB", 0.60, 0.00, -0.02),
        ("rb2", "carry", "RB", 0.40, -0.04, -0.02),
    ]
)


def availability(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    """(gsis_id, status, play_prob) for KC 2024 week 5."""
    frame = pl.DataFrame(rows, schema=["gsis_id", "status", "play_prob"], orient="row")
    return frame.with_columns(
        season=pl.lit(2024, pl.Int32),
        week=pl.lit(5, pl.Int32),
        team=pl.lit("KC"),
        practice=pl.lit("none"),
        regular=pl.lit(True),
    )


def ranks(rows: list[tuple[str, int]]) -> pl.DataFrame:
    frame = pl.DataFrame(rows, schema=["gsis_id", "depth_rank"], orient="row")
    return frame.with_columns(
        season=pl.lit(2024, pl.Int32),
        week=pl.lit(5, pl.Int32),
        team=pl.lit("KC"),
        depth_rank=pl.col("depth_rank").cast(pl.Int32),
    )


HEALTHY = availability([(g, "unlisted", 1.0) for g in ("qb1", "qb2", "qb3", "rb1", "rb2")])
# The chart lists qb3 ahead of qb2, and rb2 ahead of rb1.
RANKS = ranks([("qb1", 1), ("qb3", 2), ("qb2", 3), ("rb2", 1), ("rb1", 2)])


QB1_OUT = availability(
    [("qb1", "out", 0.0), *[(g, "unlisted", 1.0) for g in ("qb2", "qb3", "rb1", "rb2")]]
)


def test_healthy_lineup_reproduces_phase2_value() -> None:
    lineup = lineups(RATED, HEALTHY, RANKS, INJ, PR)
    values = expected_role_values(lineup)
    phase2 = RATED.group_by("role").agg((pl.col("share") * pl.col("dev")).sum().alias("v"))
    merged = values.join(phase2, on="role")
    assert merged["value"].to_numpy() == pytest.approx(merged["v"].to_numpy())


def test_lineup_order_by_usage_breaks_ties_with_depth_chart() -> None:
    out = lineups(RATED, HEALTHY, RANKS, INJ.model_copy(update={"backup_order": "usage"}), PR)
    qbs = out.filter(pl.col("role") == "dropback").sort("order")["gsis_id"].to_list()
    assert qbs == ["qb1", "qb2", "qb3"]  # usage first (qb2 has 0.05, qb3 none)
    rbs = out.filter(pl.col("role") == "carry").sort("order")["gsis_id"].to_list()
    assert rbs == ["rb1", "rb2"]
    by_depth = lineups(RATED, HEALTHY, RANKS, INJ.model_copy(update={"backup_order": "depth"}), PR)
    qbs = by_depth.filter(pl.col("role") == "dropback").sort("order")["gsis_id"].to_list()
    assert qbs == ["qb1", "qb3", "qb2"]


def test_lineup_replacement_dev_and_out_starter() -> None:
    avail = QB1_OUT
    lineup = lineups(RATED, avail, RANKS, INJ, PR)
    qb = lineup.filter(pl.col("role") == "dropback")
    # Replacement QB: group mean 0.08 + unknown deviation - role mean 0.05.
    assert qb["repl_dev"][0] == pytest.approx(0.08 + PR.unknown_deviation["dropback"] - 0.05)
    players, repl = expected_lineups(lineup)
    shares = dict(
        players.filter(pl.col("role") == "dropback").select("gsis_id", "expected_share").iter_rows()
    )
    assert shares == pytest.approx({"qb1": 0.0, "qb2": 0.95, "qb3": 0.05})
    assert repl.filter(pl.col("role") == "dropback")["repl_share"][0] == pytest.approx(0.0)


def test_role_features_flags_missing_roles() -> None:
    rated = pl.concat(
        [
            RATED,
            rated_rows([("wr1", "target", "WR", None, None, None)]).with_columns(
                pl.col("share").cast(pl.Float64)
            ),
        ],
        how="vertical_relaxed",
    )
    lineup = lineups(rated, HEALTHY, RANKS, INJ, PR)
    targets = pl.DataFrame({"season": [2024], "week": [5], "team": ["KC"]}).with_columns(
        pl.col("season", "week").cast(pl.Int32)
    )
    feats = role_features(targets, expected_role_values(lineup), rated).row(0, named=True)
    assert feats["off_recv"] == 0.0
    assert feats["player_proj_missing"] == 1
    assert feats["off_qb"] == pytest.approx(0.95 * 0.15 + 0.05 * -0.05)


def snaps(rows: list[tuple[Any, ...]]) -> pl.DataFrame:
    """(season, week, team, gsis_id, offense_pct)."""
    frame = pl.DataFrame(
        rows,
        schema={
            "season": pl.Int32,
            "week": pl.Int32,
            "team": pl.String,
            "gsis_id": pl.String,
            "offense_pct": pl.Float64,
        },
        orient="row",
    )
    return frame.with_columns(
        id_matched=pl.lit(True),
        defense_pct=pl.lit(0.0),
        offense_snaps=pl.col("offense_pct") * 60,
        defense_snaps=pl.lit(0.0),
        st_snaps=pl.lit(0.0),
        snap_position=pl.lit("QB"),
    )


SNAPS = snaps(
    [
        (2024, 3, "KC", "qb1", 1.0),
        (2024, 3, "KC", "wr9", 0.1),
        (2024, 4, "KC", "qb2", 1.0),  # qb1 missed week 4
    ]
)


def test_regulars_as_of_uses_last_game_even_for_future_weeks() -> None:
    targets = pl.DataFrame({"season": [2024, 2024], "week": [4, 9], "team": ["KC", "KC"]})
    targets = targets.with_columns(pl.col("season", "week").cast(pl.Int32))
    reg = regulars_as_of(SNAPS, targets, 0.25)
    assert sorted(reg.select("week", "gsis_id").rows()) == [(4, "qb1"), (9, "qb2")]


def test_player_availability() -> None:
    players = pl.DataFrame(
        {
            "season": [2024] * 4,
            "week": [5] * 4,
            "team": ["KC"] * 4,
            "gsis_id": ["qb1", "qb2", "wr1", "rb1"],
            "position": ["QB", "QB", "WR", "RB"],
        }
    ).with_columns(pl.col("season", "week").cast(pl.Int32))
    reports = pl.DataFrame(
        {
            "season": [2024, 2024],
            "week": [5, 5],
            "team": ["KC", "KC"],
            "gsis_id": ["qb2", "wr1"],
            "status": ["questionable", "out"],
            "practice": ["limited", "dnp"],
        }
    ).with_columns(pl.col("season", "week").cast(pl.Int32))
    avail = player_availability(players, reports, SNAPS, INJ)
    p = dict(avail.select("gsis_id", "play_prob").iter_rows())
    # qb2 was a regular in week 4 (KC's last game): questionable regular QB.
    q = INJ.play_probability_regular["questionable"]["limited"]
    assert p["qb2"] == pytest.approx(q + INJ.questionable_position_adjustment["QB"])
    assert p["wr1"] == 0.0
    assert p["qb1"] == INJ.unlisted_other  # missed last game, not a regular now
    assert p["rb1"] == INJ.unlisted_other


def test_depth_ranks_best_offensive_rank() -> None:
    depth = pl.DataFrame(
        {
            "season": [2024] * 3,
            "week": [5] * 3,
            "team": ["KC"] * 3,
            "gsis_id": ["rb1", "rb1", "rb1"],
            "side": ["offense", "offense", "special"],
            "slot": ["RB", "FB", "KR"],
            "depth": [2, 1, 1],
        }
    )
    assert depth_ranks(depth)["depth_rank"].to_list() == [1]


def test_evaluate_injury_usage_rewards_known_absence() -> None:
    avail = QB1_OUT
    lineup = lineups(RATED, avail, RANKS, INJ, PR)
    history = pl.DataFrame(
        {
            "season": [2024] * 3,
            "week": [5] * 3,
            "team": ["KC"] * 3,
            "gsis_id": ["qb2", "rb1", "rb2"],
            "role": ["dropback", "carry", "carry"],
            "share": [1.0, 0.6, 0.4],
        }
    ).with_columns(pl.col("season", "week").cast(pl.Int32))
    result = evaluate_injury_usage(history, lineup, [2024], 0.10)
    qb = result.filter((pl.col("role") == "dropback") & (pl.col("subset") == "affected"))
    row = qb.row(0, named=True)
    assert row["se_injuries"] < row["se_phase2"]
    assert row["top_injuries"] == 1.0 and row["top_phase2"] == 0.0
    rb = result.filter((pl.col("role") == "carry") & (pl.col("subset") == "all")).row(0, named=True)
    assert rb["se_phase2"] == pytest.approx(0.0) and rb["se_injuries"] == pytest.approx(0.0)
