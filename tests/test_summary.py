from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import polars as pl
import pytest

from nflsim.summary import (
    betting_spread,
    format_spread,
    game_lines,
    line_dict,
    quantile_dict,
    records,
    spread_result,
    total_result,
)

# Home minus away margins and totals for six runs.
MARGIN = np.array([10, 3, 3, -3, 7, 0])
TOTAL = np.array([40, 47, 48, 44, 51, 47])


def test_spread_result_counts_covers_and_pushes_for_the_home_line() -> None:
    # Home -3: covers when it wins by 4+, pushes on exactly 3.
    r = spread_result(MARGIN, -3.0, "given")
    assert (r.win, r.push, r.lose) == pytest.approx((2 / 6, 2 / 6, 2 / 6))
    assert r.win_no_push == pytest.approx(0.5)
    # Home +2.5 (underdog): covers unless it loses by 3+.
    r = spread_result(MARGIN, 2.5, "given")
    assert (r.win, r.push, r.lose) == pytest.approx((5 / 6, 0.0, 1 / 6))


def test_total_result_counts_overs_and_pushes() -> None:
    r = total_result(TOTAL, 47.0, "schedule")
    assert (r.win, r.push, r.lose) == pytest.approx((2 / 6, 2 / 6, 2 / 6))
    assert total_result(TOTAL, 47.5, "given").push == 0.0


def test_win_no_push_is_nan_when_every_run_pushes() -> None:
    assert math.isnan(total_result(np.array([47, 47]), 47.0, "given").win_no_push)


def test_betting_spread_flips_the_nflverse_sign() -> None:
    assert betting_spread(3.5) == -3.5  # nflverse: home favored by 3.5
    assert betting_spread(-2.0) == 2.0
    assert str(betting_spread(0.0)) == "0.0"  # not -0.0


def test_game_lines_prefer_given_lines_and_fall_back_to_the_schedule() -> None:
    given = game_lines(MARGIN, TOTAL, -3.0, None, 6.5, 44.5)
    assert [(x.kind, x.line, x.source) for x in given] == [
        ("spread", -3.0, "given"),
        ("total", 44.5, "schedule"),
    ]
    sched = game_lines(MARGIN, TOTAL, None, None, 6.5, None)
    assert [(x.kind, x.line, x.source) for x in sched] == [("spread", -6.5, "schedule")]
    assert game_lines(MARGIN, TOTAL, None, None, None, None) == []


def test_results_need_runs() -> None:
    with pytest.raises(ValueError, match="no simulated runs"):
        spread_result(np.array([], dtype=np.int64), -3.0, "given")


def test_format_spread() -> None:
    assert format_spread("KC", -3.5) == "KC -3.5"
    assert format_spread("KC", 2.0) == "KC +2"
    assert format_spread("KC", 0.0) == "KC pk"


def test_line_dict_names_the_sides() -> None:
    spread = line_dict(spread_result(MARGIN, -3.0, "given"))
    assert set(spread) >= {"home_cover", "push", "away_cover", "home_cover_no_push"}
    over = line_dict(total_result(np.array([47]), 47.0, "given"))
    assert over["over"] == 0.0 and over["push"] == 1.0
    assert over["over_no_push"] is None  # NaN is not valid JSON


def test_records_and_quantiles_are_json_ready() -> None:
    frame = pl.DataFrame(
        {"player": ["A", "B"], "p": [0.5, float("nan")], "when": [dt.date(2026, 11, 18)] * 2}
    )
    rows = records(frame)
    assert rows[1]["p"] is None and rows[0]["when"] == "2026-11-18"
    assert records(None) == []
    assert quantile_dict({0.1: 3.0, 0.5: 7.0, 0.9: 14.0}) == {"p10": 3.0, "p50": 7.0, "p90": 14.0}
    json.dumps({"rows": rows}, allow_nan=False)
