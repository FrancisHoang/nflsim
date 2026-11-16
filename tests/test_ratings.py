from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from nflsim.config import RatingsConfig
from nflsim.model import RATING_COLUMNS, ratings_as_of, team_game_stats

CFG = RatingsConfig(min_wp=0.0, max_wp=1.0, carryover=0.5)


def play(
    season: int,
    week: int,
    off: str,
    dfn: str,
    epa: float,
    *,
    play_type: str = "pass",
    success: float | None = None,
    wp: float = 0.5,
    two_point: float = 0.0,
) -> dict[str, Any]:
    return {
        "season": season,
        "week": week,
        "game_id": f"{season}_{week:02d}_{min(off, dfn)}_{max(off, dfn)}",
        "posteam": off,
        "defteam": dfn,
        "play_type": play_type,
        "epa": epa,
        "success": float(epa > 0) if success is None else success,
        "wp": wp,
        "two_point_attempt": two_point,
    }


def pbp(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "season": pl.Int32,
            "week": pl.Int32,
            "game_id": pl.String,
            "posteam": pl.String,
            "defteam": pl.String,
            "play_type": pl.String,
            "epa": pl.Float64,
            "success": pl.Float64,
            "wp": pl.Float64,
            "two_point_attempt": pl.Float64,
        },
    )


def targets(*rows: tuple[int, int, str]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["season", "week", "team"], orient="row").cast(
        {"season": pl.Int32, "week": pl.Int32}
    )


def one(frame: pl.DataFrame, season: int, week: int, team: str) -> dict[str, Any]:
    rows = frame.filter(season=season, week=week, team=team).to_dicts()
    assert len(rows) == 1
    return rows[0]


def test_team_game_stats_filters_non_scrimmage_plays() -> None:
    rows = [
        play(2020, 1, "KC", "BUF", 1.0),
        play(2020, 1, "KC", "BUF", -0.5, play_type="run"),
        play(2020, 1, "KC", "BUF", -2.0, play_type="qb_kneel"),
        play(2020, 1, "KC", "BUF", -1.0, play_type="no_play"),
        play(2020, 1, "KC", "BUF", 0.7, two_point=1.0),
        play(2020, 1, "BUF", "KC", 0.2, play_type="run"),
    ]
    games = team_game_stats(pbp(rows), CFG)
    kc = games.filter(team="KC").row(0, named=True)
    assert kc["off_n"] == 2
    assert kc["off_epa"] == pytest.approx(0.5)
    assert kc["off_pass_epa"] == pytest.approx(1.0)
    assert kc["off_rush_epa"] == pytest.approx(-0.5)
    assert kc["off_success"] == 1
    assert kc["def_n"] == 1
    assert kc["def_epa"] == pytest.approx(0.2)
    assert kc["plays"] == 2


def test_win_probability_window_affects_ratings_not_pace() -> None:
    rows = [
        play(2020, 1, "KC", "BUF", 1.0, wp=0.5),
        play(2020, 1, "KC", "BUF", 3.0, wp=0.99),  # garbage time
        play(2020, 1, "BUF", "KC", 0.0),
    ]
    games = team_game_stats(pbp(rows), RatingsConfig(min_wp=0.05, max_wp=0.95))
    kc = games.filter(team="KC").row(0, named=True)
    assert kc["off_n"] == 1
    assert kc["off_epa"] == pytest.approx(1.0)
    assert kc["plays"] == 2


def season_rows(season: int, kc_epa: float, buf_epa: float, weeks: int = 2) -> list[Any]:
    """Ten plays per offense per week (alternating pass and run), all at one EPA value."""
    rows = []
    for week in range(1, weeks + 1):
        for i in range(10):
            kind = "pass" if i % 2 == 0 else "run"
            rows.append(play(season, week, "KC", "BUF", kc_epa, play_type=kind))
            rows.append(play(season, week, "BUF", "KC", buf_epa, play_type=kind))
    return rows


def test_no_leakage_from_target_week_or_later() -> None:
    base = season_rows(2021, 0.2, -0.2, weeks=3)
    games = team_game_stats(pbp(base), CFG)
    before = ratings_as_of(games, targets((2021, 3, "KC")), CFG)

    # Wildly different results in week 3 and later must not change the week-3 rating.
    future = [play(2021, w, "KC", "BUF", 9.0) for w in (3, 4) for _ in range(50)]
    games_future = team_game_stats(pbp(base + future), CFG)
    after = ratings_as_of(games_future, targets((2021, 3, "KC")), CFG)

    # Approximate: summation order can change the last bit of a float.
    expected = one(before, 2021, 3, "KC")
    got = one(after, 2021, 3, "KC")
    for column in RATING_COLUMNS:
        assert expected[column] is not None
        assert got[column] == pytest.approx(expected[column], abs=1e-12), column
    assert got["games_played"] == 2


def test_shrinkage_formula_with_league_prior() -> None:
    games = team_game_stats(pbp(season_rows(2021, 0.2, -0.2, weeks=1)), CFG)
    out = ratings_as_of(games, targets((2021, 2, "KC")), CFG)
    kc = one(out, 2021, 2, "KC")
    # League average EPA is 0 (10 plays at +0.2, 10 at -0.2). No previous season.
    k = CFG.prior_plays_epa
    assert kc["league_epa"] == pytest.approx(0.0)
    assert kc["off_epa"] == pytest.approx((10 * 0.2 + k * 0.0) / (10 + k))
    assert kc["def_epa"] == pytest.approx((10 * -0.2 + k * 0.0) / (10 + k))
    assert kc["prior_source"] == "league"


def test_prior_season_carryover_and_fading() -> None:
    # 2020: KC +0.3 per play, BUF -0.3; league average 0.
    rows = season_rows(2020, 0.3, -0.3, weeks=2) + season_rows(2021, 0.0, 0.0, weeks=1)
    games = team_game_stats(pbp(rows), CFG)
    out = ratings_as_of(games, targets((2021, 1, "KC"), (2021, 2, "KC")), CFG)

    week1 = one(out, 2021, 1, "KC")
    assert week1["prior_source"] == "prior_season"
    assert week1["games_played"] == 0
    # Prior = league 0 + 0.5 * (0.3 - 0) with no 2021 data yet.
    assert week1["off_epa"] == pytest.approx(0.15)
    assert week1["def_epa"] == pytest.approx(-0.15)

    week2 = one(out, 2021, 2, "KC")
    k = CFG.prior_plays_epa
    # League pools 2020 (40 plays, mean 0) with 2021 week 1 (20 plays at 0).
    assert week2["off_epa"] == pytest.approx((10 * 0.0 + k * 0.15) / (10 + k))
    assert 0 < week2["off_epa"] < week1["off_epa"]


def test_large_samples_approach_raw_rate() -> None:
    games = team_game_stats(pbp(season_rows(2021, 0.4, -0.4, weeks=1)), CFG)
    games = games.with_columns(pl.col(r"^(off|def)_.*$") * 1_000_000)
    kc = one(ratings_as_of(games, targets((2021, 2, "KC")), CFG), 2021, 2, "KC")
    assert kc["off_epa"] == pytest.approx(0.4, abs=1e-3)


def test_pace_is_shrunk_plays_per_game() -> None:
    rows = season_rows(2021, 0.1, 0.1, weeks=2)
    rows += [play(2021, 1, "KC", "BUF", 0.1) for _ in range(20)]  # KC: 30 then 10 plays
    games = team_game_stats(pbp(rows), CFG)
    kc = one(ratings_as_of(games, targets((2021, 3, "KC")), CFG), 2021, 3, "KC")
    league_pace = (40 + 20) / 4
    k = CFG.prior_games_pace
    assert kc["league_pace"] == pytest.approx(league_pace)
    assert kc["pace"] == pytest.approx((40 + k * league_pace) / (2 + k))


def test_no_data_at_all_gives_null_ratings_and_flag() -> None:
    games = team_game_stats(pbp(season_rows(2021, 0.1, -0.1)), CFG)
    kc = one(ratings_as_of(games, targets((2021, 1, "KC")), CFG), 2021, 1, "KC")
    assert kc["prior_source"] == "none"
    assert all(kc[c] is None for c in RATING_COLUMNS)


def test_bye_week_uses_latest_earlier_game() -> None:
    rows = season_rows(2021, 0.2, -0.2, weeks=1)
    rows += [play(2021, 5, "KC", "BUF", 1.0) for _ in range(10)]
    games = team_game_stats(pbp(rows), CFG)
    out = ratings_as_of(games, targets((2021, 4, "KC"), (2021, 6, "KC")), CFG)
    assert one(out, 2021, 4, "KC")["games_played"] == 1
    assert one(out, 2021, 6, "KC")["games_played"] == 2


def test_invalid_wp_window_rejected() -> None:
    with pytest.raises(ValueError, match="min_wp"):
        RatingsConfig(min_wp=0.9, max_wp=0.1)
