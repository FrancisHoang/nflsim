from __future__ import annotations

import polars as pl
import pytest

from nflsim.config import MarketConfig
from nflsim.model.market import market_ratings

STRENGTH = {"A": 6.0, "B": 2.0, "C": -2.0, "D": -6.0}
HOME = 1.5
TENDENCY = {"A": 3.0, "B": -1.0, "C": 2.0, "D": -4.0}
LEAGUE_TOTAL = 44.0


def schedule(seasons: tuple[int, ...] = (2020,), weeks: int = 12) -> pl.DataFrame:
    pairs = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            for i, (x, y) in enumerate(pairs[(week % 3) * 2 : (week % 3) * 2 + 2]):
                home, away = (x, y) if (week + i) % 2 else (y, x)
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "home_team": home,
                        "away_team": away,
                        "spread_line": HOME + STRENGTH[home] - STRENGTH[away],
                        "total_line": LEAGUE_TOTAL + TENDENCY[home] + TENDENCY[away],
                        "location": "Home",
                    }
                )
    return pl.DataFrame(rows)


def at(frame: pl.DataFrame, season: int, week: int, col: str = "mkt") -> dict[str, float]:
    rows = frame.filter((pl.col("season") == season) & (pl.col("week") == week))
    return dict(zip(rows["team"].to_list(), rows[col].to_list(), strict=True))


def test_market_ratings_recover_strengths_from_the_spreads() -> None:
    m = market_ratings(schedule(), MarketConfig(prior_games=0.5))
    late = at(m, 2020, 12)
    for team, s in STRENGTH.items():
        assert late[team] == pytest.approx(s, abs=0.3)
    assert at(m, 2020, 1) == {t: 0.0 for t in STRENGTH}  # no lines yet, no prior season


def test_market_ratings_use_only_earlier_weeks() -> None:
    base = market_ratings(schedule(), MarketConfig())
    changed = schedule().with_columns(
        pl.when(pl.col("week") >= 6).then(-pl.col("spread_line")).otherwise(pl.col("spread_line"))
    )
    other = market_ratings(changed, MarketConfig())
    for week in range(1, 7):
        assert at(base, 2020, week) == at(other, 2020, week)
    assert at(base, 2020, 7) != at(other, 2020, 7)


def test_market_ratings_start_from_carried_over_prior_season() -> None:
    cfg = MarketConfig(carryover=0.5)
    m = market_ratings(schedule((2019, 2020)), cfg)
    week1 = at(m, 2020, 1)
    for team, s in STRENGTH.items():
        assert week1[team] == pytest.approx(0.5 * s, abs=0.2)
    games = m.filter((pl.col("season") == 2020) & (pl.col("week") == 3))
    assert games["mkt_games"].to_list() == [2, 2, 2, 2]  # one game a week, weeks 1-2


def test_market_tendencies_recover_scoring_from_the_totals() -> None:
    m = market_ratings(schedule(), MarketConfig(prior_games=0.5))
    late = at(m, 2020, 12, "mkt_tot")
    for team, t in TENDENCY.items():
        assert late[team] == pytest.approx(t, abs=0.3)
    assert at(m, 2020, 1, "mkt_tot") == {t: 0.0 for t in TENDENCY}


def test_market_tendencies_ignore_the_league_total_and_later_weeks() -> None:
    base = market_ratings(schedule(), MarketConfig())
    # A higher-scoring season moves the league total, not the team tendencies.
    shifted = market_ratings(schedule().with_columns(pl.col("total_line") + 6.0), MarketConfig())
    for week in (4, 12):
        for team, t in at(base, 2020, week, "mkt_tot").items():
            assert at(shifted, 2020, week, "mkt_tot")[team] == pytest.approx(t, abs=1e-6)
    changed = schedule().with_columns(
        pl.when(pl.col("week") >= 6)
        .then(90.0 - pl.col("total_line"))
        .otherwise(pl.col("total_line"))
        .alias("total_line")
    )
    other = market_ratings(changed, MarketConfig())
    for week in range(1, 7):
        assert at(base, 2020, week, "mkt_tot") == at(other, 2020, week, "mkt_tot")
    assert at(base, 2020, 7, "mkt_tot") != at(other, 2020, 7, "mkt_tot")


def test_market_tendencies_skip_games_without_a_total() -> None:
    sched = schedule().with_columns(
        pl.when(pl.col("week") == 2).then(None).otherwise(pl.col("total_line")).alias("total_line")
    )
    m = market_ratings(sched, MarketConfig())
    assert at(m, 2020, 2, "mkt_tot") == at(m, 2020, 3, "mkt_tot")
    assert at(m, 2020, 3, "mkt") != at(m, 2020, 2, "mkt")
