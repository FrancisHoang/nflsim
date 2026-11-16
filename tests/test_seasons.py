from __future__ import annotations

import datetime as dt

import pytest

from nflsim.data import current_season, parse_seasons


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (dt.date(2024, 9, 5), 2024),
        (dt.date(2024, 12, 31), 2024),
        (dt.date(2025, 2, 9), 2024),  # Super Bowl belongs to the previous season
        (dt.date(2025, 8, 31), 2024),
    ],
)
def test_current_season(day: dt.date, expected: int) -> None:
    assert current_season(day) == expected


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("2020", [2020]),
        ("2016-2018", [2016, 2017, 2018]),
        ("2020, 2018,2018", [2018, 2020]),
        ("2016-2017,2020", [2016, 2017, 2020]),
    ],
)
def test_parse_seasons(spec: str, expected: list[int]) -> None:
    assert parse_seasons(spec) == expected


@pytest.mark.parametrize("spec", ["", "2020-2018", "abc", " , "])
def test_parse_seasons_rejects_bad_input(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_seasons(spec)
