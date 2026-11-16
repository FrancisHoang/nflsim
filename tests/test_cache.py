from __future__ import annotations

import os
import socket
import time

import polars as pl
import pytest

from nflsim.data import DataCache, DatasetSpec
from tests.conftest import FakeSource


def test_first_load_fetches_and_writes_parquet(cache: DataCache, source: FakeSource) -> None:
    frame = cache.load("games", [2021])
    assert frame.height == 2
    assert source.calls == [("games", 2021)]
    assert cache.path("games", 2021).exists()


def test_second_load_reads_cache_without_fetching(cache: DataCache, source: FakeSource) -> None:
    first = cache.load("games", [2021])
    second = cache.load("games", [2021])
    assert source.calls == [("games", 2021)]
    assert first.equals(second)


def test_refresh_forces_refetch(cache: DataCache, source: FakeSource) -> None:
    cache.load("games", [2021])
    cache.load("games", [2021], refresh=True)
    assert source.calls == [("games", 2021), ("games", 2021)]


def test_multiple_seasons_stack_with_schema_drift(cache: DataCache) -> None:
    frame = cache.load("games", [2023, 2022, 2022])
    assert frame["season"].to_list() == [2022, 2022, 2023, 2023]
    assert frame["spread_line"].null_count() == 2


def test_only_missing_seasons_are_fetched(cache: DataCache, source: FakeSource) -> None:
    cache.load("games", [2021])
    cache.load("games", [2021, 2022])
    assert source.calls == [("games", 2021), ("games", 2022)]


def test_stale_current_season_is_refetched(cache: DataCache, source: FakeSource) -> None:
    cache.load("games", [2024])
    old = time.time() - 13 * 3600
    os.utime(cache.path("games", 2024), (old, old))
    cache.load("games", [2024])
    assert source.calls == [("games", 2024), ("games", 2024)]


def test_completed_season_never_goes_stale(cache: DataCache, source: FakeSource) -> None:
    cache.load("games", [2022])
    old = time.time() - 365 * 24 * 3600
    os.utime(cache.path("games", 2022), (old, old))
    cache.load("games", [2022])
    assert source.calls == [("games", 2022)]


def test_static_dataset(cache: DataCache, source: FakeSource) -> None:
    cache.load("teams")
    frame = cache.load("teams")
    assert frame["team_abbr"].to_list() == ["KC", "BUF"]
    assert source.calls == [("teams", None)]
    assert cache.path("teams").name == "teams.parquet"


@pytest.mark.parametrize(
    ("name", "seasons", "message"),
    [
        ("games", [2019], "starts in 2020"),
        ("games", [2025], "not started"),
        ("games", [], "empty"),
        ("games", None, "needs seasons"),
        ("teams", [2020], "not split by season"),
    ],
)
def test_invalid_requests(
    cache: DataCache, name: str, seasons: list[int] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cache.load(name, seasons)


def test_unknown_dataset_lists_valid_names(cache: DataCache) -> None:
    with pytest.raises(KeyError, match="games, teams"):
        cache.load("nope", [2021])


def test_failed_fetch_leaves_no_file(cache: DataCache) -> None:
    def broken(season: int) -> pl.DataFrame:
        raise ConnectionError("boom")

    cache.datasets = {
        **cache.datasets,
        "games": DatasetSpec("games", "broken", first_season=2020, fetch_season=broken),
    }
    with pytest.raises(ConnectionError):
        cache.load("games", [2021])
    assert not cache.path("games", 2021).exists()


def test_network_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="network"):
        socket.create_connection(("example.com", 80))
