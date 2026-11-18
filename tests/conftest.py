"""Shared fixtures. The autouse ``no_network`` fixture guarantees tests never go online."""

from __future__ import annotations

import datetime as dt
import socket
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from nflsim.config import DataConfig
from nflsim.data import DataCache, DatasetSpec

# A date inside the 2024 season, so 2024 is "current" and 2023 and earlier are complete.
TODAY = dt.date(2024, 11, 10)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that tries to open a network connection.

    Loopback is allowed: on Windows, asyncio's event loop (used by Streamlit's AppTest)
    builds its internal socket pair by connecting to 127.0.0.1.
    """

    def guard(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network access is not allowed in tests")

    connect = socket.socket.connect

    def loopback_only(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            guard()
        connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    monkeypatch.setattr(socket, "create_connection", guard)


@dataclass
class FakeSource:
    """Stands in for nflreadpy: returns tiny frames and records every call."""

    calls: list[tuple[str, int | None]] = field(default_factory=list)

    def games(self, season: int) -> pl.DataFrame:
        self.calls.append(("games", season))
        frame = pl.DataFrame({"season": [season, season], "week": [1, 2], "home_score": [24, 17]})
        # Simulate schema drift: newer seasons gain a column.
        if season >= 2023:
            frame = frame.with_columns(spread_line=pl.lit(-3.5))
        return frame

    def teams(self) -> pl.DataFrame:
        self.calls.append(("teams", None))
        return pl.DataFrame({"team_abbr": ["KC", "BUF"]})


@pytest.fixture
def source() -> FakeSource:
    return FakeSource()


@pytest.fixture
def cache(tmp_path: Path, source: FakeSource) -> Iterator[DataCache]:
    datasets = {
        "games": DatasetSpec("games", "fake games", first_season=2020, fetch_season=source.games),
        "teams": DatasetSpec("teams", "fake teams", fetch_all=source.teams),
    }
    config = DataConfig(cache_dir=tmp_path / "data", current_season_max_age_hours=12)
    yield DataCache(config, datasets=datasets, today=TODAY)
