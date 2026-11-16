"""Check every registry entry calls a real nflreadpy loader with the right arguments.

The loaders are replaced with recorders, so nothing is downloaded. ``monkeypatch``
raises if an attribute is missing, which catches nflreadpy API renames.
"""

from __future__ import annotations

from typing import Any

import nflreadpy
import polars as pl
import pytest

from nflsim.data import DATASETS


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    loaders = {
        "load_pbp", "load_schedules", "load_rosters", "load_rosters_weekly",
        "load_player_stats", "load_team_stats", "load_injuries", "load_depth_charts",
        "load_snap_counts", "load_participation", "load_nextgen_stats",
        "load_pfr_advstats", "load_players", "load_teams",
    }  # fmt: skip
    for loader in loaders:

        def fake(*args: Any, _name: str = loader, **kwargs: Any) -> pl.DataFrame:
            calls.append((_name, args, kwargs))
            return pl.DataFrame({"x": [1]})

        monkeypatch.setattr(nflreadpy, loader, fake)
    return calls


def test_every_dataset_fetches_through_nflreadpy(
    recorded: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
) -> None:
    for spec in DATASETS.values():
        frame = spec.fetch_season(2020) if spec.fetch_season else spec.fetch_all()  # type: ignore[misc]
        assert frame.height == 1, spec.name
    assert len(recorded) == len(DATASETS)


def test_variant_datasets_pass_stat_type(
    recorded: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
) -> None:
    fetch = DATASETS["ngs_receiving"].fetch_season
    assert fetch is not None
    fetch(2021)
    fetch = DATASETS["pfr_def"].fetch_season
    assert fetch is not None
    fetch(2021)
    assert recorded == [
        ("load_nextgen_stats", (2021,), {"stat_type": "receiving"}),
        ("load_pfr_advstats", (2021,), {"stat_type": "def", "summary_level": "week"}),
    ]


def test_each_spec_has_exactly_one_fetcher() -> None:
    for spec in DATASETS.values():
        assert (spec.fetch_season is None) != (spec.fetch_all is None), spec.name
        assert spec.seasonal == (spec.first_season is not None), spec.name
