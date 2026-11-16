"""Local parquet cache in front of the nflverse loaders.

Layout under the cache root::

    <root>/<dataset>/season=<YYYY>.parquet   # seasonal datasets
    <root>/<dataset>.parquet                 # static datasets (players, teams)

A cached file is reused unless ``refresh=True`` or it belongs to the current
season and is older than ``current_season_max_age_hours``. Completed seasons never
change, so they are fetched exactly once.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl

from nflsim.config import DataConfig
from nflsim.data.registry import DATASETS, DatasetSpec
from nflsim.data.seasons import current_season


class DataCache:
    """Fetch nflverse datasets once and read them from parquet afterwards."""

    def __init__(
        self,
        config: DataConfig,
        datasets: Mapping[str, DatasetSpec] | None = None,
        today: dt.date | None = None,
    ) -> None:
        self.root = config.cache_dir
        self.max_age_seconds = config.current_season_max_age_hours * 3600
        self.datasets = DATASETS if datasets is None else datasets
        self.current_season = current_season(today)

    def spec(self, name: str) -> DatasetSpec:
        """Look up a dataset by name, raising ``KeyError`` with the valid names."""
        try:
            return self.datasets[name]
        except KeyError:
            valid = ", ".join(sorted(self.datasets))
            raise KeyError(f"unknown dataset {name!r}; choose from: {valid}") from None

    def path(self, name: str, season: int | None = None) -> Path:
        """Return the parquet path for a dataset (and season, for seasonal data)."""
        if season is None:
            return self.root / f"{name}.parquet"
        return self.root / name / f"season={season}.parquet"

    def load(
        self, name: str, seasons: Iterable[int] | None = None, *, refresh: bool = False
    ) -> pl.DataFrame:
        """Return a dataset, fetching and caching whatever is missing or stale.

        Seasonal datasets need ``seasons``; frames from different seasons are stacked
        with missing columns filled as null, since nflverse schemas drift over time.
        """
        paths = self.ensure(name, seasons, refresh=refresh)
        frames = [pl.read_parquet(p) for p in paths]
        return frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")

    def scan(
        self, name: str, seasons: Iterable[int] | None = None, *, refresh: bool = False
    ) -> pl.LazyFrame:
        """Like ``load`` but lazy, so only the columns a query uses are read from disk."""
        paths = self.ensure(name, seasons, refresh=refresh)
        frames = [pl.scan_parquet(p) for p in paths]
        return frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")

    def ensure(
        self, name: str, seasons: Iterable[int] | None = None, *, refresh: bool = False
    ) -> list[Path]:
        """Fetch whatever is missing or stale and return the cached parquet paths."""
        spec = self.spec(name)
        if not spec.seasonal:
            if seasons is not None:
                raise ValueError(f"dataset {name!r} is not split by season")
            return [self._ensure_one(spec, None, refresh)]

        if seasons is None:
            raise ValueError(f"dataset {name!r} needs seasons")
        wanted = sorted(set(seasons))
        if not wanted:
            raise ValueError("seasons is empty")
        first = spec.first_season or 0
        too_early = [s for s in wanted if s < first]
        if too_early:
            raise ValueError(f"{name!r} starts in {first}; cannot load {too_early}")
        too_late = [s for s in wanted if s > self.current_season]
        if too_late:
            raise ValueError(f"season(s) {too_late} have not started yet")
        return [self._ensure_one(spec, s, refresh) for s in wanted]

    def _ensure_one(self, spec: DatasetSpec, season: int | None, refresh: bool) -> Path:
        path = self.path(spec.name, season)
        if not refresh and path.exists() and not self._is_stale(path, season):
            return path

        if season is None:
            assert spec.fetch_all is not None
            frame = spec.fetch_all()
        else:
            assert spec.fetch_season is not None
            frame = spec.fetch_season(season)
        self._write(frame, path)
        return path

    def _is_stale(self, path: Path, season: int | None) -> bool:
        if season is not None and season < self.current_season:
            return False
        age = time.time() - path.stat().st_mtime
        return age > self.max_age_seconds

    @staticmethod
    def _write(frame: pl.DataFrame, path: Path) -> None:
        """Write atomically so an interrupted fetch never leaves a half-written file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        frame.write_parquet(tmp)
        tmp.replace(path)
