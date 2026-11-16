"""Data access: nflverse datasets cached locally as parquet."""

from nflsim.data.cache import DataCache
from nflsim.data.registry import DATASETS, DatasetSpec
from nflsim.data.seasons import current_season, parse_seasons

__all__ = ["DATASETS", "DataCache", "DatasetSpec", "current_season", "parse_seasons"]
