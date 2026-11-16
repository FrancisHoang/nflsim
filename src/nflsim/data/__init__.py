"""Data access: nflverse datasets cached locally as parquet."""

from nflsim.data.cache import DataCache
from nflsim.data.registry import DATASETS, DatasetSpec
from nflsim.data.seasons import current_season, parse_seasons
from nflsim.data.teams import TEAM_ALIASES, normalize_teams

__all__ = [
    "DATASETS",
    "TEAM_ALIASES",
    "DataCache",
    "DatasetSpec",
    "current_season",
    "normalize_teams",
    "parse_seasons",
]
