"""Data access: nflverse datasets cached locally as parquet."""

from nflsim.data.cache import DataCache
from nflsim.data.injuries import depth_chart_weeks, injury_reports, kickoffs
from nflsim.data.players import (
    PBP_PLAYER_COLUMNS,
    id_coverage,
    pfr_crosswalk,
    player_snaps,
    roster_weeks,
    unmatched_snaps,
)
from nflsim.data.registry import DATASETS, DatasetSpec
from nflsim.data.seasons import current_season, parse_seasons
from nflsim.data.teams import TEAM_ALIASES, normalize_teams

__all__ = [
    "DATASETS",
    "PBP_PLAYER_COLUMNS",
    "TEAM_ALIASES",
    "DataCache",
    "DatasetSpec",
    "current_season",
    "depth_chart_weeks",
    "id_coverage",
    "injury_reports",
    "kickoffs",
    "normalize_teams",
    "parse_seasons",
    "pfr_crosswalk",
    "player_snaps",
    "roster_weeks",
    "unmatched_snaps",
]
