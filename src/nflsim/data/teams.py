"""Team abbreviation normalization.

nflverse play-by-play already uses each franchise's current abbreviation in every season,
but schedules keep the historical one (e.g. ``OAK`` in 2019). Normalize before joining.
"""

from __future__ import annotations

import polars as pl

# Relocated franchises: historical abbreviation -> current abbreviation.
TEAM_ALIASES: dict[str, str] = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def normalize_teams(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Map historical team abbreviations in ``columns`` to current ones."""
    return frame.with_columns(pl.col(columns).replace(TEAM_ALIASES))
