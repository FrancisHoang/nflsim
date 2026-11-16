"""Helpers for working with NFL season numbers."""

from __future__ import annotations

import datetime as dt


def current_season(today: dt.date | None = None) -> int:
    """Return the NFL season in progress (or most recently finished) on ``today``.

    A season is labelled by the year it kicks off in September. January through
    August belong to the previous season (playoffs and offseason).
    """
    today = today or dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


def parse_seasons(spec: str) -> list[int]:
    """Parse a season spec like ``"2016-2025"``, ``"2018,2020"`` or ``"2016-2018,2021"``.

    Returns a sorted list with duplicates removed. Raises ``ValueError`` on bad input.
    """
    seasons: set[int] = set()
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            start_s, _, end_s = part.partition("-")
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"season range {part!r} is backwards")
            seasons.update(range(start, end + 1))
        else:
            seasons.add(int(part))
    if not seasons:
        raise ValueError(f"no seasons in {spec!r}")
    return sorted(seasons)
