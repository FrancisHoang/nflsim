"""Display tables for the dashboard, built from ``week_frame`` (no Streamlit here).

Spreads are the home team's line in sportsbook sign (-3.5 = home favored by 3.5), as in
``nflsim week``. Numbers stay numeric where the table should sort by them.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl


def spread_text(team: str | None, line: float | None) -> str | None:
    """``"KC -3.5"``, ``"KC pk"`` for a pick'em, None without a line."""
    if team is None or line is None:
        return None
    line = round(line, 1)
    return f"{team} pk" if line == 0 else f"{team} {line:+.1f}"


def lean(
    home: str | None, away: str | None, model: float | None, vegas: float | None
) -> tuple[str | None, float | None]:
    """The team the model likes more than Vegas does, and by how many points.

    Both lines are the home team's, so ``vegas - model`` > 0 means the model makes the
    home team a bigger favorite (or smaller underdog) than Vegas.
    """
    if home is None or away is None or model is None or vegas is None:
        return None, None
    gap = round(vegas - model, 1)
    if gap == 0:
        return "none", 0.0
    return (home, gap) if gap > 0 else (away, -gap)


def kickoff_text(kickoff: str | None) -> str | None:
    """``"2026-11-22 13:00"`` -> ``"Sun 11/22 13:00"`` (schedule times are US Eastern)."""
    if kickoff is None:
        return None
    try:
        when = dt.datetime.strptime(kickoff, "%Y-%m-%d %H:%M")
    except ValueError:
        return kickoff
    return f"{when:%a} {when.month}/{when.day} {when:%H:%M}"


def _pct(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def final_text(row: dict[str, Any]) -> str | None:
    """``"HOU 23-19"`` (winner first), ``"tie 20-20"``, None before the game."""
    home, away = row.get("home_score"), row.get("away_score")
    if home is None or away is None:
        return None
    if home == away:
        return f"tie {home}-{away}"
    winner = row["home"] if home > away else row["away"]
    return f"{winner} {max(home, away)}-{min(home, away)}"


def pick_mark(row: dict[str, Any]) -> str | None:
    """Whether the model's pick (home win above or below 50%) won: "right"/"wrong"."""
    home, away, p = row.get("home_score"), row.get("away_score"), row.get("home_win")
    if home is None or away is None or p is None or home == away or p == 0.5:
        return None
    return "right" if (p > 0.5) == (home > away) else "wrong"


WEEK_COLUMNS = [
    "Game",
    "Kickoff",
    "Home win",
    "Model line",
    "Vegas line",
    "Model leans",
    "By (pts)",
    "Home covers",
    "Model total",
    "Vegas total",
    "Over",
    "Final",
    "Model pick",
]


def week_table(frame: pl.DataFrame) -> pl.DataFrame:
    """One display row per game of a ``week_frame``, in its order (chances in percent)."""
    rows = []
    for r in frame.iter_rows(named=True):
        leans, by = lean(r["home"], r["away"], r["model_spread"], r["vegas_spread"])
        rows.append(
            {
                "Game": f"{r['away']} @ {r['home']}",
                "Kickoff": kickoff_text(r["kickoff"]),
                "Home win": _pct(r["home_win"]),
                "Model line": spread_text(r["home"], r["model_spread"]),
                "Vegas line": spread_text(r["home"], r["vegas_spread"]),
                "Model leans": leans,
                "By (pts)": by,
                "Home covers": _pct(r["home_cover"]),
                "Model total": r["model_total"],
                "Vegas total": r["vegas_total"],
                "Over": _pct(r["over"]),
                "Final": final_text(r),
                "Model pick": pick_mark(r),
            }
        )
    schema = {c: pl.String for c in WEEK_COLUMNS} | {
        c: pl.Float64
        for c in ("Home win", "By (pts)", "Home covers", "Model total", "Vegas total", "Over")
    }
    return pl.DataFrame(rows, schema=schema)


def lines_chart_data(frame: pl.DataFrame) -> pl.DataFrame:
    """Long table for the model-vs-Vegas line chart: game, source, home line."""
    rows = []
    for r in frame.iter_rows(named=True):
        game = f"{r['away']} @ {r['home']}"
        for source, col in (("nflsim", "model_spread"), ("Vegas", "vegas_spread")):
            if r[col] is not None:
                rows.append({"game": game, "source": source, "line": round(r[col], 1)})
    return pl.DataFrame(rows, schema={"game": pl.String, "source": pl.String, "line": pl.Float64})


__all__ = [
    "WEEK_COLUMNS",
    "final_text",
    "kickoff_text",
    "lean",
    "lines_chart_data",
    "pick_mark",
    "spread_text",
    "week_table",
]
