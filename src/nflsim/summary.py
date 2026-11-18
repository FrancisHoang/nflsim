"""Summaries of one simulated game: line probabilities and JSON output (Phase 7, step 4).

Lines are quoted the way sportsbooks quote them: the **home** team's spread, negative when
home is favored (home -3.5 covers when it wins by 4 or more), and the game total. nflverse
schedules store the spread the other way round (``spread_line`` > 0 when home is
favored), so ``betting_spread`` converts it.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

LineKind = Literal["spread", "total"]
LineSource = Literal["given", "schedule"]


@dataclass(frozen=True)
class LineResult:
    """Chances against one line, from the simulated runs.

    ``win`` is home covers (spread) or over (total); ``lose`` is away covers or under.
    ``push`` is non-zero only for whole-number lines.
    """

    kind: LineKind
    line: float
    source: LineSource
    win: float
    push: float
    lose: float

    @property
    def win_no_push(self) -> float:
        """``win`` with pushes left out (NaN when every run is a push)."""
        decided = self.win + self.lose
        return self.win / decided if decided > 0 else math.nan


def betting_spread(spread_line: float) -> float:
    """nflverse ``spread_line`` (positive when home is favored) as the home team's line."""
    return 0.0 - spread_line


def spread_result(margin: np.ndarray, home_spread: float, source: LineSource) -> LineResult:
    """Home cover / push / away cover chances for the home team's ``home_spread``.

    ``margin`` is home minus away per run; home covers when ``margin + home_spread > 0``.
    """
    adjusted = np.asarray(margin, dtype=np.float64) + home_spread
    return _result("spread", home_spread, source, adjusted)


def total_result(total: np.ndarray, line: float, source: LineSource) -> LineResult:
    """Over / push / under chances for the game total ``line``."""
    return _result("total", line, source, np.asarray(total, dtype=np.float64) - line)


def _result(kind: LineKind, line: float, source: LineSource, diff: np.ndarray) -> LineResult:
    if diff.size == 0:
        raise ValueError("no simulated runs")
    return LineResult(
        kind=kind,
        line=line,
        source=source,
        win=float(np.mean(diff > 0)),
        push=float(np.mean(diff == 0)),
        lose=float(np.mean(diff < 0)),
    )


def game_lines(
    margin: np.ndarray,
    total: np.ndarray,
    spread: float | None,
    total_line: float | None,
    schedule_spread: float | None,
    schedule_total: float | None,
) -> list[LineResult]:
    """Results for the lines given on the command line, else the schedule's lines.

    ``spread`` is the home team's betting line; ``schedule_spread`` is nflverse's
    ``spread_line`` (home favored > 0). Each line falls back on its own.
    """
    out: list[LineResult] = []
    if spread is not None:
        out.append(spread_result(margin, spread, "given"))
    elif schedule_spread is not None:
        out.append(spread_result(margin, betting_spread(schedule_spread), "schedule"))
    if total_line is not None:
        out.append(total_result(total, total_line, "given"))
    elif schedule_total is not None:
        out.append(total_result(total, schedule_total, "schedule"))
    return out


def format_spread(team: str, line: float) -> str:
    """``KC -3.5``, ``KC +2``, ``KC pk``."""
    if line == 0:
        return f"{team} pk"
    return f"{team} {line:+g}"


def line_dict(line: LineResult) -> dict[str, Any]:
    if line.kind == "spread":
        names = ("home_cover", "push", "away_cover")
    else:
        names = ("over", "push", "under")
    return {
        "kind": line.kind,
        "line": line.line,
        "source": line.source,
        names[0]: line.win,
        names[1]: line.push,
        names[2]: line.lose,
        f"{names[0]}_no_push": _clean(line.win_no_push),
    }


def records(frame: pl.DataFrame | None) -> list[dict[str, Any]]:
    """Rows of ``frame`` as JSON-ready dicts (NaN -> None); empty for None."""
    if frame is None:
        return []
    return [{k: _clean(v) for k, v in row.items()} for row in frame.iter_rows(named=True)]


def quantile_dict(values: dict[float, float]) -> dict[str, float]:
    """``{0.1: x}`` -> ``{"p10": x}``."""
    return {f"p{round(q * 100)}": v for q, v in values.items()}


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    return value


__all__ = [
    "LineResult",
    "betting_spread",
    "format_spread",
    "game_lines",
    "line_dict",
    "quantile_dict",
    "records",
    "spread_result",
    "total_result",
]
