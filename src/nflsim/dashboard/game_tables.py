"""Display tables and chart data for the game page (no Streamlit here)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nflsim.pipeline import MatchupResult
from nflsim.summary import LineResult, format_spread


def margin_distribution(r: MatchupResult) -> pl.DataFrame:
    """Share of runs at each final margin (home minus away), with who won.

    ``winner`` is the home team, the away team or ``"tie"``. Shares sum to 1.
    """
    margin = (r.home_scores - r.away_scores).astype(np.int64)
    if margin.size == 0:
        return pl.DataFrame(schema={"margin": pl.Int64, "share": pl.Float64, "winner": pl.String})
    values, counts = np.unique(margin, return_counts=True)
    winner = np.where(values > 0, r.home, np.where(values < 0, r.away, "tie"))
    return pl.DataFrame(
        {
            "margin": values.astype(np.int64),
            "share": counts / margin.size,
            "winner": winner.astype(str),
        }
    )


def score_ranges(r: MatchupResult) -> pl.DataFrame:
    """Points and margin at each simulated quantile, one row per quantity."""
    qs = list(r.margin_quantiles)
    rows = [
        (f"{r.home} points", r.home_quantiles),
        (f"{r.away} points", r.away_quantiles),
        (f"Margin ({r.home} minus {r.away})", r.margin_quantiles),
    ]
    return pl.DataFrame(
        {
            "": [label for label, _ in rows],
            **{f"{q:.0%}": [float(values[q]) for _, values in rows] for q in qs},
        }
    )


def line_rows(lines: list[LineResult], r: MatchupResult) -> pl.DataFrame:
    """One row per line: the line, where it came from, and the chances (percent)."""
    rows = []
    for x in lines:
        source = "your line" if x.source == "given" else "Vegas (nflverse)"
        if x.kind == "spread":
            line = f"Spread {format_spread(r.home, x.line)}"
            sides = (f"{r.home} covers", f"{r.away} covers")
        else:
            line = f"Total {x.line:g}"
            sides = ("Over", "Under")
        rows.append(
            {
                "Line": line,
                "Source": source,
                "Outcome": sides[0],
                "Chance": 100.0 * x.win,
                "Other side": sides[1],
                "Other chance": 100.0 * x.lose,
                "Push": 100.0 * x.push,
            }
        )
    schema = {
        "Line": pl.String,
        "Source": pl.String,
        "Outcome": pl.String,
        "Chance": pl.Float64,
        "Other side": pl.String,
        "Other chance": pl.Float64,
        "Push": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def compare_rows(base: MatchupResult, r: MatchupResult) -> pl.DataFrame:
    """Headline numbers without and with the overrides (same seed), and the change."""
    rows = [
        (f"{r.home} win chance (%)", 100.0 * base.home_win_prob, 100.0 * r.home_win_prob),
        (f"Expected points {r.home}", base.expected_home, r.expected_home),
        (f"Expected points {r.away}", base.expected_away, r.expected_away),
        (f"Mean margin ({r.home} minus {r.away})", base.mean_margin, r.mean_margin),
        ("Mean total", base.mean_total, r.mean_total),
    ]
    return pl.DataFrame(
        {
            "": [x[0] for x in rows],
            "Without": [x[1] for x in rows],
            "With": [x[2] for x in rows],
            "Change": [x[2] - x[1] for x in rows],
        }
    )


def roster_choices(roster: pl.DataFrame) -> dict[str, str]:
    """Picker labels (``"Josh Allen (BUF QB)"``) -> GSIS ID, for players whose
    availability the model uses (QB, RB, WR, TE), QBs first."""
    order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
    shown = (
        roster.filter(pl.col("position").is_in(list(order)))
        .with_columns(pl.col("position").replace_strict(order, return_dtype=pl.Int32).alias("_o"))
        .sort("team", "_o", "name")
    )
    return {
        f"{row['name']} ({row['team']} {row['position']})": row["gsis_id"]
        for row in shown.iter_rows(named=True)
    }


__all__ = [
    "compare_rows",
    "line_rows",
    "margin_distribution",
    "roster_choices",
    "score_ranges",
]
