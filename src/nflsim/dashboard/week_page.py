"""The week page: every scheduled game of a week, model next to the Vegas line."""

from __future__ import annotations

import altair as alt
import polars as pl
import streamlit as st

from nflsim.dashboard import data
from nflsim.dashboard.tables import lines_chart_data, week_table
from nflsim.weekly import WeekRun, pick_record

# Reference palette, categorical slots 1 and 2 (validated pair): model blue, Vegas orange.
MODEL_COLOR = "#2a78d6"
VEGAS_COLOR = "#eb6834"
SIM_CHOICES = (1_000, 10_000, 50_000)


def _controls() -> tuple[int, int, int, int, bool] | None:
    """Sidebar pickers. None when nothing is cached."""
    seasons = data.seasons()
    if not seasons:
        st.error(
            "No cached schedules and play-by-play found. Fetch data first, e.g. "
            "`uv run nflsim fetch --datasets pbp,schedules`."
        )
        return None
    with st.sidebar:
        season = st.selectbox("Season", seasons, index=0)
        weeks, current = data.weeks(season)
        default = weeks.index(current) if current in weeks else len(weeks) - 1
        week = st.selectbox("Week", weeks, index=default)
        n = st.select_slider("Simulations per game", SIM_CHOICES, value=10_000)
        seed = int(st.number_input("Seed", min_value=0, value=42, step=1))
        injuries = st.toggle(
            "Use the injury report",
            value=data.settings().model.injury_aware,
            help="Injury-aware player features. Needs injuries, depth charts and snap "
            "counts cached.",
        )
    return season, week, int(n), seed, injuries


def _headline(run: WeekRun) -> None:
    frame = run.frame
    cols = st.columns(3)
    cols[0].metric("Games simulated", f"{run.simulated} of {frame.height}")
    record = pick_record(frame)
    if record.games:
        cols[1].metric("Winners picked", f"{record.model} of {record.games}")
        cols[1].caption(f"Vegas favorites won {record.vegas} of {record.vegas_picks}")
    else:
        cols[1].metric("Winners picked", "-")
        cols[1].caption("No games played yet")
    gaps = frame.filter(pl.col("model_spread").is_not_null() & pl.col("vegas_spread").is_not_null())
    if gaps.height:
        top = gaps.with_columns(gap=(pl.col("model_spread") - pl.col("vegas_spread")).abs()).sort(
            "gap", descending=True
        )
        r = top.row(0, named=True)
        cols[2].metric("Biggest gap to Vegas", f"{r['gap']:.1f} pts")
        cols[2].caption(f"{r['away']} @ {r['home']}")


def _table(run: WeekRun) -> None:
    table = week_table(run.frame)
    # Before kickoff there is no result to show.
    empty = [c for c in ("Final", "Model pick") if table[c].null_count() == table.height]
    table = table.drop(empty)
    pct = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=35 * (table.height + 1) + 3,  # every game visible, no inner scroll
        column_config={
            "Game": st.column_config.TextColumn(pinned=True),
            "Home win": st.column_config.ProgressColumn(
                "Home win", format="%.1f%%", min_value=0.0, max_value=100.0
            ),
            "By (pts)": st.column_config.NumberColumn(format="%.1f"),
            "Home covers": pct,
            "Model total": st.column_config.NumberColumn(format="%.1f"),
            "Vegas total": st.column_config.NumberColumn(format="%.1f"),
            "Over": pct,
            "Model leans": st.column_config.TextColumn(
                help="The team the model likes more than the Vegas line does"
            ),
            "Model pick": st.column_config.TextColumn(
                help="Whether the model's pick (home win above or below 50%) won"
            ),
        },
    )
    for row in run.frame.filter(pl.col("error").is_not_null()).iter_rows(named=True):
        st.warning(f"Not simulated: {row['away']} @ {row['home']}: {row['error']}")


def _lines_chart(run: WeekRun) -> None:
    long = lines_chart_data(run.frame)
    if long.is_empty():
        return
    order = run.frame.select(pl.format("{} @ {}", "away", "home")).to_series().to_list()
    base = alt.Chart(long).encode(
        y=alt.Y("game:N", sort=order, title=None),
    )
    rule = base.mark_rule(strokeWidth=2, color="#8a8984").encode(
        x=alt.X(
            "min(line):Q",
            title="Home team's line (negative = home favored)",
            axis=alt.Axis(tickMinStep=2, grid=True),
        ),
        x2="max(line):Q",
    )
    dots = base.mark_circle(size=110, opacity=1).encode(
        x="line:Q",
        color=alt.Color(
            "source:N",
            scale=alt.Scale(domain=["nflsim", "Vegas"], range=[MODEL_COLOR, VEGAS_COLOR]),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[
            alt.Tooltip("game:N", title="Game"),
            alt.Tooltip("source:N", title="Source"),
            alt.Tooltip("line:Q", title="Home line", format="+.1f"),
        ],
    )
    chart = (rule + dots).properties(height=34 * len(order) + 40)
    st.altair_chart(chart.configure_axisY(labelFontSize=13), width="stretch")


def _used(run: WeekRun) -> None:
    if run.flags:
        with st.expander(f"Flags ({len(run.flags)})", expanded=False):
            for flag, games in run.flags.items():
                where = "all games" if len(games) == run.simulated > 1 else ", ".join(games)
                st.markdown(f"- {flag} ({where})")
    with st.expander("What these predictions used"):
        through = (
            f"through week {run.ratings_through}"
            if run.ratings_through is not None
            else "none yet (last season only)"
        )
        lines = [
            f"Ratings: {run.season} games {through}; {run.completed_before} completed "
            f"before week {run.week}.",
            f"{run.n:,} simulations per game, seed {run.seed} (same numbers as "
            f"`nflsim week --season {run.season} --week {run.week} --n {run.n} "
            f"--seed {run.seed}`).",
        ]
        if run.pbp_cached_at is not None:
            lines.append(f"Play-by-play cached {run.pbp_cached_at:%Y-%m-%d %H:%M}.")
        if run.market:
            lines.append(
                f"Market strength: closing spreads of {run.season} games before week "
                f"{run.week}, never a game's own line. The Vegas columns are shown for "
                "comparison only."
            )
        lines.append(f"Injury report: {'used' if run.injury_aware else 'not used'}.")
        st.markdown("\n".join(f"- {x}" for x in lines))


def render() -> None:
    controls = _controls()
    if controls is None:
        return
    season, week, n, seed, injuries = controls
    st.title(f"{season} week {week}")
    st.caption(
        "Every scheduled game, simulated from information available before kickoff. "
        "Lines are the home team's (-3.5 = home favored by 3.5); cover and over are the "
        "model's chances against that game's Vegas line."
    )
    try:
        with st.status(f"Simulating every game {n:,} times...") as status:
            status.write("Building ratings from play-by-play (first load of a season only)")
            data.games(season, injuries)
            status.write("Simulating the week's games")
            run = data.week_run(season, week, n, seed, injuries)
            status.update(
                label=f"Simulated {run.simulated} games", state="complete", expanded=False
            )
    except (ValueError, FileNotFoundError) as err:
        st.error(str(err))
        return
    _headline(run)
    _table(run)
    st.subheader("Model vs. Vegas line")
    _lines_chart(run)
    _used(run)


__all__ = ["render"]
