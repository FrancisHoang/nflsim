"""The game page: one game in depth, with what-ifs (the dashboard's ``nflsim simulate``)."""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import polars as pl
import streamlit as st

from nflsim.dashboard import data
from nflsim.dashboard.game_tables import (
    compare_rows,
    line_rows,
    margin_distribution,
    roster_choices,
    score_ranges,
)
from nflsim.dashboard.week_page import MODEL_COLOR, SIM_CHOICES, VEGAS_COLOR
from nflsim.game import GameRequest, GameRun, stat_line_cells
from nflsim.summary import betting_spread, format_spread

OTHER = "Another matchup..."
AWAY_COLOR = "#8a8984"  # neutral gray for the away side of the margin chart


@dataclass(frozen=True)
class _Choice:
    game: GameRequest
    injuries: bool
    out: tuple[str, ...]
    doubtful: tuple[str, ...]
    players: bool
    spread: float | None
    total: float | None


def _controls() -> _Choice | None:
    seasons = data.seasons()
    if not seasons:
        st.error(
            "No cached schedules and play-by-play found. Fetch data first, e.g. "
            "`uv run nflsim fetch --datasets pbp,schedules`."
        )
        return None
    with st.sidebar:
        season = st.selectbox("Season", seasons, index=0, key="game_season")
        weeks, current = data.weeks(season)
        default = weeks.index(current) if current in weeks else len(weeks) - 1
        week = st.selectbox("Week", weeks, index=default, key="game_week")
        scheduled = data.week_games(season, week)
        labels = [f"{a} @ {h}" for a, h in scheduled] + [OTHER]
        picked = st.selectbox("Game", labels, key="game_pick")
        neutral = False
        if picked == OTHER:
            teams = data.teams(season)
            home = st.selectbox("Home team", teams, key="game_home")
            away = st.selectbox("Away team", [t for t in teams if t != home], key="game_away")
            neutral = st.checkbox("Neutral site", key="game_neutral")
        else:
            away, home = scheduled[labels.index(picked)]
        n = st.select_slider("Simulations", SIM_CHOICES, value=10_000, key="game_n")
        seed = int(st.number_input("Seed", min_value=0, value=42, step=1, key="game_seed"))
        injuries = st.toggle(
            "Use the injury report",
            value=data.settings().model.injury_aware,
            key="game_injuries",
            help="Injury-aware player features. What-ifs and stat lines always use it.",
        )
        out: list[str] = []
        doubtful: list[str] = []
        with st.expander("What if...", expanded=False):
            roster = data.roster(season, week, home, away)
            if roster is None:
                st.caption("Rosters not cached (`nflsim fetch --datasets rosters_weekly,players`).")
            else:
                choices = roster_choices(roster)
                out = [
                    choices[x] for x in st.multiselect("Rule out", list(choices), key="game_out")
                ]
                doubtful = [
                    choices[x]
                    for x in st.multiselect("Doubtful", list(choices), key="game_doubtful")
                    if choices[x] not in out
                ]
                st.caption("Only QB, RB, WR and TE availability changes the prediction.")
        players = st.toggle("Player stat lines", key="game_players")
        with st.expander("Your own line", expanded=False):
            use_spread = st.checkbox("Spread", key="game_use_spread")
            spread = (
                st.number_input(
                    f"{home} spread (-3.5 = {home} favored)",
                    value=-3.0,
                    step=0.5,
                    key="game_spread",
                )
                if use_spread
                else None
            )
            use_total = st.checkbox("Total", key="game_use_total")
            total = (
                st.number_input("Total", value=45.0, step=0.5, key="game_total")
                if use_total
                else None
            )
    return _Choice(
        GameRequest(season, week, home, away, neutral, int(n), seed),
        injuries,
        tuple(out),
        tuple(doubtful),
        players,
        None if spread is None else float(spread),
        None if total is None else float(total),
    )


def _headline(g: GameRun) -> None:
    r, base = g.result, g.base
    cols = st.columns(4)

    def delta(now: float, before: float | None) -> str | None:
        return None if before is None else f"{100 * (now - before):+.1f} vs. without"

    cols[0].metric(
        f"{r.home} win",
        f"{r.home_win_prob:.1%}",
        delta(r.home_win_prob, base.home_win_prob if base else None),
    )
    cols[1].metric(
        f"{r.away} win",
        f"{1 - r.home_win_prob:.1%}",
        delta(1 - r.home_win_prob, 1 - base.home_win_prob if base else None),
    )
    cols[2].metric(
        f"Expected score ({r.home}-{r.away})", f"{r.expected_home:.1f}-{r.expected_away:.1f}"
    )
    cols[3].metric("Model line", format_spread(r.home, round(-r.mean_margin, 1)))
    vegas = g.matchup.spread_line
    if vegas is not None:
        cols[3].caption(f"Vegas: {format_spread(r.home, betting_spread(vegas))}")


def _margin_chart(g: GameRun) -> None:
    r = g.result
    dist = margin_distribution(r)
    if dist.is_empty():
        return
    bars = (
        alt.Chart(dist)
        .mark_bar(width={"band": 0.85}, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X(
                "margin:Q",
                title=f"Final margin ({r.home} minus {r.away})",
                scale=alt.Scale(domain=[-35, 35], clamp=True),
                axis=alt.Axis(values=list(range(-35, 36, 7))),
            ),
            y=alt.Y("share:Q", title="Share of simulations", axis=alt.Axis(format="%")),
            color=alt.Color(
                "winner:N",
                scale=alt.Scale(
                    domain=[r.home, r.away, "tie"], range=[MODEL_COLOR, AWAY_COLOR, "#c3c2b7"]
                ),
                legend=alt.Legend(title="Winner", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("margin:Q", title="Margin"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
                alt.Tooltip("winner:N", title="Winner"),
            ],
        )
    )
    layers = [bars]
    vegas = g.matchup.spread_line
    if vegas is not None:
        label = f"Vegas: {format_spread(r.home, betting_spread(vegas))}"
        rule = pl.DataFrame({"margin": [vegas], "label": [label]})
        layers.append(
            alt.Chart(rule)
            .mark_rule(color=VEGAS_COLOR, strokeWidth=2, strokeDash=[5, 3])
            .encode(x="margin:Q", tooltip=alt.Tooltip("label:N", title="Line"))
        )
    st.altair_chart(alt.layer(*layers).properties(height=280), width="stretch")
    if vegas is not None:
        st.caption(
            f"Dashed line: the Vegas line ({format_spread(r.home, betting_spread(vegas))}) as "
            "a margin. Margins of 3 and 7 show up as taller bars."
        )


def _tables(g: GameRun) -> None:
    # Stacked, not side by side: at laptop widths two tables clip their last column.
    st.markdown("**Score ranges**")
    st.dataframe(
        score_ranges(g.result),
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.0f")
            for c in ("10%", "25%", "50%", "75%", "90%")
        },
    )
    st.caption("10% of simulations end at or below the 10% column, and so on.")
    st.markdown("**Against the lines**")
    lines = line_rows(g.lines, g.result)
    if not lines.is_empty():
        if (lines["Push"] == 0).all():
            lines = lines.drop("Push")
        if (lines["Source"] == "Vegas (nflverse)").all():
            lines = lines.drop("Source")
    if lines.is_empty():
        st.caption("No line for this game. Set one under *Your own line*.")
    else:
        pct = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(
            lines,
            hide_index=True,
            width="stretch",
            column_config={
                c: pct for c in ("Chance", "Other chance", "Push") if c in lines.columns
            },
        )
        st.caption("From the simulated scores. This game's own line is never a model input.")


def _scenario(g: GameRun) -> None:
    if g.effects is None:
        return
    st.subheader("What if")
    st.dataframe(
        g.effects.rename({"before": "chance to play before", "after": "after"}),
        hide_index=True,
        width="stretch",
        column_config={
            "chance to play before": st.column_config.NumberColumn(format="%.2f"),
            "after": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    if g.base is not None:
        st.dataframe(
            compare_rows(g.base, g.result),
            hide_index=True,
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(format="%+.1f" if c == "Change" else "%.1f")
                for c in ("Without", "With", "Change")
            },
        )
        st.caption(
            f"Same seed ({g.result.seed}), with and without the what-ifs. Running back and "
            "receiver ratings barely tell players apart, so RB/WR/TE what-ifs move little; "
            "quarterbacks matter most."
        )


def _availability(g: GameRun) -> None:
    if g.report is None:
        return
    with st.expander(f"May not play ({g.report.height})", expanded=False):
        if g.report.is_empty():
            st.caption("Everyone the model uses is expected to play.")
        else:
            st.dataframe(
                g.report,
                hide_index=True,
                width="stretch",
                column_config={
                    "slot_share": st.column_config.NumberColumn("share", format="%.2f"),
                    "play_prob": st.column_config.NumberColumn("chance to play", format="%.2f"),
                },
            )
            how = "drawn in each run" if g.result.sampled_availability else "expected value"
            st.caption(f"From the final injury report; availability is {how}.")


def _stat_lines(g: GameRun) -> None:
    if g.stat_frame is None:
        return
    st.subheader("Player stat lines")
    cfg = data.settings().stat_lines
    st.dataframe(stat_line_cells(g.stat_frame), hide_index=True, width="stretch")
    st.caption(
        f"Expected ({cfg.low_quantile:.0%}-{cfg.high_quantile:.0%} range), from team volume x "
        "share x rate, games before this week only. Rushing yards are no better than a "
        "player's per-game average (2023-2025 check); the other stats beat it."
    )


def _used(g: GameRun) -> None:
    r, u = g.result, g.used
    if r.flags:
        with st.expander(f"Flags ({len(r.flags)})"):
            st.markdown("\n".join(f"- {f}" for f in r.flags))
    with st.expander("What this prediction used"):
        lines = [
            "Game: hypothetical (not on that week's schedule)."
            if u.scheduled_id is None
            else f"Game: {u.scheduled_id} from the schedule (site, roof, rest).",
            f"Ratings: {r.season} games through week {u.last_week}; {u.completed} completed "
            f"before week {r.week}."
            if u.last_week is not None
            else f"Ratings: no {r.season} games yet (last season only).",
            f"{r.n:,} simulations, seed {r.seed}.",
        ]
        if u.pbp_cached is not None:
            lines.append(f"Play-by-play cached {u.pbp_cached:%Y-%m-%d %H:%M}.")
        if u.market:
            lines.append(
                f"Market strength: closing spreads of {r.season} games before week {r.week}, "
                "never this game's line."
            )
        if u.starters is not None:
            names = ", ".join(
                f"{t} {name}" if name else f"{t} none listed (projection used)"
                for t, name in u.starters.items()
            )
            lines.append(f"Starting QBs (nflverse listing): {names}.")
        lines.append(f"Injury report: {'used' if u.injury_aware else 'not used'}.")
        st.markdown("\n".join(f"- {x}" for x in lines))
        if u.listed is not None and not u.listed.is_empty():
            st.dataframe(u.listed, hide_index=True, width="stretch")


def render() -> None:
    choice = _controls()
    if choice is None:
        return
    game = choice.game
    try:
        with st.spinner(f"Simulating {game.away} at {game.home} {game.n:,} times..."):
            g = data.game_run(
                game,
                choice.injuries,
                choice.out,
                choice.doubtful,
                choice.players,
                choice.spread,
                choice.total,
            )
    except (ValueError, FileNotFoundError) as err:
        st.title(f"{game.away} at {game.home}")
        st.error(str(err))
        return
    r = g.result
    site = "neutral site" if g.neutral_site else f"at {r.home}"
    st.title(f"{r.away} {site}")
    sub = f"{r.season} week {r.week}"
    if g.matchup.final is not None:
        h, a = g.matchup.final
        sub += f" · final: {r.home} {h}, {r.away} {a} (the ratings use only earlier weeks)"
    elif g.matchup.scheduled_id is None:
        sub += " · hypothetical game"
    st.caption(sub)
    _headline(g)
    _margin_chart(g)
    _tables(g)
    _scenario(g)
    _stat_lines(g)
    _availability(g)
    _used(g)


__all__ = ["render"]
