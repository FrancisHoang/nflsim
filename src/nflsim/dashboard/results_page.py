"""The "How good is it" page: walk-forward backtest results and the experiment record."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st

from nflsim.dashboard import data
from nflsim.dashboard.results import (
    calibration_data,
    experiments,
    headline,
    method_table,
    season_range,
    season_table,
)
from nflsim.dashboard.week_page import MODEL_COLOR, VEGAS_COLOR

MUTED = "#8a8984"


def _pick_predictions() -> pl.DataFrame | None:
    files = data.backtests()
    if not files:
        st.info(
            "No saved backtest yet. Create one (takes a few minutes):\n\n"
            "`uv run nflsim backtest --seasons 2021-2024 --n 10000 --seed 42 "
            f"--output {data.backtest_folder().as_posix()}/default_2021-2024.parquet`"
        )
        return None
    with st.sidebar:
        path = st.selectbox(
            "Backtest", files, format_func=lambda p: Path(p).stem, key="results_file"
        )
        preds = data.predictions(path)
        seasons = sorted(preds["season"].unique().to_list())
        chosen = st.multiselect("Seasons", seasons, default=seasons, key="results_seasons")
    if not chosen:
        st.warning("Pick at least one season.")
        return None
    return preds.filter(pl.col("season").is_in(chosen))


def _headline(preds: pl.DataFrame) -> None:
    cols = st.columns(3)
    for col, (name, (model, vegas)) in zip(cols, headline(preds).items(), strict=True):
        digits = 4 if name == "Brier" else 2
        col.metric(name, f"{model:.{digits}f}")
        col.caption(f"Vegas closing line: {vegas:.{digits}f} (lower is better)")


def _calibration(preds: pl.DataFrame) -> None:
    cal = calibration_data(preds)
    diagonal = (
        alt.Chart(pl.DataFrame({"p": [0.0, 1.0]}))
        .mark_line(color=MUTED, strokeDash=[4, 3], strokeWidth=1)
        .encode(x="p:Q", y="p:Q")
    )
    color = alt.Color(
        "source:N",
        scale=alt.Scale(domain=["nflsim", "Vegas closing line"], range=[MODEL_COLOR, VEGAS_COLOR]),
        legend=alt.Legend(title=None, orient="top"),
    )
    base = alt.Chart(cal).encode(
        x=alt.X(
            "mean_pred:Q",
            title="Predicted home win chance",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
        y=alt.Y(
            "actual:Q",
            title="Actual home win rate",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
        color=color,
    )
    lines = base.mark_line(strokeWidth=2)
    dots = base.mark_circle(opacity=1).encode(
        size=alt.Size("games:Q", legend=None, scale=alt.Scale(range=[40, 260])),
        tooltip=[
            alt.Tooltip("source:N", title="Source"),
            alt.Tooltip("mean_pred:Q", title="Predicted", format=".1%"),
            alt.Tooltip("actual:Q", title="Actual", format=".1%"),
            alt.Tooltip("games:Q", title="Games"),
        ],
    )
    st.altair_chart(alt.layer(diagonal, lines, dots).properties(height=420), width="stretch")
    st.caption(
        "10-point bins of predicted chance (bins under 10 games left out); dot size = games. "
        "On the dashed line, predictions come true exactly as often as they say."
    )


def _experiments() -> None:
    record = experiments()
    if record is None:
        st.caption("The experiment record (`docs/experiments.csv`) is in the source repository.")
        return
    order = record["experiment"].to_list()
    base = alt.Chart(record).encode(
        y=alt.Y("experiment:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color(
            "adopted:N",
            scale=alt.Scale(domain=[True, False], range=[MODEL_COLOR, MUTED]),
            legend=alt.Legend(
                title="Adopted", orient="top", labelExpr="datum.value ? 'yes' : 'no'"
            ),
        ),
        tooltip=[
            alt.Tooltip("experiment:N", title="Experiment"),
            alt.Tooltip("date:N", title="Date"),
            alt.Tooltip("brier_diff:Q", title="Brier change", format="+.4f"),
            alt.Tooltip("ci_low:Q", title="95% CI low", format="+.4f"),
            alt.Tooltip("ci_high:Q", title="95% CI high", format="+.4f"),
            alt.Tooltip("benchmark:N", title="Compared with"),
        ],
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=MUTED).encode(x="x:Q")
    ci = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("ci_low:Q", title="Brier change vs. the benchmark at the time (left = better)"),
        x2="ci_high:Q",
    )
    dots = base.mark_circle(size=90, opacity=1).encode(x="brier_diff:Q")
    st.altair_chart(
        alt.layer(zero, ci, dots).properties(height=26 * record.height + 60), width="stretch"
    )
    st.caption(
        "Each change was compared with the model it would replace on the same games (paired "
        "game bootstrap, 95% interval), against a rule written down before the run. 'Indoor "
        "and wind' and 'Market totals' were judged on total MAE."
    )
    with st.expander("Write-ups"):
        st.dataframe(
            record.select("date", "experiment", "brier_diff", "adopted", "link"),
            hide_index=True,
            width="stretch",
            column_config={
                "brier_diff": st.column_config.NumberColumn("Brier change", format="%+.4f"),
                "link": st.column_config.LinkColumn("Write-up", display_text="docs"),
            },
        )


def render() -> None:
    st.title("How good is it")
    st.caption(
        "Walk-forward backtest: each season is predicted from a model fit only on earlier "
        "seasons, with ratings built only from games before kickoff."
    )
    preds = _pick_predictions()
    if preds is None:
        st.subheader("What was tested")
        _experiments()
        return
    st.markdown(f"**{season_range(preds)}, {preds.height:,} games**")
    _headline(preds)
    st.subheader("Every method on the same games")
    st.dataframe(
        method_table(preds),
        hide_index=True,
        width="stretch",
        column_config={
            "Margin MAE": st.column_config.NumberColumn(format="%.2f"),
            "Total MAE": st.column_config.NumberColumn(format="%.2f"),
            "Brier": st.column_config.NumberColumn(format="%.4f"),
            "Log loss": st.column_config.NumberColumn(format="%.4f"),
            "Winners picked (%)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        "nflsim uses earlier weeks' Vegas spreads as an input (market strength); the row "
        "without market inputs is the independent measure of the play-by-play model."
    )
    st.subheader("By season")
    st.dataframe(
        season_table(preds),
        hide_index=True,
        width="stretch",
        column_config={
            c: st.column_config.NumberColumn(format="%.4f" if "Brier" in c else "%.2f")
            for c in ("Brier nflsim", "Brier Vegas", "Margin MAE nflsim", "Margin MAE Vegas")
        },
    )
    st.subheader("Calibration")
    _calibration(preds)
    st.subheader("What was tested")
    _experiments()


__all__ = ["render"]
