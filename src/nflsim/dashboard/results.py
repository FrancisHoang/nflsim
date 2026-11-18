"""Backtest results for the dashboard's "How good is it" page (no Streamlit here).

The page reads saved walk-forward predictions (``nflsim backtest --output``) from
``<cache_dir>/backtest/*.parquet`` and scores them with the backtest harness's own
functions (``summarize``, ``calibration``), so its numbers match the CLI's tables.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nflsim.backtest import calibration, summarize

METHOD_LABELS = {
    "model": "nflsim",
    "pbp_only": "nflsim without market inputs",
    "vegas": "Vegas closing line",
    "home": "Home field only",
}
REPO_URL = "https://github.com/FrancisHoang/nflsim/blob/main/"
# docs/experiments.csv in a source checkout (src/nflsim/dashboard -> repo root).
EXPERIMENTS_CSV = Path(__file__).resolve().parents[3] / "docs" / "experiments.csv"


def backtest_files(folder: Path) -> list[Path]:
    """Saved prediction files in ``folder`` by name, last first (``default_2021-2024``
    before ``default_2017-2024``); empty when there are none."""
    if not folder.exists():
        return []
    return sorted(folder.glob("*.parquet"), key=lambda p: p.name, reverse=True)


def season_range(predictions: pl.DataFrame) -> str:
    seasons = sorted(predictions["season"].unique().to_list())
    return f"{seasons[0]}-{seasons[-1]}" if len(seasons) > 1 else str(seasons[0])


def method_table(predictions: pl.DataFrame) -> pl.DataFrame:
    """Every method's metrics over all the given games (``summarize``'s "all" rows)."""
    table = summarize(predictions).filter(pl.col("seasons") == "all")
    return table.with_columns(
        pl.col("method").replace_strict(METHOD_LABELS, default=pl.col("method")).alias("Method")
    ).select(
        "Method",
        pl.col("games").alias("Games"),
        pl.col("margin_mae").alias("Margin MAE"),
        pl.col("total_mae").alias("Total MAE"),
        pl.col("brier").alias("Brier"),
        pl.col("log_loss").alias("Log loss"),
        (100 * pl.col("pick_rate")).alias("Winners picked (%)"),
    )


def season_table(predictions: pl.DataFrame) -> pl.DataFrame:
    """Brier and margin MAE per season, nflsim next to Vegas."""
    per = summarize(predictions).filter(
        (pl.col("seasons") != "all") & pl.col("method").is_in(["model", "vegas"])
    )
    wide = per.pivot(on="method", index="seasons", values=["brier", "margin_mae"])
    return wide.select(
        pl.col("seasons").alias("Season"),
        pl.col("brier_model").alias("Brier nflsim"),
        pl.col("brier_vegas").alias("Brier Vegas"),
        pl.col("margin_mae_model").alias("Margin MAE nflsim"),
        pl.col("margin_mae_vegas").alias("Margin MAE Vegas"),
    ).sort("Season")


def calibration_data(
    predictions: pl.DataFrame, bins: int = 10, min_games: int = 10
) -> pl.DataFrame:
    """Calibration bins for nflsim and Vegas, long form (bins with few games dropped)."""
    frames = [
        calibration(predictions.drop_nulls(f"{m}_p"), m, bins).with_columns(
            pl.lit(METHOD_LABELS[m]).alias("source")
        )
        for m in ("model", "vegas")
    ]
    return pl.concat(frames).filter(pl.col("games") >= min_games)


def headline(predictions: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """(nflsim, Vegas) for Brier, margin MAE and total MAE over all games."""
    t = summarize(predictions).filter(pl.col("seasons") == "all")
    by = {r["method"]: r for r in t.iter_rows(named=True)}
    m, v = by["model"], by["vegas"]
    return {
        "Brier": (m["brier"], v["brier"]),
        "Margin MAE": (m["margin_mae"], v["margin_mae"]),
        "Total MAE": (m["total_mae"], v["total_mae"]),
    }


def experiments(path: Path = EXPERIMENTS_CSV) -> pl.DataFrame | None:
    """The experiment record (``docs/experiments.csv``), None outside a source checkout."""
    if not path.exists():
        return None
    frame = pl.read_csv(path)
    return frame.with_columns(
        (pl.col("adopted") == "yes").alias("adopted"),
        pl.when(pl.col("doc").is_not_null())
        .then(pl.lit(REPO_URL) + pl.col("doc"))
        .otherwise(None)
        .alias("link"),
    )


__all__ = [
    "METHOD_LABELS",
    "backtest_files",
    "calibration_data",
    "experiments",
    "headline",
    "method_table",
    "season_range",
    "season_table",
]
