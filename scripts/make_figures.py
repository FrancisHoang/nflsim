"""Draw the README figures.

    uv run nflsim backtest --seasons 2021-2024 --n 10000 --seed 42 --output preds.parquet
    uv run --group docs python scripts/make_figures.py preds.parquet

Writes ``docs/img/accuracy.png``, ``docs/img/calibration.png`` and
``docs/img/experiments.png``. The experiment record is ``docs/experiments.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

mpl.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"

# Reference palette (light mode): surface, ink, and the first two categorical slots.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#9a9893"
GRID = "#e6e5e0"
MODEL = "#2a78d6"  # series 1, blue
VEGAS = "#eb6834"  # series 2, orange

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    }
)


def _title(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Bold title with a subtitle line under it, left-aligned to the figure."""
    fig.text(0.02, 0.975, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(
        0.02, 0.975 - 0.32 / fig.get_figheight(), subtitle, color=INK_2, fontsize=9.5, va="top"
    )


def _save(fig: plt.Figure, name: str, header: float = 0.62) -> None:
    fig.tight_layout(rect=(0, 0, 1, 1 - header / fig.get_figheight()))
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def accuracy(preds: pl.DataFrame) -> None:
    """Dot plot of Brier score by method, lower is better."""
    y = preds["outcome"].to_numpy()
    rows = [
        ("Home field only", brier(preds["home_p"].to_numpy(), y), MUTED),
        ("Play-by-play ratings only", brier(preds["pbp_only_p"].to_numpy(), y), MUTED),
        ("nflsim (current default)", brier(preds["model_p"].to_numpy(), y), MODEL),
        ("Vegas closing line", brier(preds["vegas_p"].to_numpy(), y), VEGAS),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    pos = np.arange(len(rows))[::-1]
    for (_label, value, color), yy in zip(rows, pos, strict=True):
        ax.plot([0.20, value], [yy, yy], color=GRID, lw=2, zorder=1)
        ax.scatter([value], [yy], s=90, color=color, edgecolor=SURFACE, lw=2, zorder=3)
        ax.text(value + 0.0012, yy, f"{value:.4f}", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(pos, [r[0] for r in rows])
    ax.set_xlim(0.20, 0.255)
    ax.set_xlabel("Brier score of home win probability (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    _title(fig, "Win probability accuracy", f"Backtest 2021-2024, {len(y):,} games")
    _save(fig, "accuracy.png")


def calibration(preds: pl.DataFrame) -> None:
    """Predicted vs. actual home win rate in 10-point bins, model and Vegas."""
    y = preds["outcome"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.8, 6.0))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1, label="Perfect")
    edges = np.linspace(0, 1, 11)
    for name, col, color in (("nflsim", "model_p", MODEL), ("Vegas", "vegas_p", VEGAS)):
        p = preds[col].to_numpy()
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        xs, ys, ns = [], [], []
        for b in range(10):
            mask = idx == b
            if mask.sum() >= 10:  # bins with fewer games are too noisy to plot
                xs.append(p[mask].mean())
                ys.append(y[mask].mean())
                ns.append(int(mask.sum()))
        ax.plot(xs, ys, color=color, lw=2, zorder=2, label=name, marker="o", ms=0)
        sizes = [18 + n / 4 for n in ns]
        ax.scatter(xs, ys, s=sizes, color=color, edgecolor=SURFACE, lw=2, zorder=3)
    ax.legend(loc="upper left", frameon=False, labelcolor=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted home win probability")
    ax.set_ylabel("Actual home win rate")
    _title(
        fig,
        "Calibration",
        "Home win rate by predicted probability, 10-point bins.\n"
        "Dot size = games; bins under 10 games left out.",
    )
    _save(fig, "calibration.png", header=0.85)


def experiments() -> None:
    """Brier difference vs. the benchmark of the time, with 95% CI, per experiment."""
    rows = pl.read_csv(ROOT / "docs" / "experiments.csv")
    n = rows.height
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * n + 1.5))
    pos = np.arange(n)[::-1]
    ax.axvline(0, color=INK_2, lw=1, zorder=1)
    for r, yy in zip(rows.iter_rows(named=True), pos, strict=True):
        adopted = r["adopted"] == "yes"
        color = MODEL if adopted else MUTED
        ax.plot([r["ci_low"], r["ci_high"]], [yy, yy], color=color, lw=2, zorder=2)
        ax.scatter([r["brier_diff"]], [yy], s=50, color=color, edgecolor=SURFACE, lw=2, zorder=3)
        if adopted:
            note = "adopted (totals)" if "total MAE" in r["experiment"] else "adopted"
            ax.text(r["ci_high"] + 0.0004, yy, note, va="center", color=INK, fontsize=8.5)
    ax.set_yticks(pos, rows["experiment"].to_list())
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Brier difference vs. the benchmark at the time (left = better)")
    _title(
        fig,
        "Every experiment, with its 95% interval",
        "Backtest 2021-2024, paired game bootstrap. "
        "Adopted only if it met a rule set before testing.",
    )
    _save(fig, "experiments.png")


def main(path: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = pl.read_parquet(path).filter(pl.col("vegas_p").is_not_null())
    accuracy(preds)
    calibration(preds)
    experiments()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
