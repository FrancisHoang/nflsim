"""Drive-by-drive game simulation (Phase 5, steps 2-3).

Each simulated drive is a real drive from the training seasons, resampled from those that
started in the same field-position bin and clock bucket. It brings its points for each
side, the time it used and where the opponent's next drive started, so score shapes
(3s, 7s, 6-8 points per touchdown) come from real football. After a touchdown or field
goal (and after an opponent touchdown, which hands the ball back to the same offense) the
next drive starts from a kickoff drawn from the latest training season, because kickoff
rules change between seasons. Halves end when the clock runs out; tied games go to
overtime under the rules of the simulated season.

Team strength (step 3) is exponential tilting: within each bin and bucket, a drive's
chance is multiplied by ``exp(theta * (offense points - defense points))``. Each team's
``theta`` is set so its expected points per drive match a target, e.g. the points
model's expected points divided by the expected number of drives. Theta 0 is the league
average. Sampling is vectorized over games x runs; the only loop is over drive number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from nflsim.config import DrivesConfig
from nflsim.model.simulate import SimResult

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_HALF_SECONDS = 1800.0
# Results after which the next drive starts from a kickoff, and the one after which the
# same offense gets the ball back (the defense scored and kicks off).
_KICK_NEXT = ("Touchdown", "Field goal", "Opp touchdown")
_KEEP_BALL = "Opp touchdown"


@dataclass(frozen=True)
class DriveLibrary:
    """Historical drives packed for sampling.

    Drives are grouped by cell (field bin x clock bucket) and, within a cell, by points
    category (a distinct ``(offense points, defense points)`` pair).
    """

    field_bin_yards: float
    clock_edges: FloatArray  # half seconds left; bucket = number of edges <= clock
    n_buckets: int
    n_bins: int
    cell_map: IntArray  # raw cell -> cell with drives (empty cells borrow a neighbor)
    cat_off: FloatArray  # (K,) offense points of each category
    cat_def: FloatArray  # (K,) defense points of each category
    counts: FloatArray  # (cells, K) drives per cell and category
    starts: IntArray  # (cells, K) first drive of each cell/category in the arrays below
    seconds: FloatArray  # (drives,)
    next_start: FloatArray  # (drives,) opponent's next start (yards to goal), NaN if none
    kick_next: npt.NDArray[np.bool_]  # (drives,)
    keep_ball: npt.NDArray[np.bool_]  # (drives,)
    kickoff_starts: FloatArray  # start yard lines after kickoffs, latest training season

    def cell(self, yardline: FloatArray, clock: FloatArray) -> IntArray:
        field = np.clip(
            ((yardline - 1) // self.field_bin_yards).astype(np.int64), 0, self.n_bins - 1
        )
        bucket = np.searchsorted(self.clock_edges, clock, side="right").astype(np.int64)
        return np.asarray(self.cell_map[field * self.n_buckets + bucket], dtype=np.int64)

    @property
    def net(self) -> FloatArray:
        return self.cat_off - self.cat_def

    def expected_points(self, theta: FloatArray, cell_weights: FloatArray) -> FloatArray:
        """Expected offense points per drive for each ``theta``, averaged over cells with
        ``cell_weights`` (e.g. how often drives start in each cell)."""
        logits = np.log(self.counts + 1e-300)[None, :, :] + theta[:, None, None] * self.net
        logits -= logits.max(axis=2, keepdims=True)
        p = np.exp(logits) * (self.counts > 0)[None, :, :]
        p /= p.sum(axis=2, keepdims=True)
        per_cell = (p * self.cat_off).sum(axis=2)  # (thetas, cells)
        return np.asarray(per_cell @ (cell_weights / cell_weights.sum()), dtype=np.float64)


def build_library(
    drives: pl.DataFrame, seasons: list[int], config: DrivesConfig
) -> tuple[DriveLibrary, FloatArray]:
    """Library from the drives of ``seasons`` (``nflsim.data.drives``), plus how often
    drives start in each cell (for calibrating ``theta``).

    Drives without a start position or clock are dropped; a missing duration uses the
    median. Kickoff starts come from the latest of ``seasons``.
    """
    frame = (
        drives.filter(pl.col("season").is_in(seasons))
        .sort("game_id", "drive")
        .with_columns(
            next_start=pl.when(
                (pl.col("half").shift(-1).over("game_id") == pl.col("half"))
                & (pl.col("offense").shift(-1).over("game_id") == pl.col("defense"))
            ).then(pl.col("start_yardline").shift(-1).over("game_id")),
        )
        .filter(
            pl.col("start_yardline").is_not_null()
            & pl.col("start_half_seconds").is_not_null()
            & pl.col("result").is_not_null()
        )
    )
    median_seconds = float(frame.select(pl.col("seconds").median()).item() or 0.0)
    frame = frame.with_columns(pl.col("seconds").fill_null(median_seconds))
    if frame.is_empty():
        raise ValueError(f"no drives in seasons {seasons} to build a drive library from")

    edges = np.asarray(sorted(config.clock_edges_seconds), dtype=np.float64)
    n_buckets = edges.size + 1
    n_bins = int(np.ceil(99 / config.field_bin_yards))
    pairs = frame.select("off_points", "def_points").unique().sort("off_points", "def_points")
    cat_off = pairs["off_points"].to_numpy().astype(np.float64)
    cat_def = pairs["def_points"].to_numpy().astype(np.float64)
    lib0 = DriveLibrary(
        field_bin_yards=config.field_bin_yards,
        clock_edges=edges,
        n_buckets=n_buckets,
        n_bins=n_bins,
        cell_map=np.arange(n_bins * n_buckets, dtype=np.int64),
        cat_off=cat_off,
        cat_def=cat_def,
        counts=np.zeros((1, 1)),
        starts=np.zeros((1, 1), dtype=np.int64),
        seconds=np.zeros(0),
        next_start=np.zeros(0),
        kick_next=np.zeros(0, dtype=bool),
        keep_ball=np.zeros(0, dtype=bool),
        kickoff_starts=np.zeros(0),
    )
    cells = lib0.cell(
        frame["start_yardline"].to_numpy().astype(np.float64),
        frame["start_half_seconds"].to_numpy().astype(np.float64),
    )
    cat_index = {(o, d): k for k, (o, d) in enumerate(zip(cat_off, cat_def, strict=True))}
    cats = np.array(
        [
            cat_index[(o, d)]
            for o, d in zip(
                frame["off_points"].to_list(), frame["def_points"].to_list(), strict=True
            )
        ],
        dtype=np.int64,
    )
    n_cells, n_cats = n_bins * n_buckets, cat_off.size
    order = np.lexsort((cats, cells))
    key = cells[order] * n_cats + cats[order]
    counts = np.bincount(key, minlength=n_cells * n_cats).reshape(n_cells, n_cats)
    starts = (np.cumsum(counts.ravel()) - counts.ravel()).reshape(n_cells, n_cats)
    result = frame["result"].to_numpy()[order]
    latest = max(seasons)
    kick = (
        drives.filter(
            (pl.col("season") == latest)
            & (pl.col("start_transition") == "KICKOFF")
            & pl.col("start_yardline").is_not_null()
        )["start_yardline"]
        .to_numpy()
        .astype(np.float64)
    )
    if kick.size == 0:
        kick = frame.filter(pl.col("start_transition") == "KICKOFF")["start_yardline"].to_numpy()
    library = DriveLibrary(
        field_bin_yards=config.field_bin_yards,
        clock_edges=edges,
        n_buckets=n_buckets,
        n_bins=n_bins,
        cell_map=_fill_empty_cells(counts.sum(axis=1), n_bins, n_buckets),
        cat_off=cat_off,
        cat_def=cat_def,
        counts=counts.astype(np.float64),
        starts=starts.astype(np.int64),
        seconds=frame["seconds"].to_numpy().astype(np.float64)[order],
        next_start=frame["next_start"].fill_null(np.nan).to_numpy().astype(np.float64)[order],
        kick_next=np.isin(result, _KICK_NEXT),
        keep_ball=result == _KEEP_BALL,
        kickoff_starts=np.asarray(kick, dtype=np.float64),
    )
    cell_weights = np.bincount(cells, minlength=n_cells).astype(np.float64)
    return library, cell_weights


def _fill_empty_cells(totals: np.ndarray, n_bins: int, n_buckets: int) -> IntArray:
    """Map each cell to itself, or (if it has no drives) to the nearest field bin with
    drives in the same clock bucket, else the nearest such cell in any bucket."""
    has = (totals > 0).reshape(n_bins, n_buckets)
    if not has.any():
        raise ValueError("drive library is empty")
    out = np.arange(n_bins * n_buckets, dtype=np.int64)
    for b in range(n_bins):
        for k in range(n_buckets):
            if has[b, k]:
                continue
            same = np.nonzero(has[:, k])[0]
            if same.size:
                nb = int(same[np.argmin(np.abs(same - b))])
                out[b * n_buckets + k] = nb * n_buckets + k
            else:
                bins, buckets = np.nonzero(has)
                j = int(np.argmin(np.abs(bins - b) + np.abs(buckets - k)))
                out[b * n_buckets + k] = int(bins[j]) * n_buckets + int(buckets[j])
    return out


def theta_for_points(
    library: DriveLibrary, cell_weights: FloatArray, target: FloatArray, grid: FloatArray
) -> FloatArray:
    """Tilt that gives each ``target`` offense points per drive (interpolated on ``grid``)."""
    curve = library.expected_points(grid, cell_weights)
    return np.asarray(np.interp(target, curve, grid), dtype=np.float64)


@dataclass(frozen=True)
class Overtime:
    """Overtime rules for one game."""

    seconds: float
    both_possess: bool  # both teams get the ball unless the defense scores
    can_tie: bool  # regular season: the game ends tied when overtime runs out


def overtime_rules(season: int, playoff: bool, config: DrivesConfig) -> Overtime:
    if playoff:
        return Overtime(
            seconds=config.overtime_seconds_playoffs,
            both_possess=season >= config.both_possess_playoffs_from,
            can_tie=False,
        )
    seconds = (
        config.overtime_seconds_regular
        if season >= config.overtime_regular_10min_from
        else config.overtime_seconds_playoffs
    )
    return Overtime(
        seconds=seconds,
        both_possess=season >= config.both_possess_regular_from,
        can_tie=True,
    )


def simulate_drives(
    library: DriveLibrary,
    theta_home: FloatArray,
    theta_away: FloatArray,
    overtime: list[Overtime],
    n: int,
    seed: int,
    max_drives: int,
) -> SimResult:
    """Simulate ``n`` runs of each game (one ``theta`` pair and overtime rule per game).

    Returns final scores like ``nflsim.model.simulate_scores`` (games x runs). Regular
    season games can end tied when overtime runs out.
    """
    games = theta_home.size
    rng = np.random.default_rng(seed)
    shape = (games, n)
    theta = np.stack([np.repeat(theta_home[:, None], n, 1), np.repeat(theta_away[:, None], n, 1)])
    score = np.zeros((2, *shape))
    first = rng.integers(0, 2, shape)  # who receives the opening kickoff
    poss = first.copy()
    half = np.ones(shape, dtype=np.int64)  # 1, 2, 3 = overtime
    clock = np.full(shape, _HALF_SECONDS)
    yard = _kickoff(library, rng, shape)
    done = np.zeros(shape, dtype=bool)
    ot_seconds = np.array([o.seconds for o in overtime])[:, None] * np.ones(shape)
    both = np.array([o.both_possess for o in overtime])[:, None] & np.ones(shape, dtype=bool)
    can_tie = np.array([o.can_tie for o in overtime])[:, None] & np.ones(shape, dtype=bool)
    ot_poss = np.zeros((2, *shape), dtype=np.int64)
    ot_first_td = np.zeros(shape, dtype=bool)

    for _ in range(max_drives):
        live = ~done
        if not live.any():
            break
        idx = np.nonzero(live)
        cell = library.cell(yard[idx], clock[idx])
        th = np.where(poss[idx] == 0, theta[0][idx], theta[1][idx])
        cat, drive = _sample(library, cell, th, rng)
        off_pts, def_pts = library.cat_off[cat], library.cat_def[cat]
        p = poss[idx]
        score[p, idx[0], idx[1]] += off_pts
        score[1 - p, idx[0], idx[1]] += def_pts

        in_ot = half[idx] == 3
        ot_poss[p[in_ot], idx[0][in_ot], idx[1][in_ot]] += 1
        first_ot_drive = in_ot & (ot_poss[0][idx] + ot_poss[1][idx] == 1)
        ot_first_td[idx[0][first_ot_drive], idx[1][first_ot_drive]] = off_pts[first_ot_drive] >= 6

        keep = library.keep_ball[drive]
        nxt = library.next_start[drive]
        kicked = library.kick_next[drive] | np.isnan(nxt)
        new_yard = np.where(kicked, _kickoff(library, rng, (drive.size,)), nxt)
        new_poss = np.where(keep, p, 1 - p)
        new_clock = clock[idx] - library.seconds[drive]
        yard[idx], poss[idx], clock[idx] = new_yard, new_poss, new_clock

        # Overtime endings, checked after each overtime drive.
        diff = score[0][idx] - score[1][idx]
        both_had = (ot_poss[0][idx] > 0) & (ot_poss[1][idx] > 0)
        decided = np.where(
            both[idx],
            both_had | (def_pts > 0),  # 2022+/2024+: both possess unless the defense scores
            both_had | ot_first_td[idx] | (def_pts > 0),  # earlier: first-drive TD wins
        )
        end_ot = in_ot & (diff != 0) & decided
        done[idx[0][end_ot], idx[1][end_ot]] = True

        # Clock expiry: halftime, end of regulation, or end of overtime.
        expired = (new_clock <= 0) & ~end_ot
        h = half[idx]
        to_second = expired & (h == 1)
        end_reg = expired & (h == 2)
        end_ot_clock = expired & (h == 3)
        g, r = idx
        half[g[to_second], r[to_second]] = 2
        clock[g[to_second], r[to_second]] = _HALF_SECONDS
        poss[g[to_second], r[to_second]] = 1 - first[g[to_second], r[to_second]]
        yard[g[to_second], r[to_second]] = _kickoff(library, rng, (int(to_second.sum()),))
        tied = diff == 0
        finished = end_reg & ~tied
        done[g[finished], r[finished]] = True
        to_ot = end_reg & tied
        half[g[to_ot], r[to_ot]] = 3
        clock[g[to_ot], r[to_ot]] = ot_seconds[g[to_ot], r[to_ot]]
        poss[g[to_ot], r[to_ot]] = rng.integers(0, 2, int(to_ot.sum()))
        yard[g[to_ot], r[to_ot]] = _kickoff(library, rng, (int(to_ot.sum()),))
        # Overtime clock out: regular season ends tied, playoffs play another period.
        stop = end_ot_clock & (can_tie[idx] | (diff != 0))
        done[g[stop], r[stop]] = True
        more = end_ot_clock & ~stop
        clock[g[more], r[more]] = ot_seconds[g[more], r[more]]

    return SimResult(home=score[0].astype(np.int64), away=score[1].astype(np.int64))


def _kickoff(library: DriveLibrary, rng: np.random.Generator, shape: tuple[int, ...]) -> FloatArray:
    pool = library.kickoff_starts
    return np.asarray(pool[rng.integers(0, pool.size, shape)], dtype=np.float64)


def _sample(
    library: DriveLibrary, cell: IntArray, theta: FloatArray, rng: np.random.Generator
) -> tuple[IntArray, IntArray]:
    """Points category (tilted by ``theta``) and a historical drive for each draw."""
    counts = library.counts[cell]  # (N, K)
    logits = np.where(
        counts > 0, np.log(np.maximum(counts, 1e-300)) + theta[:, None] * library.net, -np.inf
    )
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    cum = np.cumsum(weights, axis=1)
    u = rng.random(cell.size) * cum[:, -1]
    cat = (cum < u[:, None]).sum(axis=1)
    cat = np.minimum(cat, library.cat_off.size - 1)
    within = (rng.random(cell.size) * counts[np.arange(cell.size), cat]).astype(np.int64)
    drive = library.starts[cell, cat] + within
    return cat.astype(np.int64), drive.astype(np.int64)
