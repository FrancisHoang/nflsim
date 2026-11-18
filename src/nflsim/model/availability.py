"""Who plays: availability from injury reports and next-man-up usage (Phase 3, step 2).

Each projected skill player gets a play probability from his final injury report
(``nflsim.model.injuries``). Within a team, role and position (QB, RB, WR, TE), players
are put in order (depth chart, then projected usage; see ``InjuriesConfig.backup_order``).
The healthy projection gives each place in that order its usage share. When a player is
out, everyone below him moves up one place and takes that place's share, so the backup
inherits the starter's usage and plays at his own rating. Shares of places left empty go
to a replacement-level player (position average plus ``unknown_deviation``), which is
flagged.

With every player available this is exactly the Phase 2 projection. Usage and features
can be computed exactly in expectation (``expected_lineups``), or sampled per simulation
run (``sample_role_values``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nflsim.config import InjuriesConfig, PlayerRatingsConfig
from nflsim.model.injuries import UNLISTED, attach_play_probability
from nflsim.model.players import ROLE_FEATURES, role_values_wide

_TEAM_WEEK = ["season", "week", "team"]
_LINEUP = [*_TEAM_WEEK, "role", "position"]

FloatArray = np.ndarray


def _time(season: pl.Expr, week: pl.Expr) -> pl.Expr:
    return (season * 100 + week).cast(pl.Int32)


def team_usage(snaps: pl.DataFrame) -> pl.DataFrame:
    """Per player-game: scrimmage snap ``share`` (max of offense and defense share), total
    ``snaps`` (incl. special teams) and PFR ``snap_position``. ``snaps`` comes from
    ``nflsim.data.player_snaps``; rows without a GSIS ID are dropped."""
    return (
        snaps.filter(pl.col("id_matched"))
        .group_by(*_TEAM_WEEK, "gsis_id")
        .agg(
            pl.max_horizontal(pl.col("offense_pct").max(), pl.col("defense_pct").max())
            .fill_null(0.0)
            .alias("share"),
            pl.sum_horizontal("offense_snaps", "defense_snaps", "st_snaps").sum().alias("snaps"),
            pl.col("snap_position").first(),
        )
    )


def previous_games(snaps: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Each target team-week with the team's last game with snap counts before it.

    Returns ``season, week, team, prev_season, prev_week`` (null when there is none).
    Works for future weeks too (e.g. a matchup being simulated).
    """
    games = (
        snaps.select(*_TEAM_WEEK)
        .unique()
        .with_columns(_t=_time(pl.col("season"), pl.col("week")))
        .select(
            "team", "_t", pl.col("season").alias("prev_season"), pl.col("week").alias("prev_week")
        )
        .sort("_t")
    )
    return (
        targets.select(*_TEAM_WEEK)
        .unique()
        .with_columns(_t=_time(pl.col("season"), pl.col("week")))
        .sort("_t")
        .join_asof(
            games,
            on="_t",
            by="team",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .drop("_t")
    )


def regulars_as_of(snaps: pl.DataFrame, targets: pl.DataFrame, min_share: float) -> pl.DataFrame:
    """Regulars for each target team-week: players with at least ``min_share`` of the
    offensive or defensive snaps in the team's previous game (known before kickoff).

    Returns ``season, week, team, gsis_id, prev_position`` (PFR position label).
    """
    usage = team_usage(snaps).filter(pl.col("share") >= min_share)
    return (
        previous_games(snaps, targets)
        .join(
            usage.select(
                pl.col("season").alias("prev_season"),
                pl.col("week").alias("prev_week"),
                "team",
                "gsis_id",
                pl.col("snap_position").alias("prev_position"),
            ),
            on=["prev_season", "prev_week", "team"],
            how="inner",
        )
        .select(*_TEAM_WEEK, "gsis_id", "prev_position")
    )


def player_availability(
    players: pl.DataFrame,
    reports: pl.DataFrame,
    snaps: pl.DataFrame,
    config: InjuriesConfig,
) -> pl.DataFrame:
    """Play probability for each (season, week, team, gsis_id, position) in ``players``.

    Adds ``status`` and ``practice`` from the final report (``unlisted``/``none`` when
    not on it), ``regular`` and ``play_prob``. Unlisted non-regulars play with
    ``config.unlisted_other``.
    """
    keys = [*_TEAM_WEEK, "gsis_id"]
    regular = regulars_as_of(snaps, players, config.regular_min_share).select(
        *keys, pl.lit(True).alias("regular")
    )
    report = reports.select(*keys, "status", "practice")
    frame = (
        players.select(*keys, "position")
        .unique(keys)
        .join(report, on=keys, how="left")
        .join(regular, on=keys, how="left")
        .with_columns(
            pl.col("status").fill_null(UNLISTED),
            pl.col("practice").fill_null("none"),
            pl.col("regular").fill_null(False),
        )
    )
    return attach_play_probability(frame, config).with_columns(
        pl.col("play_prob").fill_null(config.unlisted_other)
    )


def depth_ranks(depth: pl.DataFrame) -> pl.DataFrame:
    """Each player's best offensive depth-chart rank per team-week (1 = starter).

    ``depth`` comes from ``nflsim.data.depth_chart_weeks``.
    """
    return (
        depth.filter(pl.col("side") == "offense")
        .group_by(*_TEAM_WEEK, "gsis_id")
        .agg(pl.col("depth").min().alias("depth_rank"))
    )


def lineups(
    rated: pl.DataFrame,
    availability: pl.DataFrame,
    ranks: pl.DataFrame,
    injuries: InjuriesConfig,
    players: PlayerRatingsConfig,
) -> pl.DataFrame:
    """Ordered usage places for each (team-week, role, position).

    ``rated`` comes from ``nflsim.model.players.rated_usage``, ``availability`` from
    ``player_availability`` and ``ranks`` from ``depth_ranks``. Team-week-roles without
    a projection (null shares) are dropped. Each row is one player with ``order`` (0 =
    first), ``slot_share`` (the healthy share of that place), ``play_prob``, ``dev`` and
    ``repl_dev``: the EPA per play over the role average of a replacement-level player
    at the position (``group_mean + unknown_deviation - role_mean``).
    """
    unknown = pl.col("role").replace_strict(players.unknown_deviation, return_dtype=pl.Float64)
    frame = (
        rated.filter(pl.col("share").is_not_null() & pl.col("position").is_not_null())
        .join(
            availability.select(
                *_TEAM_WEEK, "gsis_id", "status", "practice", "regular", "play_prob"
            ),
            on=[*_TEAM_WEEK, "gsis_id"],
            how="left",
        )
        .join(ranks, on=[*_TEAM_WEEK, "gsis_id"], how="left")
        .with_columns(
            pl.col("play_prob").fill_null(1.0),
            # Replacement player's group mean: any player's in the same (week, role, group).
            pl.col("group_mean").max().over("season", "week", "role", "group").alias("_gm"),
        )
        .with_columns(
            repl_dev=(pl.col("_gm") + unknown - pl.col("role_mean")).fill_null(0.0),
        )
    )
    if injuries.backup_order == "depth":
        order_by = [pl.col("depth_rank"), -pl.col("share"), pl.col("gsis_id")]
    else:
        order_by = [-pl.col("share"), pl.col("depth_rank"), pl.col("gsis_id")]
    ordered = frame.sort(*order_by, nulls_last=True).with_columns(
        order=pl.int_range(pl.len()).over(_LINEUP).cast(pl.Int32)
    )
    # The share of each place is the healthy share of whoever is in it.
    return (
        ordered.with_columns(slot_share=pl.col("share"))
        .select(
            *_LINEUP,
            "gsis_id",
            "order",
            "slot_share",
            "play_prob",
            "dev",
            "repl_dev",
            "status",
            "practice",
            "regular",
            "depth_rank",
        )
        .sort(*_LINEUP, "order")
    )


def _count_distribution(p: FloatArray) -> FloatArray:
    """``dist[j, k]``: chance that exactly k of players 0..j-1 are available."""
    size = p.size
    dist = np.zeros((size + 1, size + 1))
    dist[0, 0] = 1.0
    for j in range(size):
        dist[j + 1, 1:] = dist[j, :-1] * p[j]
        dist[j + 1, :] += dist[j, :] * (1.0 - p[j])
    return dist


def expected_places(slot_share: FloatArray, play_prob: FloatArray) -> tuple[FloatArray, float]:
    """Expected share of each player, and of the replacement player, for one lineup.

    Player j, if available, takes place k = number of available players ahead of him.
    The replacement takes every place left empty (k >= number of available players).
    """
    dist = _count_distribution(play_prob)
    size = slot_share.size
    ahead = dist[:size, :size]  # ahead[j, k]: chance k of players 0..j-1 are available
    expected = play_prob * (ahead @ slot_share)
    total = dist[size]  # chance exactly k of all players are available
    # Place k is empty when at most k players are available.
    empty = np.cumsum(total)[:size]
    return expected, float(empty @ slot_share)


@dataclass(frozen=True)
class LineupArrays:
    """Lineups packed for fast per-run sampling: one block of players per lineup."""

    keys: pl.DataFrame  # season, week, team, role, position (one row per lineup)
    starts: np.ndarray  # first row of each lineup in the flat arrays
    sizes: np.ndarray
    slot_share: FloatArray
    play_prob: FloatArray
    dev: FloatArray
    repl_dev: FloatArray  # per lineup


def pack(lineup: pl.DataFrame) -> LineupArrays:
    """Pack ``lineups`` output (already sorted by lineup and order) for sampling."""
    sizes = lineup.group_by(_LINEUP, maintain_order=True).agg(
        pl.len().alias("n"), pl.col("repl_dev").first()
    )
    counts = sizes["n"].to_numpy().astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    return LineupArrays(
        keys=sizes.select(_LINEUP),
        starts=starts,
        sizes=counts,
        slot_share=lineup["slot_share"].to_numpy().astype(np.float64),
        play_prob=lineup["play_prob"].to_numpy().astype(np.float64),
        dev=lineup["dev"].to_numpy().astype(np.float64),
        repl_dev=sizes["repl_dev"].to_numpy().astype(np.float64),
    )


def expected_lineups(lineup: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Exact expected usage given availability.

    Returns ``(players, replacements)``: ``players`` is ``lineup`` plus
    ``expected_share``; ``replacements`` has one row per lineup with ``repl_share``
    (expected share going to a replacement-level player) and ``repl_dev``.
    """
    packed = pack(lineup)
    expected = np.zeros_like(packed.slot_share)
    repl = np.zeros(packed.sizes.size)
    for i, (start, size) in enumerate(zip(packed.starts, packed.sizes, strict=True)):
        sl = slice(start, start + size)
        expected[sl], repl[i] = expected_places(packed.slot_share[sl], packed.play_prob[sl])
    players = lineup.with_columns(expected_share=pl.Series(expected))
    replacements = packed.keys.with_columns(
        repl_share=pl.Series(repl), repl_dev=pl.Series(packed.repl_dev)
    )
    return players, replacements


def expected_role_values(lineup: pl.DataFrame) -> pl.DataFrame:
    """Expected per-role team value: ``season, week, team, role, value``.

    ``value`` is the expected ``sum(share * dev)`` including the replacement player, the
    same quantity as the Phase 2 features but with injuries taken into account.
    """
    players, repl = expected_lineups(lineup)
    from_players = players.group_by(*_TEAM_WEEK, "role").agg(
        (pl.col("expected_share") * pl.col("dev")).sum().alias("v")
    )
    from_repl = repl.group_by(*_TEAM_WEEK, "role").agg(
        (pl.col("repl_share") * pl.col("repl_dev")).sum().alias("r")
    )
    return (
        from_players.join(from_repl, on=[*_TEAM_WEEK, "role"], how="full", coalesce=True)
        .select(
            *_TEAM_WEEK,
            "role",
            (pl.col("v").fill_null(0.0) + pl.col("r").fill_null(0.0)).alias("value"),
        )
        .sort(*_TEAM_WEEK, "role")
    )


def sample_role_values(
    lineup: pl.DataFrame, n: int, rng: np.random.Generator
) -> tuple[pl.DataFrame, FloatArray]:
    """Sample availability ``n`` times and return each run's per-role team value.

    Returns ``(keys, values)``: ``keys`` is ``season, week, team, role`` (one row per
    team-week-role) and ``values`` has shape ``(len(keys), n)``. Each player's
    availability is drawn independently per run.
    """
    packed = pack(lineup)
    role_keys = packed.keys.select(*_TEAM_WEEK, "role").unique(maintain_order=True)
    index = {k: i for i, k in enumerate(role_keys.iter_rows())}
    rows = [index[k] for k in packed.keys.select(*_TEAM_WEEK, "role").iter_rows()]
    values = np.zeros((role_keys.height, n))
    for i, (start, size) in enumerate(zip(packed.starts, packed.sizes, strict=True)):
        sl = slice(start, start + size)
        s, p, d = packed.slot_share[sl], packed.play_prob[sl], packed.dev[sl]
        avail = rng.random((n, size)) < p
        place = np.cumsum(avail, axis=1) - 1
        share = np.where(avail, s[np.clip(place, 0, size - 1)], 0.0)
        repl = s.sum() - share.sum(axis=1)
        values[rows[i]] += share @ d + repl * packed.repl_dev[i]
    return role_keys, values


def sampled_player_features(
    games: pl.DataFrame, lineup: pl.DataFrame, n: int, rng: np.random.Generator
) -> dict[str, FloatArray]:
    """Per-run player features for each game: availability sampled ``n`` times.

    ``games`` has ``season, week, home_team, away_team`` (current codes). Returns
    ``h_off_qb`` ... ``a_off_recv`` arrays of shape ``(games, n)``. Roles without a
    lineup (no projection) are 0, as in ``role_features``. The mean over runs converges
    to ``expected_role_values``.
    """
    sides = {"h_": "home_team", "a_": "away_team"}
    out = {
        f"{prefix}{feature}": np.zeros((games.height, n))
        for prefix in sides
        for feature in ROLE_FEATURES.values()
    }
    targets = (
        pl.concat([games.select("season", "week", pl.col(c).alias("team")) for c in sides.values()])
        .unique()
        .cast({"season": lineup.schema["season"], "week": lineup.schema["week"]})
    )
    sub = lineup.join(targets, on=_TEAM_WEEK, how="semi")
    if sub.is_empty():
        return out
    keys, values = sample_role_values(sub, n, rng)
    index = {k: i for i, k in enumerate(keys.iter_rows())}
    for prefix, col in sides.items():
        rows = games.select("season", "week", col).iter_rows()
        for g, (season, week, team) in enumerate(rows):
            for role, feature in ROLE_FEATURES.items():
                i = index.get((season, week, team, role))
                if i is not None:
                    out[f"{prefix}{feature}"][g] = values[i]
    return out


def uncertain_players(
    lineup: pl.DataFrame, targets: pl.DataFrame, min_share: float
) -> pl.DataFrame:
    """Players in ``targets`` team-weeks who are on the injury report, may not play
    (``play_prob < 1``) and hold at least ``min_share`` of their role when healthy.

    Regulars not on the report (who also play slightly under 100%) are left out.
    """
    keys = targets.select(*_TEAM_WEEK).cast(
        {"season": lineup.schema["season"], "week": lineup.schema["week"]}
    )
    return (
        lineup.join(keys, on=_TEAM_WEEK, how="semi")
        .filter(
            (pl.col("status") != UNLISTED)
            & (pl.col("play_prob") < 1.0)
            & (pl.col("slot_share") >= min_share)
        )
        .select(
            "team",
            "gsis_id",
            "position",
            "role",
            "slot_share",
            "status",
            "practice",
            "play_prob",
        )
        .sort("team", "position", -pl.col("slot_share"))
    )


def role_features(targets: pl.DataFrame, values: pl.DataFrame, rated: pl.DataFrame) -> pl.DataFrame:
    """Expected per-role values as Phase 2 feature columns (``off_qb`` ...).

    Roles where ``rated`` has no projection count as missing, as in Phase 2.
    """
    missing = rated.group_by(*_TEAM_WEEK, "role").agg(
        pl.col("share").is_null().all().alias("missing")
    )
    per_role = missing.join(values, on=[*_TEAM_WEEK, "role"], how="left").with_columns(
        pl.when(pl.col("missing"))
        .then(0.0)
        .otherwise(pl.col("value").fill_null(0.0))
        .alias("value")
    )
    return role_values_wide(targets, per_role)
