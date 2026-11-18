"""Scenario overrides: "he's out", "he plays" (Phase 7, step 2).

Names are matched to the two teams' rosters for the simulated week (or the latest roster
week before it). An override replaces the player's chance to play in the injury-aware
lineup, so backups move up exactly as for a real injury, and the matchup's expected
player features are recomputed from the changed lineup.

Only players in a projected lineup (QB, RB, WR, TE with a usage projection) can change a
prediction; an override on anyone else is reported as having no effect, never silently
dropped.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

import polars as pl

from nflsim.config import InjuriesConfig
from nflsim.model.availability import expected_role_values
from nflsim.model.players import ROLE_FEATURES

OverrideKind = Literal["out", "doubtful", "plays"]
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_GSIS = re.compile(r"^00-\d{7}$")


@dataclass(frozen=True)
class Override:
    kind: OverrideKind
    query: str


@dataclass(frozen=True)
class ResolvedOverride:
    kind: OverrideKind
    query: str
    gsis_id: str
    name: str
    team: str
    position: str | None


def normalize_name(name: str) -> str:
    """Lowercase, no accents or punctuation, suffixes (Jr., II, ...) dropped."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace(".", "")).split()
    return " ".join(w for w in words if w not in _SUFFIXES)


def week_roster(
    rosters: pl.DataFrame, season: int, week: int, teams: tuple[str, str]
) -> pl.DataFrame:
    """Each team's roster for ``week``, or its latest roster week before it.

    ``rosters`` comes from ``nflsim.data.roster_weeks``. Returns ``gsis_id, name, team,
    position``.
    """
    frames = []
    for team in teams:
        own = rosters.filter(
            (pl.col("season") == season) & (pl.col("team") == team) & (pl.col("week") <= week)
        )
        if own.is_empty():
            continue
        latest = own.select(pl.col("week").max()).item()
        frames.append(own.filter(pl.col("week") == latest))
    if not frames:
        raise ValueError(f"no {season} roster for {' or '.join(teams)} up to week {week}")
    # Sorted so messages (e.g. ambiguous candidates) come out in the same order every run.
    return (
        pl.concat(frames)
        .select("gsis_id", "name", "team", "position")
        .unique("gsis_id")
        .sort("team", "name", "gsis_id")
    )


def resolve(overrides: list[Override], roster: pl.DataFrame) -> list[ResolvedOverride]:
    """Match each override to exactly one rostered player, or raise ``ValueError``.

    Tries, in order: a GSIS ID; the full name; the last name. Comparisons ignore case,
    accents, punctuation and suffixes. Several matches, or none, is an error that lists
    the candidates. The same player in two overrides is an error too.
    """
    keyed = roster.with_columns(
        pl.col("name").map_elements(normalize_name, return_dtype=pl.String).alias("_full"),
        pl.col("name")
        .map_elements(lambda n: normalize_name(n).split(" ")[-1], return_dtype=pl.String)
        .alias("_last"),
    )
    out: list[ResolvedOverride] = []
    for o in overrides:
        query = o.query.strip()
        if _GSIS.match(query):
            hits = keyed.filter(pl.col("gsis_id") == query)
        else:
            q = normalize_name(query)
            hits = keyed.filter(pl.col("_full") == q)
            if hits.is_empty() and " " not in q:
                hits = keyed.filter(pl.col("_last") == q)
        if hits.height == 0:
            raise ValueError(f"no player matching {o.query!r} on either roster")
        if hits.height > 1:
            names = "; ".join(
                f"{r['name']} ({r['team']} {r['position']}, {r['gsis_id']})"
                for r in hits.iter_rows(named=True)
            )
            raise ValueError(f"{o.query!r} matches several players: {names}")
        r = hits.row(0, named=True)
        out.append(
            ResolvedOverride(o.kind, o.query, r["gsis_id"], r["name"], r["team"], r["position"])
        )
    seen = [r.gsis_id for r in out]
    dupes = {g for g in seen if seen.count(g) > 1}
    if dupes:
        raise ValueError(f"the same player is overridden more than once: {sorted(dupes)}")
    return out


def override_probability(kind: OverrideKind, config: InjuriesConfig) -> float:
    """Chance to play for an override: 0 (out), 1 (plays) or the doubtful rate."""
    if kind == "out":
        return 0.0
    if kind == "plays":
        return 1.0
    return float(config.play_probability_regular["doubtful"]["none"])


def _qb_change_updates(
    info: dict[str, object],
    sides: dict[str, str],
    resolved: list[ResolvedOverride],
    updates: dict[str, float],
) -> dict[str, float]:
    """New ``qb_change``/``qb_change_proj`` values for the game row after overrides."""
    out: dict[str, float] = {}
    for team, side in sides.items():
        proj, old_off, new_off = (
            info.get(f"{side}qb_change_proj"),
            info.get(f"{side}off_qb"),
            updates.get(f"{side}off_qb"),
        )
        if not isinstance(proj, float) or not isinstance(old_off, float) or new_off is None:
            continue
        new_proj = proj + new_off - old_off
        out[f"{side}qb_change_proj"] = new_proj
        starter = info.get(f"{side}qb_starter_id")
        starter_out = any(
            r.team == team and r.kind in ("out", "doubtful") and r.gsis_id == starter
            for r in resolved
        )
        if starter_out or not info.get(f"{side}qb_starter_listed"):
            out[f"{side}qb_change"] = new_proj
    return out


def apply_overrides(
    games: pl.DataFrame,
    game_id: str,
    lineup: pl.DataFrame,
    resolved: list[ResolvedOverride],
    config: InjuriesConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Apply overrides to the lineup of the game ``game_id`` in ``games``.

    Returns ``(games, lineup, effects)``: the game's ``h_``/``a_`` player features
    recomputed as expected values of the changed lineup, the changed lineup, and one row
    per override with ``before``/``after`` chance to play (null when the player has no
    projected role, i.e. the override cannot change the prediction).

    Quarterback change (``qb_change_proj``) moves with the projected QB value. ``qb_change``
    keeps the listed starter, unless he is overridden out or doubtful (or no starter is
    listed); then it is the override-adjusted projection, i.e. his backup.
    """
    row = games.filter(pl.col("game_id") == game_id)
    if row.height != 1:
        raise ValueError(f"games must contain exactly one row for {game_id}")
    info = row.row(0, named=True)
    season, week = int(info["season"]), int(info["week"])
    sides = {info["home_team"]: "h_", info["away_team"]: "a_"}

    in_game = (pl.col("season") == season) & (pl.col("week") == week)
    effects = []
    new_lineup = lineup
    for r in resolved:
        target = in_game & (pl.col("team") == r.team) & (pl.col("gsis_id") == r.gsis_id)
        current = new_lineup.filter(target)
        after = override_probability(r.kind, config)
        before = float(current["play_prob"][0]) if not current.is_empty() else None
        effects.append(
            {
                "team": r.team,
                "player": r.name,
                "position": r.position,
                "override": r.kind,
                "before": before,
                "after": after if before is not None else None,
            }
        )
        if before is not None:
            new_lineup = new_lineup.with_columns(
                pl.when(target)
                .then(pl.lit(after))
                .otherwise(pl.col("play_prob"))
                .alias("play_prob")
            )

    game_lineup = new_lineup.filter(in_game & pl.col("team").is_in(list(sides)))
    updates: dict[str, float] = {}
    if not game_lineup.is_empty():
        for v in expected_role_values(game_lineup).iter_rows(named=True):
            updates[f"{sides[v['team']]}{ROLE_FEATURES[v['role']]}"] = float(v["value"])
    updates |= _qb_change_updates(info, sides, resolved, updates)
    is_game = pl.col("game_id") == game_id
    new_games = games.with_columns(
        pl.when(is_game).then(pl.lit(value)).otherwise(pl.col(col)).alias(col)
        for col, value in updates.items()
        if col in games.columns
    )
    effect_frame = pl.DataFrame(
        effects,
        schema={
            "team": pl.String,
            "player": pl.String,
            "position": pl.String,
            "override": pl.String,
            "before": pl.Float64,
            "after": pl.Float64,
        },
    )
    return new_games, new_lineup, effect_frame
