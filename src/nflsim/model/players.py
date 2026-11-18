"""Player ratings by role, and projected usage, from play-by-play.

Roles (see ``PlayerRatingsConfig``):

- ``dropback``: every QB dropback (passes, sacks, scrambles), credited to the passer.
- ``carry``: designed runs (scrambles excluded), credited to the rusher.
- ``target``: pass plays with a named receiver, credited to the receiver.

Ratings are EPA per play in the role, shrunk toward the average of the player's position
group *in that role* (a QB's designed runs are not compared with an RB's). Like the team
ratings, a rating for week W uses only games before week W of that season, with the
previous season as the prior. Only plays inside the [ratings] win-probability window
count, so garbage-time snaps by backups are left out of both ratings and usage.

Usage is a player's share of his team's plays in a role, exponentially weighted over his
recent rostered games (with any team), and renormalized over the players on the team's
roster for the target week.
"""

from __future__ import annotations

import polars as pl

from nflsim.config import PlayerRatingsConfig, PlayerRole, RatingsConfig

ROLES: tuple[PlayerRole, ...] = ("dropback", "carry", "target")
# Position groups with their own average in each role; other positions pool as "other".
ROLE_GROUPS: dict[str, tuple[str, ...]] = {
    "dropback": ("QB",),
    "carry": ("QB", "RB", "WR"),
    "target": ("WR", "TE", "RB"),
}
# Positions whose usage is projected (players outside these are only counted if they
# actually had a role play in a game).
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
# Positions eligible for projected usage in each role. Only QBs are projected to drop
# back: a receiver's trick-play pass must not make him the projected passer.
ROLE_ELIGIBLE: dict[str, tuple[str, ...]] = {
    "dropback": ("QB",),
    "carry": SKILL_POSITIONS,
    "target": SKILL_POSITIONS,
}

_GAME_KEYS = ["season", "week", "game_id", "team"]


def _time(season: pl.Expr, week: pl.Expr) -> pl.Expr:
    """A sortable (season, week) key."""
    return (season * 100 + week).cast(pl.Int32)


def role_plays(pbp: pl.DataFrame | pl.LazyFrame, config: RatingsConfig) -> pl.LazyFrame:
    """One row per (play, role): ``season, week, game_id, team, gsis_id, role, epa``.

    A play can appear twice (a pass is a dropback for the passer and a target for the
    receiver). ``gsis_id`` is null when play-by-play names no player; those rows still
    count toward the team's role totals.
    """
    plays = (
        pbp.lazy()
        .filter(
            pl.col("play_type").is_in(["pass", "run"]),
            pl.col("epa").is_not_null(),
            pl.col("posteam").is_not_null(),
            pl.col("two_point_attempt").fill_null(0) == 0,
            pl.col("wp").fill_null(0.5).is_between(config.min_wp, config.max_wp, closed="both"),
        )
        .select(
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            "game_id",
            pl.col("posteam").alias("team"),
            pl.col("epa").cast(pl.Float64),
            pl.col("play_type"),
            pl.col("qb_dropback").fill_null(0) == 1,
            pl.col("qb_scramble").fill_null(0) == 1,
            pl.col("sack").fill_null(0) == 1,
            pl.coalesce("passer_id", "passer_player_id", "rusher_player_id").alias("passer"),
            pl.coalesce("rusher_player_id", "rusher_id").alias("rusher"),
            pl.col("receiver_player_id").alias("receiver"),
        )
    )
    keep = ["season", "week", "game_id", "team", "epa"]
    dropbacks = plays.filter(pl.col("qb_dropback")).select(
        *keep, pl.col("passer").alias("gsis_id"), pl.lit("dropback").alias("role")
    )
    carries = plays.filter((pl.col("play_type") == "run") & ~pl.col("qb_scramble")).select(
        *keep, pl.col("rusher").alias("gsis_id"), pl.lit("carry").alias("role")
    )
    targets = plays.filter(
        (pl.col("play_type") == "pass") & ~pl.col("sack") & pl.col("receiver").is_not_null()
    ).select(*keep, pl.col("receiver").alias("gsis_id"), pl.lit("target").alias("role"))
    return pl.concat([dropbacks, carries, targets])


def attach_position(frame: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    """Add ``position``: the player's weekly-roster position at (season, week).

    Uses the latest roster week at or before (season, week); if the player has none,
    the earliest one after. Players never on a weekly roster get a null position.
    """
    roster = (
        rosters.filter(pl.col("position").is_not_null())
        .select(_time(pl.col("season"), pl.col("week")).alias("_t"), "gsis_id", "position")
        .unique(["_t", "gsis_id"])
        .sort("_t")
    )
    out = frame.with_columns(_t=_time(pl.col("season"), pl.col("week"))).sort("_t")
    back = out.join_asof(roster, on="_t", by="gsis_id", strategy="backward", check_sortedness=False)
    fwd = out.join_asof(
        roster, on="_t", by="gsis_id", strategy="forward", check_sortedness=False
    ).select(pl.col("position").alias("_fwd"))
    return (
        back.with_columns(fwd)
        .with_columns(pl.coalesce("position", "_fwd").alias("position"))
        .drop("_t", "_fwd")
    )


def _group(role: pl.Expr, position: pl.Expr) -> pl.Expr:
    """Position group used for the role average ("other" outside ``ROLE_GROUPS``)."""
    expr = pl.lit("other")
    for r, groups in ROLE_GROUPS.items():
        expr = pl.when((role == r) & position.is_in(list(groups))).then(position).otherwise(expr)
    return expr


def player_game_stats(
    pbp: pl.DataFrame | pl.LazyFrame, rosters: pl.DataFrame, config: RatingsConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per player-game role sums and per team-game role totals.

    Returns ``(players, teams)``. ``players`` has one row per (game, team, player, role)
    with ``n`` plays, ``epa`` sum, ``position`` and ``group``. ``teams`` has one row per
    (game, team, role) with ``team_n``, counting plays with no named player too.
    ``rosters`` is the output of ``nflsim.data.roster_weeks``.
    """
    plays = role_plays(pbp, config)
    teams = (
        plays.group_by(*_GAME_KEYS, "role")
        .agg(pl.len().cast(pl.Float64).alias("team_n"))
        .sort(*_GAME_KEYS, "role")
        .collect()
    )
    players = (
        plays.filter(pl.col("gsis_id").is_not_null())
        .group_by(*_GAME_KEYS, "gsis_id", "role")
        .agg(pl.len().cast(pl.Float64).alias("n"), pl.col("epa").sum())
        .collect()
    )
    players = attach_position(players, rosters).with_columns(
        _group(pl.col("role"), pl.col("position")).alias("group")
    )
    return players.sort(*_GAME_KEYS, "gsis_id", "role"), teams


def player_ratings_as_of(
    player_games: pl.DataFrame, targets: pl.DataFrame, config: PlayerRatingsConfig
) -> pl.DataFrame:
    """Rate each (season, week, gsis_id, role, group) in ``targets`` from earlier games.

    ``player_games`` is ``player_game_stats(...)[0]``. ``group`` in ``targets`` is the
    player's position group at the target week (see ``ROLE_GROUPS``). Returns the targets
    with ``rating``, ``group_mean`` (the group's league average: last season pooled with
    this season through week W-1), ``cur_n``/``cur_epa`` (this season, before week W),
    ``prev_n`` (last season's plays in the role) and ``prior_source`` ("prior_season" or
    "unknown"). ``rating`` is null only when the group has no data at all.
    """
    targets = targets.select("season", "week", "gsis_id", "role", "group").unique()
    k = pl.col("role").replace_strict(config.prior_plays, return_dtype=pl.Float64)
    unknown = pl.col("role").replace_strict(config.unknown_deviation, return_dtype=pl.Float64)

    # Group averages: this season through W-1 plus all of last season.
    group_week = (
        player_games.group_by("season", "week", "role", "group")
        .agg(pl.col("n").sum(), pl.col("epa").sum())
        .sort("season", "week")
        .with_columns(pl.col("n", "epa").cum_sum().over("season", "role", "group"))
        .sort("week")
    )
    group_season = player_games.group_by("season", "role", "group").agg(
        pl.col("n").sum().alias("g_n"), pl.col("epa").sum().alias("g_epa")
    )
    group_prev = group_season.with_columns(pl.col("season") + 1).rename(
        {"g_n": "gprev_n", "g_epa": "gprev_epa"}
    )

    # The player's own totals this season through W-1 (with any team).
    player_week = (
        player_games.group_by("season", "week", "gsis_id", "role")
        .agg(pl.col("n").sum().alias("cur_n"), pl.col("epa").sum().alias("cur_epa"))
        .sort("season", "week")
        .with_columns(pl.col("cur_n", "cur_epa").cum_sum().over("season", "gsis_id", "role"))
        .sort("week")
    )

    # Last season: plays and EPA above the average of the group each play was made in.
    prev = (
        player_games.join(group_season, on=["season", "role", "group"], how="left")
        .group_by("season", "gsis_id", "role")
        .agg(
            pl.col("n").sum().alias("prev_n"),
            (pl.col("epa") - pl.col("n") * pl.col("g_epa") / pl.col("g_n"))
            .sum()
            .alias("prev_excess"),
        )
        .with_columns(pl.col("season") + 1)
    )

    sorted_targets = targets.sort("week")
    frame = (
        sorted_targets.join_asof(
            player_week,
            on="week",
            by=["season", "gsis_id", "role"],
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .join_asof(
            group_week.rename({"n": "gcur_n", "epa": "gcur_epa"}),
            on="week",
            by=["season", "role", "group"],
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .join(group_prev, on=["season", "role", "group"], how="left")
        .join(prev, on=["season", "gsis_id", "role"], how="left")
        .with_columns(
            pl.col("cur_n", "cur_epa", "gcur_n", "gcur_epa", "gprev_n", "gprev_epa").fill_null(0.0),
            pl.col("prev_n", "prev_excess").fill_null(0.0),
        )
    )
    g_n = pl.col("gcur_n") + pl.col("gprev_n")
    group_mean = pl.when(g_n > 0).then((pl.col("gcur_epa") + pl.col("gprev_epa")) / g_n)
    deviation = config.carryover * pl.col("prev_excess") / (pl.col("prev_n") + k) + unknown * k / (
        pl.col("prev_n") + k
    )
    prior = pl.col("group_mean") + deviation
    return (
        frame.with_columns(group_mean.alias("group_mean"))
        .with_columns(((pl.col("cur_epa") + k * prior) / (pl.col("cur_n") + k)).alias("rating"))
        .with_columns(
            pl.when(pl.col("prev_n") > 0)
            .then(pl.lit("prior_season"))
            .otherwise(pl.lit("unknown"))
            .alias("prior_source")
        )
        .select(
            "season",
            "week",
            "gsis_id",
            "role",
            "group",
            "rating",
            "group_mean",
            "cur_n",
            "cur_epa",
            "prev_n",
            "prior_source",
        )
        .sort("season", "week", "gsis_id", "role")
    )


def _eligible(role: pl.Expr, position: pl.Expr) -> pl.Expr:
    expr = pl.lit(False)
    for r, positions in ROLE_ELIGIBLE.items():
        expr = expr | ((role == r) & position.is_in(list(positions)))
    return expr


def _available(config: PlayerRatingsConfig) -> pl.Expr:
    return ~pl.col("status").fill_null("").is_in(list(config.unavailable_statuses))


def usage_history(
    player_games: pl.DataFrame,
    team_games: pl.DataFrame,
    rosters: pl.DataFrame,
    config: PlayerRatingsConfig,
) -> pl.DataFrame:
    """Each player's role share in every team game he was available for, smoothed.

    One row per (player, team game, role): ``share`` (his plays / team plays) and
    ``ew_share``, the exponentially weighted mean of ``share`` over this and all his
    earlier rows, across seasons and teams. Rows come from skill-position players on the
    team's weekly roster (not on an unavailable list) plus anyone who had a role play.
    """
    games = team_games.select(*_GAME_KEYS).unique()
    rostered = (
        rosters.filter(pl.col("position").is_in(list(SKILL_POSITIONS)) & _available(config))
        .select("season", "week", "team", "gsis_id")
        .join(games, on=["season", "week", "team"], how="inner")
    )
    played = player_games.select(*_GAME_KEYS, "gsis_id").unique()
    members = pl.concat([rostered.select(*_GAME_KEYS, "gsis_id"), played]).unique()
    roles = pl.DataFrame({"role": list(ROLES)})
    frame = (
        members.join(roles, how="cross")
        .join(team_games, on=[*_GAME_KEYS, "role"], how="left")
        .join(
            player_games.select(*_GAME_KEYS, "gsis_id", "role", "n"),
            on=[*_GAME_KEYS, "gsis_id", "role"],
            how="left",
        )
        .with_columns(
            pl.col("n").fill_null(0.0),
            pl.col("team_n").fill_null(0.0),
            _time(pl.col("season"), pl.col("week")).alias("t"),
        )
        .filter(pl.col("team_n") > 0)
        .with_columns(share=pl.col("n") / pl.col("team_n"))
        .sort("gsis_id", "role", "t", "game_id")
    )
    smoothed = [
        frame.filter(pl.col("role") == role).with_columns(
            pl.col("share")
            .ewm_mean(half_life=config.usage_half_life_games[role], adjust=True)
            .over("gsis_id")
            .alias("ew_share")
        )
        for role in ROLES
    ]
    return (
        pl.concat(smoothed)
        .select(*_GAME_KEYS, "t", "gsis_id", "role", "share", "ew_share")
        .sort("gsis_id", "role", "t", "game_id")
    )


def projected_usage(
    history: pl.DataFrame,
    rosters: pl.DataFrame,
    targets: pl.DataFrame,
    config: PlayerRatingsConfig,
) -> pl.DataFrame:
    """Projected role shares for each (season, week, team) in ``targets``.

    Candidates are players on the team's weekly roster for that week who are not on an
    unavailable list and whose position is eligible for the role (``ROLE_ELIGIBLE``).
    Each candidate's smoothed share from games before that week (``usage_history``) is
    renormalized so a team's shares in a role sum to 1. Returns ``season, week, team,
    gsis_id, role, share, raw_share, has_history``. ``has_history`` is False for players
    never rostered or seen before (rookies, new signings); they project to 0. When no
    candidate has history in a role, shares are null for that team-week-role.
    """
    candidates = (
        targets.select("season", "week", "team")
        .unique()
        .join(
            rosters.filter(
                pl.col("position").is_in(list(SKILL_POSITIONS)) & _available(config)
            ).select("season", "week", "team", "gsis_id", "position"),
            on=["season", "week", "team"],
            how="inner",
        )
        .join(pl.DataFrame({"role": list(ROLES)}), how="cross")
        .filter(_eligible(pl.col("role"), pl.col("position")))
        .drop("position")
        .with_columns(_time(pl.col("season"), pl.col("week")).alias("t"))
        .sort("t")
    )
    last = (
        history.sort("t", "game_id")
        .unique(["gsis_id", "role", "t"], keep="last", maintain_order=True)
        .select("gsis_id", "role", "t", "ew_share")
    )
    raw = candidates.join_asof(
        last,
        on="t",
        by=["gsis_id", "role"],
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    team_key = ["season", "week", "team", "role"]
    total = pl.col("raw_share").sum().over(team_key)
    return (
        raw.with_columns(
            pl.col("ew_share").is_not_null().alias("has_history"),
            pl.col("ew_share").fill_null(0.0).alias("raw_share"),
        )
        .with_columns(pl.when(total > 0).then(pl.col("raw_share") / total).alias("share"))
        .select("season", "week", "team", "gsis_id", "role", "share", "raw_share", "has_history")
        .sort("season", "week", "team", "role", "gsis_id")
    )


# Points-model feature name for each role's team rating.
ROLE_FEATURES: dict[PlayerRole, str] = {
    "dropback": "off_qb",
    "carry": "off_rush",
    "target": "off_recv",
}
TEAM_PLAYER_COLUMNS = [*ROLE_FEATURES.values(), "player_proj_missing"]


def role_means_as_of(player_games: pl.DataFrame, keys: pl.DataFrame) -> pl.DataFrame:
    """League EPA per play in each role (all positions pooled) for each (season, week).

    Pools last season with this season through week W-1, like the group averages in
    ``player_ratings_as_of``. Returns ``season, week, role, role_mean`` (null when there
    is no data at all).
    """
    weekly = (
        player_games.group_by("season", "week", "role")
        .agg(pl.col("n").sum(), pl.col("epa").sum())
        .sort("season", "week")
        .with_columns(pl.col("n", "epa").cum_sum().over("season", "role"))
        .sort("week")
    )
    prev = (
        player_games.group_by("season", "role")
        .agg(pl.col("n").sum().alias("p_n"), pl.col("epa").sum().alias("p_epa"))
        .with_columns(pl.col("season") + 1)
    )
    frame = (
        keys.select("season", "week")
        .unique()
        .join(pl.DataFrame({"role": list(ROLES)}), how="cross")
        .sort("week")
        .join_asof(
            weekly,
            on="week",
            by=["season", "role"],
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .join(prev, on=["season", "role"], how="left")
        .with_columns(pl.col("n", "epa", "p_n", "p_epa").fill_null(0.0))
    )
    total_n = pl.col("n") + pl.col("p_n")
    return frame.select(
        "season",
        "week",
        "role",
        pl.when(total_n > 0).then((pl.col("epa") + pl.col("p_epa")) / total_n).alias("role_mean"),
    )


def rated_usage(
    player_games: pl.DataFrame,
    team_games: pl.DataFrame,
    rosters: pl.DataFrame,
    targets: pl.DataFrame,
    config: PlayerRatingsConfig,
) -> pl.DataFrame:
    """Projected usage of each candidate player with his rating, for ``targets`` team-weeks.

    One row per (season, week, team, gsis_id, role) from ``projected_usage``, plus
    ``position``, ``group``, ``rating`` and ``group_mean`` (``player_ratings_as_of``),
    ``role_mean`` (``role_means_as_of``) and ``dev``: rating minus the league role
    average (0 when the rating is unknown), the EPA per role play he adds.
    """
    targets = targets.select("season", "week", "team").unique()
    history = usage_history(player_games, team_games, rosters, config)
    usage = projected_usage(history, rosters, targets, config)
    positions = rosters.select("season", "week", "gsis_id", "position").unique(
        ["season", "week", "gsis_id"]
    )
    usage = usage.join(positions, on=["season", "week", "gsis_id"], how="left").with_columns(
        _group(pl.col("role"), pl.col("position")).alias("group")
    )
    rated = player_ratings_as_of(player_games, usage, config).select(
        "season", "week", "gsis_id", "role", "group", "rating", "group_mean"
    )
    means = role_means_as_of(player_games, targets)
    return (
        usage.join(rated, on=["season", "week", "gsis_id", "role", "group"], how="left")
        .join(means, on=["season", "week", "role"], how="left")
        .with_columns(dev=(pl.col("rating") - pl.col("role_mean")).fill_null(0.0))
        .sort("season", "week", "team", "role", "gsis_id")
    )


def team_player_features(
    player_games: pl.DataFrame,
    team_games: pl.DataFrame,
    rosters: pl.DataFrame,
    targets: pl.DataFrame,
    config: PlayerRatingsConfig,
) -> pl.DataFrame:
    """Team offense ratings built from players, for each (season, week, team) in ``targets``.

    For each role, ``sum(projected share * (player rating - league role average))``: the
    usage-weighted EPA per role play the team's players are expected to add over an
    average player. Everything uses games before the target week only. When no rostered
    player has usage history in a role (shares null), the feature is 0 (league average)
    and counted in ``player_proj_missing`` (number of roles without a projection).

    Returns ``season, week, team`` plus ``TEAM_PLAYER_COLUMNS``.
    """
    targets = targets.select("season", "week", "team").unique()
    usage = rated_usage(player_games, team_games, rosters, targets, config)
    per_role = usage.group_by("season", "week", "team", "role").agg(
        (pl.col("share") * pl.col("dev")).fill_null(0.0).sum().alias("value"),
        pl.col("share").is_null().all().alias("missing"),
    )
    return role_values_wide(targets, per_role)


def role_values_wide(targets: pl.DataFrame, per_role: pl.DataFrame) -> pl.DataFrame:
    """Pivot per-role ``value``/``missing`` rows to ``season, week, team`` + feature columns.

    Team-weeks or roles absent from ``per_role`` count as missing with value 0.
    """
    targets = targets.select("season", "week", "team").unique()
    full = (
        targets.join(pl.DataFrame({"role": list(ROLES)}), how="cross")
        .join(per_role, on=["season", "week", "team", "role"], how="left")
        .with_columns(pl.col("value").fill_null(0.0), pl.col("missing").fill_null(True))
    )
    wide = full.pivot(on="role", index=["season", "week", "team"], values="value").rename(
        {role: feature for role, feature in ROLE_FEATURES.items()}
    )
    missing = full.group_by("season", "week", "team").agg(
        pl.col("missing").sum().cast(pl.Int64).alias("player_proj_missing")
    )
    return (
        wide.join(missing, on=["season", "week", "team"], how="left")
        .select("season", "week", "team", *TEAM_PLAYER_COLUMNS)
        .sort("season", "week", "team")
    )
