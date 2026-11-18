"""Glue between the data cache and the models: build the game table and predict matchups."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from nflsim.config import MARKET_FEATURES, PLAYER_FEATURES, QB_CHANGE_FEATURES, Settings
from nflsim.data import (
    DataCache,
    depth_chart_weeks,
    injury_reports,
    player_snaps,
    roster_weeks,
)
from nflsim.data.teams import TEAM_ALIASES, normalize_teams
from nflsim.model import (
    PointsModel,
    build_games,
    fit_points_model,
    simulate_scores,
    team_game_stats,
)
from nflsim.model.availability import (
    depth_ranks,
    expected_lineups,
    expected_role_values,
    lineups,
    player_availability,
    role_features,
    sampled_player_features,
    uncertain_players,
)
from nflsim.model.players import player_game_stats, rated_usage, team_player_features
from nflsim.model.qb_change import qb_change_features
from nflsim.model.stat_lines import (
    expected_stats,
    player_rates_as_of,
    team_volume_as_of,
)

MATCHUP_ID = "SIM"
# Second entry of the seed sequence for availability draws, so they never share a
# random stream with the score draws (which use the seed alone).
AVAILABILITY_STREAM = 1


def availability_rng(seed: int) -> np.random.Generator:
    """Random generator for availability draws, derived from the run's ``seed``."""
    return np.random.default_rng([seed, AVAILABILITY_STREAM])


def availability_runs(
    games: pl.DataFrame,
    model: PointsModel,
    lineup: pl.DataFrame | None,
    settings: Settings,
    n: int,
    seed: int,
) -> dict[str, np.ndarray] | None:
    """Per-run player features for ``games`` when availability is sampled, else None.

    Sampled when lineups exist (``model.injury_aware``), ``sim.sample_availability`` is
    set and the fitted model uses player features.
    """
    if (
        lineup is None
        or not settings.sim.sample_availability
        or not set(model.features) & set(PLAYER_FEATURES)
    ):
        return None
    return sampled_player_features(games, lineup, n, availability_rng(seed))


def load_games(
    cache: DataCache, settings: Settings, last_season: int, extra: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Build the game table (see ``build_games``) for every cached season through ``last_season``.

    ``extra`` holds schedule-shaped rows to add, e.g. a hypothetical matchup.
    """
    return load_games_with_lineups(cache, settings, last_season, extra)[0]


def load_games_with_lineups(
    cache: DataCache, settings: Settings, last_season: int, extra: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """``load_games`` plus the injury-aware lineups behind the player features.

    The lineups (``nflsim.model.availability.lineups``) are None unless player features
    are used and ``model.injury_aware`` is set. They let callers sample availability
    per run (``availability_runs``).
    """
    seasons = list(range(settings.data.default_first_season, last_season + 1))
    pbp = cache.scan("pbp", seasons)
    schedules = cache.load("schedules", seasons)
    if extra is not None:
        schedules = pl.concat([schedules, home_roof(extra, schedules)], how="diagonal_relaxed")
    team_games = team_game_stats(pbp, settings.ratings)
    players = lineup = None
    if any(f in (*PLAYER_FEATURES, *QB_CHANGE_FEATURES) for f in settings.model.features):
        players, lineup = load_player_features(cache, settings, seasons, schedules)
    games = build_games(
        schedules, team_games, settings.ratings, players, settings.context, settings.market
    )
    return games, lineup


def load_player_features(
    cache: DataCache, settings: Settings, seasons: list[int], schedules: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Player-based team offense ratings for each scheduled team.

    The Phase 2 projection (``team_player_features``), or with ``model.injury_aware``
    injury-aware lineups (``injury_aware_features``). Returns ``(features, lineup)``;
    ``lineup`` is None for the Phase 2 projection.
    """
    rosters = roster_weeks(cache.load("rosters_weekly", seasons), cache.load("players"))
    player_games, role_totals = player_game_stats(
        cache.scan("pbp", seasons), rosters, settings.ratings
    )
    targets = schedule_targets(schedules)
    lineup = None
    if not settings.model.injury_aware:
        features = team_player_features(
            player_games, role_totals, rosters, targets, settings.player_ratings
        )
    else:
        rated, lineup = load_lineups(
            cache, settings, seasons, targets, player_games, role_totals, rosters
        )
        reports = injury_reports(cache.load("injuries", seasons))
        features = injury_aware_features(targets, rated, lineup, reports)
    if set(settings.model.features) & set(QB_CHANGE_FEATURES):
        qb = qb_change_features(player_games, schedules, targets, features, settings.player_ratings)
        features = features.cast({"season": pl.Int32, "week": pl.Int32}).join(
            qb.drop("qb_base_source"), on=["season", "week", "team"], how="left"
        )
    return features, lineup


def injury_aware_features(
    targets: pl.DataFrame, rated: pl.DataFrame, lineup: pl.DataFrame, reports: pl.DataFrame
) -> pl.DataFrame:
    """Player features from the expected values of injury-aware lineups (``load_lineups``),
    plus ``injury_report``: whether the team has an injury report that week."""
    reported = (
        reports.select(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), "team")
        .unique()
        .with_columns(injury_report=pl.lit(True))
    )
    features = role_features(targets, expected_role_values(lineup), rated)
    return features.join(reported, on=["season", "week", "team"], how="left").with_columns(
        pl.col("injury_report").fill_null(False)
    )


def schedule_targets(schedules: pl.DataFrame) -> pl.DataFrame:
    """``season, week, team`` for both teams of every scheduled game (current codes)."""
    sched = normalize_teams(schedules, ["home_team", "away_team"])
    return pl.concat(
        [
            sched.select(
                pl.col("season").cast(pl.Int32),
                pl.col("week").cast(pl.Int32),
                pl.col(c).alias("team"),
            )
            for c in ("home_team", "away_team")
        ]
    ).unique()


def load_lineups(
    cache: DataCache,
    settings: Settings,
    seasons: list[int],
    targets: pl.DataFrame,
    player_games: pl.DataFrame,
    role_totals: pl.DataFrame,
    rosters: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Rated usage (Phase 2) and injury-aware lineups for ``targets`` team-weeks.

    Returns ``(rated, lineup)``: ``rated`` from ``rated_usage`` and ``lineup`` from
    ``nflsim.model.availability.lineups``. Injury reports, snap counts and depth charts
    are loaded for ``seasons`` (snap counts from one season earlier, for week 1).
    """
    rated = rated_usage(player_games, role_totals, rosters, targets, settings.player_ratings)
    first_snaps = max(min(seasons) - 1, cache.spec("snap_counts").first_season or 0)
    players = cache.load("players")
    snaps = player_snaps(cache.load("snap_counts", range(first_snaps, max(seasons) + 1)), players)
    reports = injury_reports(cache.load("injuries", seasons))
    depth = depth_chart_weeks(cache.load("depth_charts", seasons), cache.load("schedules", seasons))
    candidates = rated.select("season", "week", "team", "gsis_id", "position").unique()
    available = player_availability(candidates, reports, snaps, settings.injuries)
    lineup = lineups(
        rated, available, depth_ranks(depth), settings.injuries, settings.player_ratings
    )
    return rated, lineup


def home_roof(extra: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """Fill a hypothetical game's missing ``roof`` from the home team's latest home game.

    Sets ``roof_inferred`` on the rows filled this way. Neutral-site games are left
    unknown (treated as outdoors), since the stadium is not the home team's.
    """
    if "roof" in extra.columns and extra["roof"].null_count() == 0:
        return extra
    sched = normalize_teams(schedules, ["home_team", "away_team"])
    latest = (
        sched.filter((pl.col("location") == "Home") & pl.col("roof").is_not_null())
        .sort("season", "week")
        .group_by("home_team")
        .agg(pl.col("roof").last().alias("_roof"))
    )
    base = extra if "roof" in extra.columns else extra.with_columns(roof=pl.lit(None, pl.String))
    filled = base.join(latest, on="home_team", how="left").with_columns(
        roof_inferred=pl.col("roof").is_null()
        & pl.col("_roof").is_not_null()
        & (pl.col("location") != "Neutral")
    )
    return filled.with_columns(
        pl.when(pl.col("roof_inferred"))
        .then(pl.col("_roof"))
        .otherwise(pl.col("roof"))
        .alias("roof")
    ).drop("_roof")


def expected_stat_lines(
    box: pl.DataFrame,
    positions: pl.DataFrame,
    lineup: pl.DataFrame,
    targets: pl.DataFrame,
    settings: Settings,
) -> pl.DataFrame:
    """Expected stats for every player in the injury-aware ``lineup`` of ``targets``
    team-weeks (``nflsim.model.stat_lines``). ``box`` from ``box_scores``; ``positions``
    is ``season, week, gsis_id, position`` (``nflsim.data.roster_weeks``)."""
    keys = targets.select("season", "week", "team").unique()
    players, _ = expected_lineups(lineup.join(keys, on=["season", "week", "team"], how="semi"))
    shares = players.select(
        "season", "week", "team", "gsis_id", "role", pl.col("expected_share").alias("share")
    )
    player_keys = (
        shares.select("season", "week", "gsis_id")
        .unique()
        .join(positions, on=["season", "week", "gsis_id"], how="left")
    )
    volume = team_volume_as_of(box, keys, settings.stat_lines)
    rates = player_rates_as_of(box, positions, player_keys, settings.stat_lines)
    return expected_stats(shares, volume, rates)


def matchup_row(season: int, week: int, home: str, away: str, neutral: bool) -> pl.DataFrame:
    """A schedule row for a hypothetical (or upcoming) game with no score yet."""
    home, away = home.upper(), away.upper()
    if home == away:
        raise ValueError("home and away must differ")
    return pl.DataFrame(
        {
            "game_id": [MATCHUP_ID],
            "season": [season],
            "week": [week],
            "game_type": [MATCHUP_ID],
            "home_team": [TEAM_ALIASES.get(home, home)],
            "away_team": [TEAM_ALIASES.get(away, away)],
            "home_score": [None],
            "away_score": [None],
            "location": ["Neutral" if neutral else "Home"],
            "spread_line": [None],
            "total_line": [None],
            "home_moneyline": [None],
            "away_moneyline": [None],
        },
        schema_overrides={
            "season": pl.Int32,
            "week": pl.Int32,
            "home_score": pl.Int64,
            "away_score": pl.Int64,
            "spread_line": pl.Float64,
            "total_line": pl.Float64,
            "home_moneyline": pl.Float64,
            "away_moneyline": pl.Float64,
        },
    )


def _no_runs() -> np.ndarray:
    return np.zeros(0, dtype=np.int64)


@dataclass(frozen=True)
class MatchupResult:
    home: str
    away: str
    season: int
    week: int
    n: int
    seed: int
    home_win_prob: float
    mean_margin: float
    mean_total: float
    expected_home: float
    expected_away: float
    margin_quantiles: dict[float, float]
    home_quantiles: dict[float, float]
    away_quantiles: dict[float, float]
    flags: list[str]
    model: PointsModel
    sampled_availability: bool = False
    # Simulated scores, one per run (line probabilities, summaries).
    home_scores: np.ndarray = field(default_factory=_no_runs, repr=False, compare=False)
    away_scores: np.ndarray = field(default_factory=_no_runs, repr=False, compare=False)


QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def predict_matchup(
    games: pl.DataFrame,
    settings: Settings,
    n: int,
    seed: int,
    lineup: pl.DataFrame | None = None,
) -> MatchupResult:
    """Simulate the ``MATCHUP_ID`` row of ``games``, fitting on seasons before its season.

    With ``lineup`` (from ``load_games_with_lineups``), who plays is drawn per run when
    ``sim.sample_availability`` is set (``availability_runs``).
    """
    row = games.filter(pl.col("game_id") == MATCHUP_ID)
    if row.height != 1:
        raise ValueError("games must contain exactly one matchup row")
    info = row.row(0, named=True)
    season = int(info["season"])
    k = settings.model.train_seasons
    train = games.filter(
        pl.col("season").is_between(season - k, season - 1) & pl.col("home_score").is_not_null()
    )
    if train.is_empty():
        raise ValueError(f"no completed seasons before {season} to fit the model on")

    others = games.filter(pl.col("game_id") != MATCHUP_ID)
    known = set(others["home_team"].to_list()) | set(others["away_team"].to_list())
    unknown = [t for t in (info["home_team"], info["away_team"]) if t not in known]
    if unknown:
        raise ValueError(f"unknown team abbreviation(s): {', '.join(unknown)}")

    flags: list[str] = []
    for prefix, name in (("h_", info["home_team"]), ("a_", info["away_team"])):
        if info[f"{prefix}off_epa"] is None:
            raise ValueError(
                f"no ratings for {name!r} before {season} week {info['week']}: "
                "unknown team abbreviation, or no data cached for that time"
            )
        if info[f"{prefix}prior_source"] == "league":
            flags.append(f"{name}: previous season not cached; prior is league average")
        if info[f"{prefix}games_played"] == 0:
            flags.append(f"{name}: no games yet this season; rating is the prior only")
        if info.get(f"{prefix}injury_report") is False:
            flags.append(
                f"{name}: no injury report for week {info['week']}; "
                "everyone treated as not on the report"
            )
        missing = info.get(f"{prefix}player_proj_missing")
        if missing:
            flags.append(
                f"{name}: no usage projection for {missing} player role(s) "
                "(no roster or no history); those use the league average"
            )

    used = set(settings.model.features)
    if "qb_change" in used:
        for prefix, name in (("h_", info["home_team"]), ("a_", info["away_team"])):
            if info.get(f"{prefix}qb_starter_listed") is False:
                flags.append(
                    f"{name}: no starting QB listed yet; quarterback change uses the "
                    "projected QB (from the injury report with --injuries)"
                )
    if used & set(MARKET_FEATURES):
        for prefix, name in (("h_", info["home_team"]), ("a_", info["away_team"])):
            if info.get(f"{prefix}mkt_games"):
                continue
            if info.get(f"{prefix}mkt_prior_games"):
                flags.append(
                    f"{name}: no closing lines yet this season; market strength is last "
                    "season's, carried over"
                )
            else:
                flags.append(f"{name}: no closing lines cached; market strength is league average")
    if used & {"rest", "opp_rest"} and info.get("rest_missing"):
        flags.append("rest days unknown for this game; both teams assumed on normal rest")
    if used & {"indoor", "wind"}:
        roof = info.get("roof")
        if roof is None:
            flags.append("roof unknown (neutral site or no home games cached); assumed outdoors")
        elif info.get("roof_inferred"):
            flags.append(f"roof ({roof}) taken from {info['home_team']}'s latest home game")
    if "wind" in used and info.get("wind_missing"):
        flags.append("wind unknown for this outdoor game; using the average outdoor wind")

    model = fit_points_model(train, settings.model)
    runs = availability_runs(row, model, lineup, settings, n, seed)
    mu_home, mu_away = model.expected_points(row, runs)
    sim = simulate_scores(mu_home, mu_away, model.dispersion, n, seed, settings.sim.overtime_points)

    def quantiles(values: np.ndarray) -> dict[float, float]:
        return {q: float(np.quantile(values, q)) for q in QUANTILES}

    return MatchupResult(
        home=info["home_team"],
        away=info["away_team"],
        season=season,
        week=int(info["week"]),
        n=n,
        seed=seed,
        home_win_prob=float(sim.home_win_prob()[0]),
        mean_margin=float(sim.mean_margin()[0]),
        mean_total=float(sim.mean_total()[0]),
        expected_home=float(np.mean(mu_home[0])),
        expected_away=float(np.mean(mu_away[0])),
        margin_quantiles=quantiles(sim.margin[0]),
        home_quantiles=quantiles(sim.home[0]),
        away_quantiles=quantiles(sim.away[0]),
        flags=flags,
        model=model,
        sampled_availability=runs is not None,
        home_scores=sim.home[0],
        away_scores=sim.away[0],
    )


@dataclass(frozen=True)
class WeekPrediction:
    """One scheduled game of a week: its result, or why it could not be simulated."""

    game_id: str
    result: MatchupResult | None
    error: str | None = None


def predict_week(
    games: pl.DataFrame,
    settings: Settings,
    season: int,
    week: int,
    n: int,
    seed: int,
    lineup: pl.DataFrame | None = None,
) -> list[WeekPrediction]:
    """Simulate every scheduled game of ``season`` week ``week`` in ``games``.

    Each game is simulated exactly as ``predict_matchup`` would simulate it alone (its
    row becomes the matchup row, and every game uses ``seed``), so a game's numbers match
    ``simulate`` for that game. A game that cannot be simulated (e.g. no ratings) gets an
    ``error`` instead of stopping the rest. Ordered by ``game_id``.
    """
    ids = (
        games.filter((pl.col("season") == season) & (pl.col("week") == week))
        .sort("game_id")["game_id"]
        .to_list()
    )
    if not ids:
        raise ValueError(f"no games scheduled in {season} week {week}")
    out: list[WeekPrediction] = []
    for game_id in ids:
        relabeled = games.with_columns(
            pl.when(pl.col("game_id") == game_id)
            .then(pl.lit(MATCHUP_ID))
            .otherwise(pl.col("game_id"))
            .alias("game_id")
        )
        try:
            result = predict_matchup(relabeled, settings, n, seed, lineup)
        except ValueError as err:
            out.append(WeekPrediction(game_id, None, str(err)))
            continue
        out.append(WeekPrediction(game_id, result))
    return out


def availability_report(
    cache: DataCache, lineup: pl.DataFrame, result: MatchupResult, min_share: float
) -> pl.DataFrame:
    """Players in the matchup who may not play (``uncertain_players``), with names."""
    targets = pl.DataFrame(
        {
            "season": [result.season] * 2,
            "week": [result.week] * 2,
            "team": [result.home, result.away],
        }
    )
    names = cache.load("players").select("gsis_id", pl.col("display_name").alias("player"))
    return (
        uncertain_players(lineup, targets, min_share)
        .join(names, on="gsis_id", how="left")
        .with_columns(pl.col("player").fill_null(pl.col("gsis_id")))
        .select(
            "team", "player", "position", "role", "slot_share", "status", "practice", "play_prob"
        )
    )
