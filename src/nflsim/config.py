"""Configuration models and loading.

All tunable numbers live in TOML files validated by these pydantic models. The packaged
``default_config.toml`` is used unless a path to another file is given.
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RatingsConfig(BaseModel):
    """Team offense/defense ratings built from play-by-play (Phase 1).

    Every rating is an empirical-Bayes blend: ``(sum + k * prior) / (count + k)``, where
    ``k`` is the prior strength below and the prior is the league average plus a
    regressed share of the team's previous-season deviation from the league average.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Plays count toward ratings only when the offense's pre-snap win probability is in
    # this window, to drop garbage time. Use 0.0 and 1.0 to keep every play.
    min_wp: float = Field(default=0.05, ge=0, le=1)
    max_wp: float = Field(default=0.95, ge=0, le=1)
    # Prior strength, in plays, for each per-play metric.
    prior_plays_epa: float = Field(default=300.0, gt=0)
    prior_plays_success: float = Field(default=200.0, gt=0)
    prior_plays_pass_epa: float = Field(default=250.0, gt=0)
    prior_plays_rush_epa: float = Field(default=400.0, gt=0)
    # Prior strength, in games, for pace (offensive plays per game).
    prior_games_pace: float = Field(default=4.0, gt=0)
    # Share of last season's deviation from league average carried into this season's prior.
    carryover: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def _wp_window(self) -> Self:
        if self.min_wp >= self.max_wp:
            raise ValueError("min_wp must be below max_wp")
        return self


class DataConfig(BaseModel):
    """Settings for downloading and caching nflverse data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_dir: Path = Path("data")
    default_first_season: int = Field(default=2016, ge=1999)
    current_season_max_age_hours: float = Field(default=12.0, gt=0)


PointsFeature = Literal[
    "off_epa",
    "def_epa",
    "off_success",
    "def_success",
    "off_pace",
    "def_pace",
    "off_qb",
    "off_rush",
    "off_recv",
]
# Features built from player ratings (Phase 2) rather than team ratings.
PLAYER_FEATURES: tuple[PointsFeature, ...] = ("off_qb", "off_rush", "off_recv")


class ModelConfig(BaseModel):
    """Game-level points model: ratings -> expected points per team (Phase 1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    features: tuple[PointsFeature, ...] = (
        "off_epa",
        "def_epa",
        "off_success",
        "def_success",
        "off_pace",
        "def_pace",
    )
    train_seasons: int = Field(default=5, ge=1)
    min_expected_points: float = Field(default=3.0, gt=0)
    # "points": margin = home points model minus away points model (Phase 1 baseline).
    # "direct": margin fit by its own regression on home-minus-away features; the total
    # still comes from the points model.
    margin_fit: Literal["points", "direct"] = "points"

    @model_validator(mode="after")
    def _unique_features(self) -> Self:
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must not repeat")
        return self


PlayerRole = Literal["dropback", "carry", "target"]


class PlayerRatingsConfig(BaseModel):
    """Player EPA ratings by role and projected usage (Phase 2).

    Roles: ``dropback`` (EPA per dropback, incl. sacks and scrambles; QBs only),
    ``carry`` (EPA per designed run) and ``target`` (EPA per target). Each rating is
    ``(sum + k * prior) / (plays + k)``, where the prior is the position group's league
    average plus ``d`` and

        d = carryover * prev_excess / (prev_plays + k) + unknown_deviation * k / (prev_plays + k)

    ``prev_excess`` is last season's EPA above the position average. A player with no
    plays in the role last season starts at ``position average + unknown_deviation``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Prior strength k, in plays of the role.
    prior_plays: dict[PlayerRole, float] = {"dropback": 200.0, "carry": 300.0, "target": 200.0}
    # Share of last season's (shrunk) deviation from position average carried over.
    carryover: float = Field(default=0.7, ge=0, le=1)
    # EPA per play of a player with no plays in the role last season, relative to the
    # position average (new players are usually below it).
    unknown_deviation: dict[PlayerRole, float] = {
        "dropback": -0.10,
        "carry": 0.0,
        "target": -0.02,
    }
    # Projected usage: exponentially weighted share of the team's role plays over the
    # player's recent rostered games, with this half-life in games (per role).
    usage_half_life_games: dict[PlayerRole, float] = {
        "dropback": 0.25,
        "carry": 1.0,
        "target": 2.0,
    }
    # Weekly roster statuses that mean the player cannot play that week. These are
    # known days before kickoff (injured reserve, PUP, suspension, released, retired).
    # Game-day inactives are Phase 3 (injuries).
    unavailable_statuses: tuple[str, ...] = ("RES", "PUP", "SUS", "CUT", "RET", "NWT")

    @model_validator(mode="after")
    def _all_roles(self) -> Self:
        roles = {"dropback", "carry", "target"}
        for name in ("prior_plays", "unknown_deviation", "usage_half_life_games"):
            missing = roles - set(getattr(self, name))
            if missing:
                raise ValueError(f"player_ratings.{name} is missing roles: {sorted(missing)}")
        for name in ("prior_plays", "usage_half_life_games"):
            if any(v <= 0 for v in getattr(self, name).values()):
                raise ValueError(f"player_ratings.{name} must be positive")
        return self


ReportStatus = Literal["out", "doubtful", "questionable", "none"]
Practice = Literal["dnp", "limited", "full", "none"]
REPORT_STATUSES: tuple[ReportStatus, ...] = ("out", "doubtful", "questionable", "none")
PRACTICES: tuple[Practice, ...] = ("dnp", "limited", "full", "none")

PlayTable = dict[ReportStatus, dict[Practice, float]]


class InjuriesConfig(BaseModel):
    """Chance a player plays given his final injury report (Phase 3).

    ``status`` is the game status on the final report (``none``: on the practice report
    with no game status) and ``practice`` the last practice participation. Players are
    split into regulars (at least ``regular_min_share`` of the offensive or defensive
    snaps in the team's previous game) and others, since regulars listed questionable
    play far more often. Regulars not on the report at all play with
    ``unlisted_regular``. Measured with ``nflsim injury-eval``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    regular_min_share: float = Field(default=0.25, gt=0, le=1)
    play_probability_regular: PlayTable = {
        "out": {"dnp": 0.0, "limited": 0.0, "full": 0.0, "none": 0.0},
        "doubtful": {"dnp": 0.02, "limited": 0.02, "full": 0.02, "none": 0.02},
        "questionable": {"dnp": 0.50, "limited": 0.81, "full": 0.95, "none": 0.70},
        "none": {"dnp": 0.98, "limited": 0.99, "full": 0.99, "none": 0.99},
    }
    play_probability_other: PlayTable = {
        "out": {"dnp": 0.0, "limited": 0.0, "full": 0.0, "none": 0.0},
        "doubtful": {"dnp": 0.01, "limited": 0.01, "full": 0.01, "none": 0.01},
        "questionable": {"dnp": 0.28, "limited": 0.58, "full": 0.77, "none": 0.70},
        "none": {"dnp": 0.74, "limited": 0.86, "full": 0.89, "none": 0.89},
    }
    unlisted_regular: float = Field(default=0.97, ge=0, le=1)
    # Rostered non-regulars not on the report (backups): assumed available if needed.
    unlisted_other: float = Field(default=1.0, ge=0, le=1)
    # Order of players within a team's position when a player is out and everyone below
    # moves up: "usage" (projected usage first, depth chart breaks ties, e.g. between
    # backup QBs with no recent snaps) or "depth" (depth chart first).
    backup_order: Literal["depth", "usage"] = "usage"
    # Added to the play probability of questionable players at these report positions
    # (result clipped to [0, 1]). Questionable QBs sit far more often than other players.
    questionable_position_adjustment: dict[str, float] = {"QB": -0.15}

    @model_validator(mode="after")
    def _complete_tables(self) -> Self:
        for name in ("play_probability_regular", "play_probability_other"):
            table = getattr(self, name)
            for status in REPORT_STATUSES:
                row = table.get(status)
                if row is None:
                    raise ValueError(f"injuries.{name} is missing status {status!r}")
                missing = [p for p in PRACTICES if p not in row]
                if missing:
                    raise ValueError(f"injuries.{name}.{status} is missing {missing}")
                if any(not 0 <= v <= 1 for v in row.values()):
                    raise ValueError(f"injuries.{name}.{status} must be within [0, 1]")
        return self


class SimConfig(BaseModel):
    """Monte Carlo score simulation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overtime_points: int = Field(default=3, ge=1)


class BacktestConfig(BaseModel):
    """Walk-forward backtest settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Equal-width bins of predicted home win probability in the calibration table.
    calibration_bins: int = Field(default=10, ge=2)


class TuningConfig(BaseModel):
    """Candidate values for the shrinkage grid search (``nflsim tune``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prior_plays_epa: tuple[float, ...] = (50.0, 100.0, 200.0, 300.0, 500.0, 800.0)
    prior_plays_success: tuple[float, ...] = (50.0, 100.0, 200.0, 400.0)
    carryover: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9)
    prior_games_pace: tuple[float, ...] = (2.0, 4.0, 8.0)
    wp_windows: tuple[tuple[float, float], ...] = ((0.05, 0.95), (0.0, 1.0), (0.1, 0.9))

    @model_validator(mode="after")
    def _non_empty(self) -> Self:
        for name in ("prior_plays_epa", "prior_plays_success", "carryover", "prior_games_pace"):
            if not getattr(self, name):
                raise ValueError(f"tuning.{name} must list at least one value")
        if not self.wp_windows:
            raise ValueError("tuning.wp_windows must list at least one window")
        return self


class Settings(BaseModel):
    """Top-level nflsim settings. New phases add new sections here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: DataConfig = DataConfig()
    ratings: RatingsConfig = RatingsConfig()
    model: ModelConfig = ModelConfig()
    player_ratings: PlayerRatingsConfig = PlayerRatingsConfig()
    injuries: InjuriesConfig = InjuriesConfig()
    sim: SimConfig = SimConfig()
    backtest: BacktestConfig = BacktestConfig()
    tuning: TuningConfig = TuningConfig()


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from ``path``, or from the packaged defaults when ``path`` is None.

    Unknown keys and out-of-range values raise ``pydantic.ValidationError`` so typos
    in a config file fail loudly instead of being silently ignored.
    """
    if path is None:
        text = resources.files("nflsim").joinpath("default_config.toml").read_text("utf-8")
    else:
        text = path.read_text("utf-8")
    return Settings.model_validate(tomllib.loads(text))
