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


PointsFeature = Literal["off_epa", "def_epa", "off_success", "def_success", "off_pace", "def_pace"]


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

    @model_validator(mode="after")
    def _unique_features(self) -> Self:
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must not repeat")
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


class Settings(BaseModel):
    """Top-level nflsim settings. New phases add new sections here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: DataConfig = DataConfig()
    ratings: RatingsConfig = RatingsConfig()
    model: ModelConfig = ModelConfig()
    sim: SimConfig = SimConfig()
    backtest: BacktestConfig = BacktestConfig()


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
