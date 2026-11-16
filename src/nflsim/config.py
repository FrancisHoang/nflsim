"""Configuration models and loading.

All tunable numbers live in TOML files validated by these pydantic models. The packaged
``default_config.toml`` is used unless a path to another file is given.
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    """Settings for downloading and caching nflverse data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_dir: Path = Path("data")
    default_first_season: int = Field(default=2016, ge=1999)
    current_season_max_age_hours: float = Field(default=12.0, gt=0)


class Settings(BaseModel):
    """Top-level nflsim settings. New phases add new sections here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: DataConfig = DataConfig()


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
