from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nflsim.config import load_settings


def test_packaged_defaults_load() -> None:
    settings = load_settings()
    assert settings.data.cache_dir == Path("data")
    assert settings.data.default_first_season == 2016


def test_override_file(tmp_path: Path) -> None:
    path = tmp_path / "custom.toml"
    path.write_text('[data]\ncache_dir = "elsewhere"\ndefault_first_season = 2018\n')
    settings = load_settings(path)
    assert settings.data.cache_dir == Path("elsewhere")
    assert settings.data.default_first_season == 2018
    assert settings.data.current_season_max_age_hours == 12.0  # untouched default


@pytest.mark.parametrize(
    "body",
    [
        "[data]\ncache_dirr = 'typo'\n",  # unknown key
        "[data]\ndefault_first_season = 1990\n",  # before nflverse coverage
        "[data]\ncurrent_season_max_age_hours = 0\n",  # must be positive
    ],
)
def test_invalid_config_fails_loudly(tmp_path: Path, body: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(body)
    with pytest.raises(ValidationError):
        load_settings(path)
