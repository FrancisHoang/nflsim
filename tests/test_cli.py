from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import nflsim.cli as cli
from nflsim.data import DatasetSpec
from tests.conftest import FakeSource

runner = CliRunner()


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: FakeSource) -> Path:
    fake = {
        "games": DatasetSpec("games", "fake games", first_season=2020, fetch_season=source.games),
        "teams": DatasetSpec("teams", "fake teams", fetch_all=source.teams),
    }
    monkeypatch.setattr("nflsim.data.cache.DATASETS", fake)
    monkeypatch.setattr(cli, "DATASETS", fake)
    path = tmp_path / "cfg.toml"
    path.write_text(f"[data]\ncache_dir = '{(tmp_path / 'data').as_posix()}'\n")
    return path


def test_fetch_writes_cache(config_file: Path, source: FakeSource, tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app, ["fetch", "--seasons", "2019-2021", "--config", str(config_file)]
    )
    assert result.exit_code == 0, result.output
    assert "games: skipping [2019]" in result.output
    assert "games: 4 rows cached" in result.output
    assert "teams: 2 rows cached" in result.output
    assert (tmp_path / "data" / "games" / "season=2021.parquet").exists()
    assert ("games", 2019) not in source.calls


def test_fetch_selected_dataset(config_file: Path, source: FakeSource) -> None:
    result = runner.invoke(
        cli.app,
        ["fetch", "--seasons", "2020", "--datasets", "teams", "--config", str(config_file)],
    )
    assert result.exit_code == 0, result.output
    assert source.calls == [("teams", None)]


def test_datasets_command_lists_registry(config_file: Path) -> None:
    result = runner.invoke(cli.app, ["datasets"])
    assert result.exit_code == 0
    assert "games" in result.output and "2020+" in result.output
