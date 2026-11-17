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


def test_player_coverage_reports_links_and_unmatched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import polars as pl

    def snap_counts(season: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "game_id": ["g1", "g1"],
                "season": [season] * 2,
                "game_type": ["REG"] * 2,
                "week": [1, 1],
                "player": ["Cue Bee", "Mystery Man"],
                "pfr_player_id": ["QbaaAa00", "ZzzzZz00"],
                "position": ["QB", "G"],
                "team": ["OAK", "OAK"],
                "opponent": ["SD", "SD"],
                "offense_snaps": [60.0, 20.0],
                "offense_pct": [1.0, 0.33],
                "defense_snaps": [0.0, 0.0],
                "defense_pct": [0.0, 0.0],
                "st_snaps": [0.0, 0.0],
                "st_pct": [0.0, 0.0],
            }
        )

    def rosters_weekly(season: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "season": [season],
                "week": [1],
                "gsis_id": ["00-1"],
                "full_name": ["Cue Bee"],
                "team": ["OAK"],
                "position": ["QB"],
                "status": ["ACT"],
            }
        )

    def pbp(season: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "season": [season] * 2,
                "play_type": ["pass", "kickoff"],
                "passer_player_id": ["00-1", None],
                "rusher_player_id": [None, None],
                "receiver_player_id": [None, None],
            },
            schema_overrides={"rusher_player_id": pl.String, "receiver_player_id": pl.String},
        )

    def players() -> pl.DataFrame:
        return pl.DataFrame({"gsis_id": ["00-1"], "pfr_id": ["QbaaAa00"], "position_group": ["QB"]})

    fake = {
        "snap_counts": DatasetSpec("snap_counts", "", first_season=2012, fetch_season=snap_counts),
        "rosters_weekly": DatasetSpec(
            "rosters_weekly", "", first_season=2002, fetch_season=rosters_weekly
        ),
        "pbp": DatasetSpec("pbp", "", first_season=1999, fetch_season=pbp),
        "players": DatasetSpec("players", "", fetch_all=players),
    }
    monkeypatch.setattr("nflsim.data.cache.DATASETS", fake)
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(f"[data]\ncache_dir = '{(tmp_path / 'data').as_posix()}'\n")

    result = runner.invoke(cli.app, ["player-coverage", "--seasons", "2019", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "0.7500" in result.output  # 60 of 80 offensive snaps matched
    assert "Mystery Man" in result.output
    assert "position from crosswalk: 0" in result.output


def test_player_eval_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polars as pl

    from tests.test_players_model import completion, pbp, run

    def season_pbp(season: int) -> pl.DataFrame:
        rows = []
        for week in (1, 2, 3):
            gid = f"{season}_{week}"
            rows += [completion("QB1", "WR1", 0.2, week=week, game_id=gid)] * 5
            rows += [run("RB1", 0.0, week=week, game_id=gid)] * 3
        return pbp(rows).with_columns(season=pl.lit(season, dtype=pl.Int32))

    def rosters_weekly(season: int) -> pl.DataFrame:
        rows = [
            (season, week, gid, gid, "KC", pos, "ACT")
            for week in (1, 2, 3)
            for gid, pos in (("QB1", "QB"), ("WR1", "WR"), ("RB1", "RB"))
        ]
        return pl.DataFrame(
            rows,
            schema=["season", "week", "gsis_id", "full_name", "team", "position", "status"],
            orient="row",
        )

    def players() -> pl.DataFrame:
        return pl.DataFrame({"gsis_id": ["QB1"], "pfr_id": ["Qb00"], "position_group": ["QB"]})

    fake = {
        "pbp": DatasetSpec("pbp", "", first_season=1999, fetch_season=season_pbp),
        "rosters_weekly": DatasetSpec(
            "rosters_weekly", "", first_season=2002, fetch_season=rosters_weekly
        ),
        "players": DatasetSpec("players", "", fetch_all=players),
    }
    monkeypatch.setattr("nflsim.data.cache.DATASETS", fake)
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        f"[data]\ncache_dir = '{(tmp_path / 'data').as_posix()}'\ndefault_first_season = 2019\n"
    )
    result = runner.invoke(cli.app, ["player-eval", "--seasons", "2020", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "dropback" in result.output and "skill_rating" in result.output
    assert "tv_distance" in result.output
