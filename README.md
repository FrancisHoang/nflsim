# nflsim

A Monte Carlo NFL game simulator. Given two teams and a week, it simulates the game many
times and reports win probability, score distributions, spread, total, and player stat
lines, accounting for season stats, player matchups, and injuries on both rosters.

> **Status:** Phase 0 (scaffold and data layer). There is no simulator yet. See the
> [roadmap](#roadmap).

NFL outcomes are high-variance. The goal is a **well-calibrated** model, not certainty:
every phase is backtested against the Phase 1 baseline, and a phase that doesn't improve
on it is reported as such.

## Install

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.11+ and all dependencies
into a project-local `.venv`:

```sh
git clone https://github.com/FrancisHoang/nflsim.git
cd nflsim
uv sync
```

## Data

All data comes from [nflverse](https://github.com/nflverse) via
[nflreadpy](https://github.com/nflverse/nflreadpy) and is cached as parquet under `data/`
(gitignored), one file per dataset per season:

```
data/pbp/season=2024.parquet
data/injuries/season=2024.parquet
data/players.parquet
```

```sh
uv run nflsim datasets                                    # list available datasets
uv run nflsim fetch                                       # everything, 2016 to now
uv run nflsim fetch --seasons 2023-2024 --datasets pbp,schedules
uv run nflsim fetch --seasons 2024 --refresh              # force re-download
```

Completed seasons are downloaded once and never refetched. Files for the current season
are refetched once they are older than `current_season_max_age_hours` (12 by default).
Seasons before a dataset's first available year are skipped with a message.

From Python:

```python
from nflsim.config import load_settings
from nflsim.data import DataCache

cache = DataCache(load_settings().data)
pbp = cache.load("pbp", [2023, 2024])  # polars DataFrame
```

See [docs/data.md](docs/data.md) for the dataset list and known gaps.

## Configuration

Every tunable number lives in a TOML config file, never in code. The defaults are in
[`src/nflsim/default_config.toml`](src/nflsim/default_config.toml). To change them, copy
that file and pass `--config my_config.toml`. Unknown keys or out-of-range values are
rejected.

## Development

```sh
uv run pytest            # tests (never touch the network)
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # strict type checking
```

CI runs all four on Python 3.11 and 3.12 for every push and pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md) for branch and commit conventions.

## Roadmap

| Phase | Feature | Status |
|---|---|---|
| 0 | Scaffold, CI, cached data loaders | ✅ |
| 1 | Baseline team-rating sim with shrinkage and a backtest harness (the benchmark) | |
| 2 | Player-level ratings | |
| 3 | Injuries for both rosters, sampled per simulation | |
| 4 | Position-group and individual matchups | |
| 5 | Tendencies, game script, drive/play-level engine | |
| 6 | Context: home field, rest, travel, weather, coaching | |
| 7 | Scenarios ("QB1 out"), CLI output, optional dashboard | |

## License

[MIT](LICENSE). nflverse data is subject to its own licensing and is never committed to
this repository.
