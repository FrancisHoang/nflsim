# nflsim

[![CI](https://github.com/FrancisHoang/nflsim/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FrancisHoang/nflsim/actions/workflows/ci.yml)

A Monte Carlo NFL game simulator. Given two teams and a week, it simulates the game many
times and reports win probability, score distributions, spread, total, and player stat
lines, accounting for season stats, player matchups, and injuries on both rosters.

> **Status:** Phases 0-3 and 6 done; Phases 4 and 5 tested and stopped (no gain). The game
> simulator and backtest harness work. Context (indoor, wind) is in the default model;
> player, injury and matchup features are built but opt-in because they have not beaten
> the baseline. See the [roadmap](#roadmap).

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

## Usage

```sh
# Simulate one game from ratings as of the given week (stats through the week before)
uv run nflsim simulate --home KC --away BUF --season 2025 --week 12 --n 10000 --seed 42
#   --spread -3.5 --total 47.5   cover/over chances (default: the schedule's lines)
#   --out "Name" --compare       what-if: a player out, with and without
#   --players                    per-player stat lines with ranges
#   --json                       everything as JSON

# Every game of a week in one table (model vs. Vegas line, cover and over chances)
uv run nflsim week --season 2026 --week 11

# Walk-forward backtest: fit on earlier seasons, predict each test season
uv run nflsim backtest --seasons 2021-2024 --n 10000 --seed 42
```

Every simulation takes an explicit seed. To use player ratings and injury reports, pass a
config with the player features and `injury_aware = true` under `[model]`; `simulate`
then draws who plays in each run and lists players on the injury report who may not play. The CLAUDE.md command list covers the
evaluation tools (`tune`, `player-eval`, `injury-eval`, `player-coverage`).

## Results

Phase 1 baseline, the benchmark every later phase is compared to (`nflsim backtest
--seasons 2021-2024 --n 10000 --seed 42`, 1,139 games):

| Method | Margin MAE | Total MAE | Brier | Log loss |
|---|---|---|---|---|
| Model | 10.17 | 10.69 | 0.2254 | 0.6437 |
| Vegas close | 9.79 | 10.29 | 0.2112 | 0.6115 |
| Home field only | 11.00 | 10.90 | 0.2474 | 0.6893 |

Later changes on the same games (paired bootstrap). None is significant; only indoor +
wind met the adoption rule (total MAE better in 3 of 4 seasons) and is now a default:

| Change | Brier | Log loss | Margin MAE |
|---|---|---|---|
| Direct margin fit | 0.2246 | 0.6428 | 10.18 |
| + player features (Phase 2) | 0.2255 | 0.6436 | 10.20 |
| + injury-aware player features (Phase 3) | 0.2250 | 0.6428 | 10.18 |
| ... with who plays drawn per run | 0.2250 | 0.6428 | 10.18 |
| Pass-rush matchup (Phase 4) | not added: failed its play-level gate | | |
| Home field without 2020 (Phase 6) | 0.2256 | 0.6440 | 10.18 |
| + rest (Phase 6) | 0.2255 | 0.6439 | 10.18 |
| Drive-by-drive engine (Phase 5) | 0.2251 | 0.6432 | 10.16 (CRPS margin 7.42 vs. 7.40) |
| + indoor, wind (Phase 6) | 0.2254 | 0.6438 | 10.17 (total MAE 10.69 -> **10.61**) |
| Opponent-adjusted ratings, coefficient ridge, recalibration | 0.2250-0.2256 | | 10.17-10.18 |
| **+ market strength (default since 2026-11-18)** | **0.2224** | **0.6374** | **10.11** (total MAE 10.56) |

**Market strength** (`mkt`, `opp_mkt`) is each team's strength implied by the closing
spreads of *earlier* weeks, shrunk toward last season. It is the first change that
improves accuracy: Brier -0.0030 on 2021-2024 (95% CI [-0.0062, +0.0002], better in 3 of
4 seasons) and -0.0042 on 2017-2024 (CI [-0.0068, -0.0015]). It also fixes most of the
under-confidence (actual vs. predicted margin slope 1.20 -> 0.92). It uses no future
information, but it does make the market an input, so the backtest also reports
`pbp_only`, the same model without it, to keep an independent measure of the
play-by-play ratings. Details: [docs/underconfidence.md](docs/underconfidence.md).

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
| 1 | Baseline team-rating sim with shrinkage and a backtest harness (the benchmark) | ✅ |
| 2 | Player-level ratings | ✅ (opt-in; no backtest gain) |
| 3 | Injuries for both rosters, sampled per simulation | ✅ (opt-in; no significant backtest gain) |
| 4 | Position-group and individual matchups | ⛔ stopped: pass-rush matchup tested, no gain ([plan and results](docs/phase4-plan.md)) |
| 5 | Tendencies, game script, drive/play-level engine | ⛔ drive engine built (opt-in), no gain at its gate; game script not built ([plan and results](docs/phase5-plan.md)) |
| 6 | Context: home field, rest, travel, weather, coaching | ✅ indoor + wind by default; home field and rest no gain ([plan and results](docs/phase6-plan.md)) |
| 7 | Scenarios ("QB1 out"), CLI output, optional dashboard | |

## License

[MIT](LICENSE). nflverse data is subject to its own licensing and is never committed to
this repository.
