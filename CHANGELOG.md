# Changelog

All notable changes, one release per roadmap phase. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may change behavior).

Every model change is backtested against the Phase 1 baseline (`nflsim backtest
--seasons 2021-2024 --n 10000 --seed 42`, 1,139 games, paired bootstrap), and results
are reported whether or not they help.

v0.1.0 and v0.2.0 were tagged after the fact, at the commits that finished those phases.
The package version in `pyproject.toml` was 0.1.0 until 0.3.0.

## [Unreleased]

Phase 4 (matchups), stopped at its go/no-go test. See `docs/phase4-plan.md`.

### Added
- Dropbacks joined with participation data: pressure, pass rushers, man/zone, coverage,
  box (`nflsim.data.dropbacks`), and PFR pressures per team-game as a cross-check.
- Pass rush and pass protection ratings as of week W-1, as deviations from the same
  season's league average (`nflsim.model.matchups`; config `[matchups]`).
- `nflsim matchup-eval`: data coverage, rating skill, and the step 3 gate.

### Fixed (data)
- The raw pressure flag changes in 2023 (sacks unflagged before, always flagged after);
  every sack now counts as pressure, and scrambles are left out of cross-season ratings.

### Results
- **The pass-rush matchup does not help.** Adding an offense x defense matchup term to
  pressure prediction changes 2023-2025 log loss by 0.00000 (95% CI [-0.00003,
  +0.00004]), and modeling pass EPA through pressure is worse than a plain EPA rating
  (+0.0011 MSE). Same result within a single season era. Per the plan's stopping rule,
  no matchup feature was added to the game model and the other matchups were not built.

## [0.3.0] - 2026-11-17

Phase 3: injuries for both rosters, sampled per simulation run.

### Added
- Weekly injury reports and depth charts (`injuries`, `depth_charts` datasets), with
  2025 daily depth chart snapshots matched to the last one before kickoff.
- Play probability by final report status, last practice participation and whether
  the player is a regular, fit on 2016-2022 and checked on 2023-2025 (`nflsim
  injury-eval`; config `[injuries]`).
- Next-man-up usage: when a player is out, the players below him at his position move
  up and take over his usage at their own ratings (`nflsim.model.availability`).
- Injury-aware player features (`[model] injury_aware`, `backtest --injuries`) and
  availability drawn per simulation run (`[sim] sample_availability`).
- `simulate` lists players on the injury report who may not play.

### Results
- Usage error (squared, 2023-2025 holdout): dropback 0.206 -> 0.168, and 0.401 -> 0.246
  when a QB is on the report; carry 0.114 -> 0.098; target 0.064 -> 0.059.
- Game model: Brier 0.2254 -> 0.2250 (95% CI [-0.0024, +0.0017]), log loss
  0.6437 -> 0.6428, margin MAE 10.17 -> 10.18. **Not significant**, so the
  injury features are opt-in and the Phase 1 baseline remains the benchmark.

## [0.2.0] - 2026-11-17

Phase 2: player-level ratings.

### Added
- Player ID crosswalk, weekly rosters and snap counts (`nflsim player-coverage`:
  99.9%+ of snaps and 100% of play-by-play player IDs linked).
- Player EPA ratings by role (dropback, carry, target), shrunk toward the position
  average with prior-season carryover, and projected usage (`nflsim player-eval`).
- Team offense features built from players (`off_qb`, `off_rush`, `off_recv`).

### Results
- Out-of-sample skill vs. the position average (2018-2025): dropback +4.6%, carry
  +0.45%, target +0.53%. Unshrunk season averages are 20-31% worse than the average.
- Game model with player features: Brier 0.2255 (+0.0001, 95% CI [-0.0018, +0.0020]).
  **No gain**, so the features are opt-in.

## [0.1.0] - 2026-11-17

Phase 0 (scaffold) and Phase 1 (baseline game simulator).

### Added
- Cached nflverse data loaders (parquet), CLI, CI on Python 3.11 and 3.12.
- Team offense and defense ratings (EPA per play, success rate, pace) with empirical-Bayes
  shrinkage and no data from the game's week or later.
- Points model and negative-binomial Monte Carlo score simulation (`nflsim simulate`).
- Walk-forward backtest against Vegas closing lines and a home-field-only baseline
  (`nflsim backtest`), shrinkage grid search (`nflsim tune`) and an optional direct
  margin fit.

### Results
| Method | Margin MAE | Total MAE | Brier | Log loss |
|---|---|---|---|---|
| Model | 10.17 | 10.69 | 0.2254 | 0.6437 |
| Vegas close | 9.79 | 10.29 | 0.2112 | 0.6115 |
| Home field only | 11.00 | 10.90 | 0.2474 | 0.6893 |

Tuned shrinkage and the direct margin fit did not significantly improve on these
numbers, so the defaults stayed.

[0.3.0]: https://github.com/FrancisHoang/nflsim/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/FrancisHoang/nflsim/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/FrancisHoang/nflsim/releases/tag/v0.1.0
