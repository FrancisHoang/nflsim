# NFL Game Simulator (nflsim)

Monte Carlo simulator for single NFL games. Given two teams and a week, it simulates the game many times and reports win probability, score distributions, spread/total, and key player stat lines. It accounts for season stats, player-level matchups, and injuries on both rosters, and supports "what-if" scenarios.

## Stack
- Python 3.11+, managed with `uv`
- Layout: `src/nflsim/`, `tests/`, `docs/`, `notebooks/` (optional)
- Data: nflverse via `nflreadpy` (play-by-play, schedules, rosters, player stats, injuries, depth charts, snap counts, PFR advanced stats, Next Gen Stats)
- polars (chosen in Phase 0; do not mix in pandas), pydantic, numpy, typer (CLI), pytest, ruff, mypy
- Cache all data as parquet under `data/` (gitignored)

## Commands
- Install: `uv sync`
- Test: `uv run pytest`
- Lint/format: `uv run ruff check . && uv run ruff format .` (use `--no-cache` before pushing; a stale cache once hid CI failures)
- Types: `uv run mypy` (strict; checks `src` and `tests` via pyproject)
- Fetch data: `uv run nflsim fetch --seasons 2016-2025 --datasets pbp,schedules,rosters_weekly,snap_counts,players`
- List datasets: `uv run nflsim datasets`
- Player ID coverage: `uv run nflsim player-coverage --seasons 2016-2025` (share of snaps and pbp player IDs that link to a known player; lists unmatched players)
- Player ratings check: `uv run nflsim player-eval --seasons 2018-2025` (out-of-sample skill of player EPA ratings by role vs. position average, and projected-usage error; settings in config `[player_ratings]`)
- Backtest: `uv run nflsim backtest --seasons 2021-2024 --n 10000 --seed 42` (`--features off_epa,def_epa` to compare feature sets; player-based `off_qb,off_rush,off_recv` need `rosters_weekly` and `players` cached; `--output preds.parquet` to save per-game predictions, `--margin-fit direct` to use the direct margin fit)
- Tune shrinkage: `uv run nflsim tune --seasons 2021-2023 --holdout 2024 --n 4000 --seed 42` (grid in config `[tuning]`; `--config` to override)
- Simulate: `uv run nflsim simulate --home KC --away BUF --season 2025 --week 12 --n 10000 --seed 42` (`--neutral` for neutral sites)

(Update this list as commands are added.)

## Working agreement
- Build ONE roadmap phase at a time, then stop and summarize. Do not start the next phase unprompted.
- Before coding: give a 5-10 line plan, and list assumptions and data gaps.
- One feature per branch, conventional commits, small reviewable PRs.
- Write tests alongside code, using small fixture data. Tests must never hit the network.
- End each phase with: what was added, how to run it, test results, backtest numbers, suggested next step.
- **No Claude attribution.** Never add `Co-Authored-By: Claude` trailers or "Generated with Claude Code" lines to commits or PRs. Commits are authored by the repo owner's git identity only.

## Modeling rules
- **No future leakage.** When simulating week W, use only information available before kickoff (stats through W-1, injury report as of that week).
- **Reproducibility.** Every simulation takes an explicit RNG seed.
- **No hardcoded magic numbers.** Shrinkage strengths, injury play-probabilities, home-field value, noise levels, etc. live in a config file.
- **Shrink small samples** toward league/position averages (empirical Bayes). Early-season ratings blend in prior-season data.
- **Degrade gracefully.** If a matchup or injury has no data, fall back to a coarser estimate and flag it in the output. Never fabricate precision.
- **Injuries are probabilistic.** Sample player availability per simulation run from report status (Out/Doubtful/Questionable/limited/DNP). Replace unavailable players with depth chart backups at the backup's own rating. Include BOTH teams.

## Evaluation (the most important rule)
- The Phase 1 baseline and backtest harness are the benchmark. Every later phase must be backtested against the previous one.
- Report: MAE of predicted vs. actual margin, Brier score and log loss for win probability, calibration table, and comparison to closing Vegas lines where available.
- If a change does not improve results, say so plainly. More factors does not automatically mean a better model. Do not claim success without numbers.
- NFL outcomes are high-variance. The goal is a well-calibrated model, not certainty.

## Roadmap
0. ✅ Scaffold: repo, pyproject, CI, data loaders with parquet caching, README
1. ✅ Baseline game sim: team offense/defense ratings (EPA/play, success rate, pace), shrinkage, score distributions, backtest harness
2. ✅ Player-level ratings by position, usage-weighted, with shrinkage (opt-in features; no backtest gain yet)
3. Injuries: weekly reports and practice participation, availability sampling, depth chart replacement
4. Matchups: pass rush vs. pass block, WR/TE vs. CB/S, run game vs. run defense, QB vs. pressure/scheme
5. Tendencies and game script: play-calling, pace, 4th down, red zone; engine moves from game-level to drive-level to play-level
6. Context: home field, rest/travel, weather/dome, coaching, divisional familiarity
7. Scenarios and output: overrides ("QB out"), N-run summaries, per-player stat lines, CLI, optional Streamlit dashboard

Mark phases done here as they merge.

## Current status
Phases 0-2 done and merged to `main` (Phase 1 via PR #1, Phase 2 via PRs #4-#6), CI green on Python 3.11 and 3.12. Phase 2 player features did not beat the Phase 1 baseline, so the baseline below is still the benchmark.

Phase 1 baseline (the benchmark; `nflsim backtest --seasons 2021-2024 --n 10000 --seed 42`, 1,139 games):

| Method | Margin MAE | Total MAE | Brier | Log loss |
|---|---|---|---|---|
| Model | 10.17 | 10.69 | 0.2254 | 0.6437 |
| Vegas close | 9.79 | 10.29 | 0.2112 | 0.6115 |
| Home only | 11.00 | 10.90 | 0.2474 | 0.6893 |

Known issues: predictions are under-confident (actual margin vs. predicted margin slope 1.20, Vegas 1.03; predicted margin SD 4.3 vs. Vegas 6.0). Feature sets tie (defensive success rate unstable). The six rating features overlap heavily, so their coefficients swing from season to season.

Shrinkage tuning (2026-11-16, `nflsim tune`): best grid settings (EPA prior 800 plays, carryover 0.7, pace prior 16 games, no garbage-time filter) improved tuning-season Brier 0.2282 -> 0.2240 but NOT the 2024 holdout (+0.0006, 95% CI [-0.0035, +0.0048]). Defaults left unchanged. Shrinkage is not the cause of the under-confidence (best settings: margin SD 4.5).

Margin fit (2026-11-17, `[model] margin_fit`): the under-confidence is structural, not early-season noise (training on weeks 8+ leaves the slope unchanged; in-sample slope is about 1.30). The per-team points model leaves out effects between the two sides. `margin_fit = "direct"` fits the margin separately on home-minus-away features: slope 1.20 -> 0.94, margin SD 4.3 -> 5.6, but Brier 0.2254 -> 0.2246 (diff -0.0008, 95% CI [-0.0030, +0.0014]), log loss 0.6437 -> 0.6428 (CI [-0.0062, +0.0044]), margin MAE 10.17 -> 10.18. The improvement is not significant (2 of 4 seasons better), so the default stays `"points"`. Calibration is not where the gap to Vegas comes from.

Player ratings (2026-11-17, Phase 2 step 2, `nflsim player-eval`, settings tuned on 2018-2022 and checked on 2023-2025): skill vs. position average 2018-2025 is dropback +4.6%, carry +0.45%, target +0.53%. Raw season-to-date averages are 20-31% WORSE than the position average, so shrinkage is essential. QBs with no prior-season dropbacks average -0.11 EPA/dropback below the QB average (config `unknown_deviation`). Usage: shortest half-lives win (dropback 0.25 games, i.e. last game's QB); total-variation distance dropback 0.12, carry 0.27, target 0.32. Known gaps until Phase 3: rookie/new starters with no history project to 0% (14 of 3,884 team-weeks); mid-season QB changes are only seen after one game.

Team offense from players (2026-11-17, Phase 2 step 3, `off_qb`/`off_rush`/`off_recv` features: projected usage x shrunk player rating minus league role average, as of W-1). Backtest 2021-2024, n 10000, seed 42, paired bootstrap vs. baseline: adding them to the six team features gives Brier 0.2255 (diff +0.0001, 95% CI [-0.0018, +0.0020]), log loss 0.6436, margin MAE 10.20 (better in 2021, worse in 2022-2024). Replacing off_epa/off_success with them is worse: Brier 0.2276 (+0.0022, CI [-0.0006, +0.0050]), margin MAE 10.28 (+0.10, CI [+0.01, +0.20]). Slope improves (1.20 -> 1.09) but accuracy does not. Default features unchanged. Without injury data the projected starter is last game's QB, which the team ratings already reflect, so player ratings add little until Phase 3 knows who actually plays.

Data notes:
- Default fetch range is 2016 onward (config `default_first_season`).
- In `schedules`, `spread_line` is positive when the HOME team is favored, and `result` is home score minus away score.
- `DataCache.load` returns whole seasons; leakage filtering (stats through W-1) is each model's job. Use `DataCache.scan` for pbp (lazy, reads only needed columns).
- Schedules keep historical team codes (OAK, SD, STL); pbp uses current ones (LV, LAC, LA). Normalize with `nflsim.data.normalize_teams` before joining. Snap counts and weekly rosters also use historical codes.
- Players are keyed by GSIS ID (`00-00xxxxx`): pbp `passer/rusher/receiver_player_id` and `rosters_weekly.gsis_id`. Snap counts use PFR IDs; map them with `nflsim.data.player_snaps` (via the one-to-one `players` crosswalk; `rosters_weekly.pfr_id` adds nothing). Coverage 2016-2025 is 99.9%+ of snaps and 100% of pbp IDs; unmatched rows are kept with `id_matched = False`.
- Take positions from `rosters_weekly` per week (`nflsim.data.roster_weeks`), not from `players`: about 370 players change position over time. Some non-QBs throw passes (trick plays), so rate by position, not by pbp role.
