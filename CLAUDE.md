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
- Fetch data: `uv run nflsim fetch --seasons 2016-2025 --datasets pbp,schedules,rosters_weekly,snap_counts,players,injuries,depth_charts,participation,pfr_pass,pfr_def`
- List datasets: `uv run nflsim datasets`
- Player ID coverage: `uv run nflsim player-coverage --seasons 2016-2025` (share of snaps and pbp player IDs that link to a known player; lists unmatched players)
- Player ratings check: `uv run nflsim player-eval --seasons 2018-2025` (out-of-sample skill of player EPA ratings by role vs. position average, and projected-usage error; settings in config `[player_ratings]`; `--injuries` adds injury-aware usage vs. Phase 2)
- Matchup check: `uv run nflsim matchup-eval --seasons 2016-2025` (per season: participation coverage of dropbacks, pressure rate and EPA pressured vs. clean, man/zone, agreement with PFR pressures; out-of-sample skill of the pass rush ratings; and the Phase 4 step 3 gate, `--fit-seasons 2018-2022 --test-seasons 2023-2025 --n-boot 2000 --seed 42`). Needs `pbp`, `participation`, `pfr_pass` cached
- Drive table check: `uv run nflsim drive-eval --seasons 2016-2025` (per season: drives per game, result shares, start field position, seconds per drive, and whether drive points add up to every final score). Needs `pbp`, `schedules`
- Stat lines check: `uv run nflsim stat-eval --fit-seasons 2018-2022 --test-seasons 2023-2025` (per-player stat lines vs. each player's per-game average, game-bootstrap CIs, range coverage; prints the range quantiles for config `[stat_lines]`). Needs `player_stats` cached
- Injury play rates: `uv run nflsim injury-eval --seasons 2023-2025` (how often players on the final injury report play, by status x practice x regular, vs. config `[injuries]`; also report and depth chart coverage). Needs `injuries`, `depth_charts`, `snap_counts`, `rosters_weekly`, `players`, `schedules` cached
- Backtest: `uv run nflsim backtest --seasons 2021-2024 --n 10000 --seed 42` (`--engine drive` for the drive-by-drive engine; also prints CRPS of margin and total, cover/over calibration against the closing lines and key-number frequencies, config `[backtest] key_margins`; `--features off_epa,def_epa` to compare feature sets; player-based `off_qb,off_rush,off_recv` need `rosters_weekly` and `players` cached; `--injuries` builds them from injury-aware lineups (config `[model] injury_aware`; also needs `injuries`, `depth_charts`, `snap_counts`), `--sample-availability/--expected-availability` draws who plays per run or not (config `[sim] sample_availability`); `--output preds.parquet` to save per-game predictions, `--margin-fit direct` to use the direct margin fit)
- Tune shrinkage: `uv run nflsim tune --seasons 2021-2023 --holdout 2024 --n 4000 --seed 42` (grid in config `[tuning]`; `--config` to override)
- Simulate: `uv run nflsim simulate --home KC --away BUF --season 2025 --week 12 --n 10000 --seed 42` (`--neutral` for neutral sites; uses the scheduled game when there is one and prints a "What I used" block with the injury report for both teams at all positions; `--injuries` to use the report in the prediction; `--offline` to never download; `--out/--plays/--doubtful "Name"` to override availability for either team, `--compare` to show the effect with the same seed; `--players` for per-player stat lines with 10-90% ranges; `--spread -3.5 --total 47.5` for cover/push/over chances, home team's line in sportsbook sign, defaulting to the scheduled game's nflverse lines; `--json` for the full result as JSON). With player features and `[model] injury_aware = true` in a `--config` file, who plays is drawn per run (`[sim] sample_availability`) and players on the injury report who may not play are listed (`[sim] report_min_share`)

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
3. ✅ Injuries: weekly reports and practice participation, availability sampling, depth chart replacement
4. ⛔ (stopped at the step 3 gate, no gain; `docs/phase4-plan.md`) Matchups: pass rush vs. pass block, WR/TE vs. CB/S, run game vs. run defense, QB vs. pressure/scheme
5. ⛔ (drive engine built, opt-in; stopped at its gate, game script not built; `docs/phase5-plan.md`) Tendencies and game script: play-calling, pace, 4th down, red zone; engine moves from game-level to drive-level to play-level
6. ✅ Context (indoor + wind on by default; home-field options and rest opt-in, no gain; `docs/phase6-plan.md`): home field, rest/travel, weather/dome, coaching, divisional familiarity
7. Scenarios and output: overrides ("QB out"), N-run summaries, per-player stat lines, CLI, optional Streamlit dashboard

Mark phases done here as they merge.

## Current status
Phases 0-3 and 6 done and merged to `main` (Phase 1 via PR #1, Phase 2 via PRs #4-#6, Phase 3 via PRs #7-#8, Phase 6 via PRs #15-#16); Phase 4 stopped at its gate (PRs #11-#14). Phase 5 stopped at its gate (PRs #17-#20): the drive engine is opt-in (`[sim] engine = "drive"`). Phase 7 (scenarios and output) is next on the roadmap. CI green on Python 3.11 and 3.12. Player (Phase 2) and injury (Phase 3) features are opt-in. Phase 6 added indoor and wind to the default features, which lowered total MAE only (not significant, but it met the pre-set rule).

**Current default model (the benchmark from Phase 6 on;** `nflsim backtest --seasons 2021-2024 --n 10000 --seed 42`, 1,139 games): margin MAE 10.17, total MAE 10.61, Brier 0.2254, log loss 0.6438, CRPS margin 7.404, CRPS total 7.517 (Vegas as a normal around the line: 7.126 / 7.302). Key numbers actual vs. model: margin 3 15.2% vs. 9.0%, 7 7.5% vs. 5.2%. Compare later phases against this.

Phase 1 baseline (the original benchmark, six team features only; `nflsim backtest --seasons 2021-2024 --n 10000 --seed 42`, 1,139 games):

| Method | Margin MAE | Total MAE | Brier | Log loss |
|---|---|---|---|---|
| Model | 10.17 | 10.69 | 0.2254 | 0.6437 |
| Vegas close | 9.79 | 10.29 | 0.2112 | 0.6115 |
| Home only | 11.00 | 10.90 | 0.2474 | 0.6893 |

Known issues: predictions are under-confident (actual margin vs. predicted margin slope 1.20, Vegas 1.03; predicted margin SD 4.3 vs. Vegas 6.0). Feature sets tie (defensive success rate unstable). The six rating features overlap heavily, so their coefficients swing from season to season.

Shrinkage tuning (2026-11-16, `nflsim tune`): best grid settings (EPA prior 800 plays, carryover 0.7, pace prior 16 games, no garbage-time filter) improved tuning-season Brier 0.2282 -> 0.2240 but NOT the 2024 holdout (+0.0006, 95% CI [-0.0035, +0.0048]). Defaults left unchanged. Shrinkage is not the cause of the under-confidence (best settings: margin SD 4.5).

Under-confidence experiments (2026-11-18, `docs/underconfidence.md`, gate set beforehand): opponent-adjusted ratings (`[ratings] opponent_adjust`), fewer features, coefficient ridge (`[model] ridge`), direct fit and walk-forward recalibration all fail (best Brier -0.0006, n.s.). Market strength from earlier weeks' closing spreads (opt-in features `mkt`, `opp_mkt`, `[market]`) passes: Brier 0.2254 -> 0.2224 (-0.0030, 95% CI [-0.0062, +0.0002]), better in 3 of 4 seasons on Brier and margin MAE; 2017-2024 -0.0042 (CI [-0.0068, -0.0015]); slope 1.20 -> 0.92, margin SD 5.8. Not leakage, but it makes Vegas an input, so the default is unchanged pending that decision. The under-confidence was missing information, not scaling. Per-season slopes swing 0.91-1.42; the base model was nearly calibrated in 2017-2020 (slope 1.11).

Margin fit (2026-11-17, `[model] margin_fit`): the under-confidence is structural, not early-season noise (training on weeks 8+ leaves the slope unchanged; in-sample slope is about 1.30). The per-team points model leaves out effects between the two sides. `margin_fit = "direct"` fits the margin separately on home-minus-away features: slope 1.20 -> 0.94, margin SD 4.3 -> 5.6, but Brier 0.2254 -> 0.2246 (diff -0.0008, 95% CI [-0.0030, +0.0014]), log loss 0.6437 -> 0.6428 (CI [-0.0062, +0.0044]), margin MAE 10.17 -> 10.18. The improvement is not significant (2 of 4 seasons better), so the default stays `"points"`. Calibration is not where the gap to Vegas comes from.

Player ratings (2026-11-17, Phase 2 step 2, `nflsim player-eval`, settings tuned on 2018-2022 and checked on 2023-2025): skill vs. position average 2018-2025 is dropback +4.6%, carry +0.45%, target +0.53%. Raw season-to-date averages are 20-31% WORSE than the position average, so shrinkage is essential. QBs with no prior-season dropbacks average -0.11 EPA/dropback below the QB average (config `unknown_deviation`). Usage: shortest half-lives win (dropback 0.25 games, i.e. last game's QB); total-variation distance dropback 0.12, carry 0.27, target 0.32. Known gaps until Phase 3: rookie/new starters with no history project to 0% (14 of 3,884 team-weeks); mid-season QB changes are only seen after one game.

Team offense from players (2026-11-17, Phase 2 step 3, `off_qb`/`off_rush`/`off_recv` features: projected usage x shrunk player rating minus league role average, as of W-1). Backtest 2021-2024, n 10000, seed 42, paired bootstrap vs. baseline: adding them to the six team features gives Brier 0.2255 (diff +0.0001, 95% CI [-0.0018, +0.0020]), log loss 0.6436, margin MAE 10.20 (better in 2021, worse in 2022-2024). Replacing off_epa/off_success with them is worse: Brier 0.2276 (+0.0022, CI [-0.0006, +0.0050]), margin MAE 10.28 (+0.10, CI [+0.01, +0.20]). Slope improves (1.20 -> 1.09) but accuracy does not. Default features unchanged. Without injury data the projected starter is last game's QB, which the team ratings already reflect, so player ratings add little until Phase 3 knows who actually plays.

Data notes:
- Default fetch range is 2016 onward (config `default_first_season`).
- In `schedules`, `spread_line` is positive when the HOME team is favored, and `result` is home score minus away score.
- `DataCache.load` returns whole seasons; leakage filtering (stats through W-1) is each model's job. Use `DataCache.scan` for pbp (lazy, reads only needed columns).
- Schedules keep historical team codes (OAK, SD, STL); pbp uses current ones (LV, LAC, LA). Normalize with `nflsim.data.normalize_teams` before joining. Snap counts and weekly rosters also use historical codes.
- Players are keyed by GSIS ID (`00-00xxxxx`): pbp `passer/rusher/receiver_player_id` and `rosters_weekly.gsis_id`. Snap counts use PFR IDs; map them with `nflsim.data.player_snaps` (via the one-to-one `players` crosswalk; `rosters_weekly.pfr_id` adds nothing). Coverage 2016-2025 is 99.9%+ of snaps and 100% of pbp IDs; unmatched rows are kept with `id_matched = False`.
- Injury reports (`nflsim.data.injury_reports`): one row per player per team-week; `status` out/doubtful/questionable/none, `practice` dnp/limited/full/none. 2023 has no reports after the Wild Card round. In the cached frame `season`/`week` come back as floats (cast them).
- Depth charts (`nflsim.data.depth_chart_weeks`): through 2024 one chart per team-week; from 2025 nflverse publishes daily snapshots with no week, so each game uses the last snapshot before kickoff (schedule times are US Eastern). Listed starters play ~91%; the depth-1 QB leads his team's QB snaps 84-90% of games.
- Play rates (Phase 3 step 1, 2026-11-17, fit 2016-2022, checked 2023-2025): Out 0%, Doubtful 1-2%, Questionable regulars 50% (DNP) / 81% (limited) / 95% (full), non-regulars far lower; regulars not on the report 97% (healthy scratches). Questionable QBs play ~15 points less (config `questionable_position_adjustment`). Holdout Brier for players on the report 0.0708 vs. 0.0746 by status alone.
- Availability (Phase 3 step 2, 2026-11-17, `nflsim.model.availability`): each projected skill player gets a play probability; within team/role/position, when a player is out everyone below moves up one place and takes that place's healthy share (plays at his own rating); empty places go to a replacement-level player. Exact expectation (`expected_role_values`) and per-run sampling (`sample_role_values`); with everyone healthy it equals Phase 2. Order by projected usage beats depth chart order (`backup_order`; charts are stale). Squared error of usage vs. actual, 2023-2025 holdout, Phase 2 -> injuries: dropback 0.206 -> 0.168 (QB on report: 0.401 -> 0.246, top QB right 76% -> 83%), carry 0.114 -> 0.098, target 0.064 -> 0.059. Judge expected shares by squared error, not total-variation distance (absolute error penalizes honest probabilities).
- Injury-aware features (Phase 3 step 3, 2026-11-17, `[model] injury_aware`, `backtest --injuries`): `off_qb`/`off_rush`/`off_recv` from expected values of the injury-aware lineups. Backtest 2021-2024, n 10000, seed 42, paired bootstrap, six team features + player features: vs. Phase 2 player features Brier 0.2255 -> 0.2250 (-0.0004, 95% CI [-0.0014, +0.0004]), log loss 0.6436 -> 0.6428, margin MAE 10.20 -> 10.18; vs. the baseline Brier -0.0004 (CI [-0.0024, +0.0017]), log loss -0.0010, margin MAE +0.01. Replacing off_epa/off_success with injury-aware player features is worse (Brier +0.0015). In the 47 games where injuries moved the predicted margin by more than 1 point: Brier 0.2348 -> 0.2200 (-0.015, CI [-0.034, +0.004]), margin MAE 10.82 -> 10.39, and the margin moved toward the Vegas line 60% of the time. Right direction, but few games are affected, so nothing is significant and the default stays off.
- Per-run availability (Phase 3 step 4, 2026-11-17, `nflsim.model.availability.sampled_player_features`, `[sim] sample_availability`): each run draws who plays and uses that run's player features; the model is still fit on expected features. Backtest vs. expected features: Brier 0.2250 vs. 0.2250 (diff -0.00003, 95% CI [-0.0003, +0.0002]), margin MAE 10.181 vs. 10.182. As expected: the features are linear, so sampling changes the shape of the score distribution (e.g. a questionable QB), not the mean. Availability uses its own random stream (`default_rng([seed, 1])`). Regulars not on the report play 97% (`unlisted_regular`), so even healthy starters sit out about 3% of runs.
- Pressure data (Phase 4 step 1, `nflsim.data.dropbacks`): the raw participation flag has no value on sacks before 2023 and is always set on them from 2023; scrambles are almost never flagged before 2023. `pressure` counts every sack as pressure; leave scrambles out of anything compared across seasons. Rates still differ by era (29% / 33-36% / 28-30%), so normalize within season. Participation pressures correlate 0.64-0.73 with PFR per team-game. Plan and gate for Phase 4: `docs/phase4-plan.md`.
- Pass rush ratings (Phase 4 step 2, `nflsim.model.matchups`, config `[matchups]`): per team, offense and defense deviations from the same season's league average (pressure rate, EPA pressured, EPA clean), weeks before W only, shrunk toward carryover x last season. Skill tune 2018-2022 / holdout 2023-2025: offense pressure 0.086 / 0.044, defense pressure 0.013 / 0.030, EPA pressured about 0 (noise), EPA clean offense 0.029 / 0.041. Defensive pressure barely carries from week to week.
- Phase 4 gate (step 3, 2026-11-17, `nflsim matchup-eval`, fit 2018-2022, scored 2023-2025, game bootstrap): **fail, Phase 4 stopped.** Pressure log loss: additive (log5, fitted weights) 0.60283, + matchup term 0.60283 (diff 0.00000, CI [-0.00003, +0.00004]); log5 with no fitting is as good. Team-game pass EPA MSE: additive EPA rating 0.09290; pressure mixture 0.09403-0.09422 (worse, +0.0011 to +0.0013). Same within one era (fit 2018-2020, test 2021-2022). No matchup feature was added to the game model; the Phase 1 baseline is still the benchmark.
- Context (Phase 6, `[context]`, `nflsim.model.points.context_columns`): `rest`/`opp_rest` (days minus 7, capped +/-7), `indoor` (dome or closed roof; `open` counts as outdoors), `wind` (outdoor mph, capped at 30; 0 indoors; missing outdoor wind, 32% of 2022 and 14% of 2023, uses earlier seasons' outdoor mean and is flagged). Hypothetical `simulate` games take the roof from the home team's latest home game (flagged). Home field can leave out seasons or weight recent ones (`[model] home_exclude_seasons`, `home_half_life_seasons`); no gain. Home margin by season: +3.1 (2016), +0.1 (2019), +0.2 (2020), +1.8 to +3.0 after.
- Drive engine (Phase 5, `nflsim.model.drive_engine`, `[drives]`, `backtest --engine drive`): resamples real drives by field-position bin x clock bucket, tilted so each team's mean matches the points model (two pilot rounds). Gate vs. the score engine: CRPS margin +0.019 (CI [-0.008, +0.046]), Brier -0.0003 (n.s.), slope 1.18. Margins of 3: 7.1% vs. actual 15.2% (score engine 9.0%) - the 3s come from end-of-game behavior, not drive outcomes. A first run showed slope 1.08 but that was a calibration artifact (totals 0.7-2.8 points too high). Keep `pilot_rounds` > 0. Drive points: use the running home/away score, not per-play posteam scores (defensive-TD extra points).
- Injury files: nflverse dropped `date_modified` from 2025 on (one row per player-week; `injury_reports` handles both). The current season in the cache is 2026 (through week 10 on 2026-11-17).
- Stat lines (Phase 7 step 3, `[stat_lines]`): beat each player's per-game average on 6 of 7 stats (2023-2025), not rushing yards (+2%). QB rushing must be rated per pass attempt (scrambles), not per designed carry. Polars `fill_null` does not catch NaN from 0/0 rates: use `fill_nan`.
- Take positions from `rosters_weekly` per week (`nflsim.data.roster_weeks`), not from `players`: about 370 players change position over time. Some non-QBs throw passes (trick plays), so rate by position, not by pbp role.
