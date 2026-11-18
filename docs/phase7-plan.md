# Phase 7 plan: scenarios and output

Status: **in progress** (2026-11-17). Steps 1-3 (live matchups, overrides, stat lines) are done.

## Goal

Make the simulator usable for a real game this season without hand-holding: pull the
latest data and injury report, say exactly what it used, let you override anything ("QB
out", "he plays"), and report per-player stat lines and useful summaries. Then, as a
tested add-on, account for injured defensive players.

Unlike Phases 2-6, most of this is **usability, not accuracy**. Those parts are judged by
tests and by whether they behave correctly; the parts that change predictions (defense,
stat lines) are backtested like everything else.

## Where we start

- `simulate` uses whatever is cached. Current-season files refresh only on `nflsim fetch`
  (12-hour staleness rule), so a live run can silently use old data.
- With `[model] injury_aware = true` (off by default), offensive skill players' availability
  comes from the week's injury report automatically; uncertain players are listed.
- **Defensive players are ignored**, though the reports list them (2025: LB 821 entries,
  CB 703, DT 592, S 459, DE 448; offensive line T/G/C 1,086).
- The current season is **2026**: nflverse has it through week 10 (150 of 272 games,
  injury reports through week 10, rosters through week 11, as of 2026-11-17, a Tuesday).
  Week 11 reports appear Wednesday-Friday.

## Steps

### Step 1: live matchups

- `nflsim simulate` refreshes the current season's datasets it needs before running
  (schedules, pbp, rosters, injuries, depth charts, snap counts), using the existing
  staleness rule; `--offline` skips it.
- It uses the **real scheduled game** when one exists for that week (site, roof, rest),
  instead of a hypothetical row.
- A **"What I used"** block: data as of (latest game week in play-by-play), injury report
  week and whether it is final (Friday) or partial, and every listed player with status,
  practice, chance to play and role share. If there is no report yet: said plainly, and
  everyone is treated as available.
- `--injuries/--no-injuries` to switch injury-aware player features on for one run.

**Result (done, `nflsim.live`):**
- `DataCache.load` already refetched stale current-season files whenever they were read,
  so `simulate` was refreshing its data already; step 1 added `--offline` (never fetch;
  use stale files; clear error for anything missing) and made the freshness visible.
- `simulate` uses the scheduled game when the teams meet that week (site, game type,
  roof, rest, weather). Swapped home/away is rejected with the right command; a played
  game shows its final score next to the prediction.
- "What I used": the scheduled game ID, the last completed week behind the ratings, when
  the play-by-play was cached, each team's injury report size for the week, every player
  listed out/doubtful/questionable **at all positions**, and whether injury-aware features
  are on (`--injuries/--no-injuries` for one run). With them off it says the report does
  not change the prediction; with them on, that only QB/RB/WR/TE count.
- Fixed a Phase 3 bug found here: nflverse dropped `date_modified` from the 2025-2026
  injury files, which crashed single-season loads (multi-season loads hid it).
- Not possible from the data: whether a report is the final (Friday) one. The cached
  files have no report dates from 2025, so the output gives the report's size instead.

### Step 2: overrides (both teams)

- `--out "Name"`, `--plays "Name"`, `--doubtful "Name"`, repeatable, matched to the team's
  roster for that week (case-insensitive, accents ignored; ambiguous names are rejected
  with the candidates listed; GSIS IDs accepted).
- Applied to availability before sampling, so backups move up exactly as for a real
  injury. An override for a player not on either roster is an error, not a silent no-op.
- Output shows the scenario and, with `--compare`, the same seed without the overrides,
  so the effect of "QB out" is a clean difference.

**Result (done, `nflsim.overrides`):** `--out`, `--plays`, `--doubtful` (repeatable),
`--compare`. Names match the week's roster (or the latest earlier roster week) by GSIS
ID, full name or unique last name, ignoring case, accents, punctuation and suffixes;
ambiguous names list every candidate, unknown or repeated players are errors. Overrides
switch injury-aware features on for the run and change the lineup's chance to play
directly, then the matchup's expected player features are recomputed, so `--compare`
needs no reload.

Example (2026 week 9, NYJ at KC, `--out Mahomes --compare`): KC win probability 71.0% ->
61.9%, KC expected points -3.4. What it shows about the model:
- **QB overrides matter; RB/WR/TE overrides barely do.** "Garrett Wilson plays" moved
  NYJ by -0.03 points: the model rates him slightly *below* his backups per target
  (-0.001 vs. +0.025 and +0.076 EPA), and the receiving coefficient is small (5.5 vs. 20.3
  for QB). That matches Phase 2 (target ratings ~0.5% skill); the output now says so.
- Defenders and players with no projected role are reported as having no effect.

### Step 3: per-player stat lines

For each run: team dropbacks, carries and targets (from the team's pace) split by the
sampled usage shares, times each player's per-play yards and touchdown rates (shrunk toward
position averages, as the EPA ratings are). Report median and 10-90% ranges for passing,
rushing and receiving yards and touchdowns.

Evaluated out of sample against box scores (`player_stats`, to be cached), 2023-2025:
mean absolute error and 10-90% interval coverage vs. a simple baseline (season-to-date
average). Shipped if it beats that baseline; otherwise reported and kept as
"usage x average" only.

**Result (done, `nflsim.model.stat_lines`, `nflsim stat-eval`, `simulate --players`):**
expected stat = team volume x expected share (injury-aware lineup, so injuries and
overrides flow through) x shrunk player rate. A QB's rushing is rated per pass attempt
(mostly scrambles on dropbacks); with per-carry rates QB rushing was badly underestimated.
Settings chosen on 2018-2022 (they matter little: 0.1% error between best and starting
values). Holdout 2023-2025 vs. each player's per-game average, 95% CIs from resampling games:

| Stat | Model MAE | Baseline MAE | Diff [95% CI] | Inside 10-90% range |
|---|---|---|---|---|
| Passing yards | 61.0 | 63.7 | -2.7 [-5.0, -0.2] | 79.9% |
| Passing TDs | 0.880 | 0.944 | -0.064 [-0.085, -0.041] | 78.4% |
| Rushing yards | 13.5 | 13.2 | **+0.27 [+0.05, +0.47]** | 76.8% |
| Rushing TDs | 0.608 | 0.665 | -0.057 [-0.073, -0.041] | 78.5% |
| Receptions | 1.373 | 1.397 | -0.024 [-0.038, -0.011] | 81.5% |
| Receiving yards | 16.5 | 17.1 | -0.62 [-0.78, -0.46] | 81.4% |
| Receiving TDs | 0.479 | 0.611 | -0.132 [-0.149, -0.115] | 77.6% |

Shipped: the model beats the baseline on 6 of 7 stats. **Rushing yards are 2% worse** (the
designed-run share is a weak signal for running backs' carries); kept for consistency with
injuries and overrides and labeled as such in the output.

### Step 4: summaries and output

- `--json` for the full result (probabilities, quantiles, players, flags, what was used).
- Line probabilities on request: `--spread -3.5 --total 47.5` gives cover, push and over
  chances from the simulated distribution.
- `nflsim week --season 2026 --week 11`: every game of a week in one table.
- Optional Streamlit dashboard as an extra dependency group (`uv sync --extra dashboard`),
  only after the CLI is stable.

### Step 5: defensive and offensive-line availability (opt-in, gated)

- Each defender's importance: snap share and PFR pressures (2018+) for pass rushers; snap
  share for coverage players. Same for offensive linemen (snap share).
- When a key defender is out, the defense's pass-rush and EPA ratings shift by his share of
  the unit's production, regressed toward the replacement.
- **Gate (set now):** like Phase 4, first test at the play level whether the adjustment
  predicts that game's pressure and EPA better than the unadjusted team rating (2023-2025
  holdout, game bootstrap). Only if it passes, backtest it in the game model. Phase 4 found
  that a defense's pressure rate barely carries from week to week, so a small or zero
  effect is the expected result, and will be reported as such.

## Data gaps and assumptions

- **Game-day inactives** (about 90 minutes before kickoff) are not in nflverse's reports;
  late scratches need `--out`.
- **Report timing:** Wednesday-Friday; a run earlier in the week has no report or a partial
  one, and says which.
- **nflverse update lag:** files update about daily, so a run on game day may miss the last
  practice report.
- **Names:** reports and rosters use full names; nicknames and suffixes (Jr., II) need the
  matching rules in step 2.
- **Box scores** (`player_stats`) are not cached yet; step 3 adds them.
- **Defensive player data:** PFR advanced stats start in 2018; coverage assignments do not
  exist (Phase 4), so coverage players are weighted by snaps only.
