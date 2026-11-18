# Phase 5 plan: drive-level engine

Status: **stopped at the step 4 gate** (2026-11-17). Steps 0-4 are done. The drive engine
does not beat the current engine on margin CRPS, so it stays opt-in (`[sim] engine =
"drive"`) and step 5 (game script) was not built. See "Results, steps 2-4".

## Goal

Replace the game-level score draw (each team's points from a negative binomial) with a
simulation of the game drive by drive: each possession starts at a field position, ends in
a touchdown, field goal, punt, turnover and so on, uses game clock, and sets up the next
possession. Team ratings shift the drive outcome probabilities. Game script (the score and
clock changing how teams play) comes after the drive engine works. A play-level engine is
out of scope unless a later need (per-player stat lines in Phase 7) requires it.

## Why

1. **Score distributions are wrong in shape.** Football scores come in 3s and 7s; a
   negative binomial does not. Backtest 2021-2024 (current default model, 2,000 runs per
   game):

   | Absolute margin | Actual | Simulated now |
   |---|---|---|
   | 3 | 15.2% | 9.0% |
   | 7 | 7.5% | 5.2% |
   | 6 | 7.5% | 5.4% |
   | 1 | 4.7% | 6.1% |

   Teams also score 1, 2, 4 or 5 points in 0.8% of simulated games. Mean margin and win
   probability can still be right, but spread and total probabilities (does the favorite
   cover -3?) are off, and Phase 7's outputs depend on them.
2. **Game script and pace** (trailing teams pass and hurry; leaders run clock) can only be
   modeled when the clock and score evolve within a game.
3. **The under-confidence** (slope 1.18-1.20) survived every factor in Phases 2-6. A drive
   engine maps ratings to outcomes differently, so it *may* help, but it is not promised:
   drive probabilities still come from fitted coefficients, which shrink toward the average
   when ratings are noisy, the same attenuation that likely causes the slope today.

## Approach

### Step 0: score the full distribution, not just its mean

Add to the backtest harness, for every engine:
- **CRPS** (continuous ranked probability score) of the margin and of the total: how well
  the whole simulated distribution fits the actual result. Lower is better.
- **Cover calibration:** predicted vs. actual chance the home team beats the closing spread
  and the game goes over the closing total (Vegas lines as a reference point only, never
  as model inputs).
- **Key numbers:** simulated vs. actual share of margins of 3, 7, 6, 10, 14.

Report the current default on these first; it is the benchmark for this phase.

**Result (done):** current default engine, 2021-2024, n 10000, seed 42, 1,139 games.

| | CRPS margin | CRPS total |
|---|---|---|
| Model | 7.404 | 7.517 |
| Vegas (normal around the closing line) | 7.126 | 7.302 |

- Key numbers (actual vs. model): margin 3: 15.2% vs. 9.0%; 7: 7.5% vs. 5.2%; 6: 7.5% vs.
  5.4%; 1: 4.7% vs. 6.1%.
- Against the closing lines the model has no edge: whatever its chance of a home cover or
  an over (0.25-0.75), the actual rate stays at 46-51%. Expected for efficient lines.

### Step 1: drive data

One row per drive from play-by-play (`fixed_drive`): game, offense, defense, start field
position, start clock and score difference, result (touchdown, field goal, missed field
goal, punt, turnover, turnover on downs, safety, opponent touchdown, end of half), points,
end field position and time used. Coverage report by season.

**Result (done, `nflsim.data.drives`, `nflsim drive-eval`):** 21.2-23.2 drives per game
(falling: 23.0 in 2016, 21.2 in 2025), 1.8-2.2 offensive points per drive, about 160-171
seconds per drive. Drive points add up to both final scores in 100% of games, every
season. Points come from the running home/away score: after a defensive touchdown the
extra point is credited to the other team within the same drive, so per-play `posteam`
scores would miscount (the first version matched only 69-78% of games). (The plan's
earlier "28.6 per game" counted rows with no team on offense.)

### Step 2: league-average drive engine (no team ratings)

Fit on earlier seasons only:
- drive result probabilities given start field position (and half/clock for end-of-half);
- next drive's start field position given the result (punt net, turnover spot, kickoff),
  **per season**, because kickoff rules changed: touchback rate 57-61% through 2022, 73%
  in 2023, 64% in 2024, 21% in 2025;
- time used per drive, so the number of drives per game comes out right;
- extra points and two-point tries, and overtime rules by season (OT shortened to 10
  minutes in 2017; both teams get the ball in the playoffs from 2022 and in the regular
  season from 2024).

Check: simulated league-average games reproduce actual totals, drives per game and the
key-number frequencies.

### Step 3: team strength in drives

Shift drive result probabilities (multinomial logit) by the offense's and defense's
ratings (the existing as-of-W-1 team ratings, so no new leakage), and drives per game by
pace. Fit on the previous 5 seasons, as the points model is. Measure the calibration slope
directly: if the fitted shifts are attenuated, try fitting with ratings from more stable
windows rather than adding factors.

### Step 4: backtest (the gate)

Drive engine vs. the current default, 2021-2024, n 10000, seed 42, paired bootstrap:
Brier, log loss, margin and total MAE, CRPS, cover calibration, key numbers.

**Decision rule (set before results):** the drive engine becomes the default if margin
CRPS improves with a 95% CI below 0 **and** Brier and log loss are no worse in point
estimate. If it improves the shape (key numbers, CRPS) but not win probability, it ships
opt-in. Otherwise it is reported as no gain.

### Results, steps 2-4

**As built** (`nflsim.model.drive_engine`, config `[drives]`): instead of a parametric drive
model, each simulated drive **resamples a real drive** from the training seasons that
started in the same field-position bin (10 yards) and clock bucket (<=120, 120-420, >420
seconds left in the half), bringing its points for both sides, time used and the
opponent's next start. Kickoff starts come from the latest training season; overtime
follows each season's rules. **Team strength** is exponential tilting toward drives with
more net points, with each team's tilt set so its mean points match the points model, then
corrected by two pilot simulations (2,000 runs). So the engine changes the *shape* of the
score distribution and keeps the existing model's means. 285 games x 10,000 runs take
about 30-45 seconds.

**League-average check (step 2):** about 2.0 points per drive and totals near 47, but
absolute margins of 3 come out at **7.3%** (actual 15.2%); 7s at 7.4% (actual 7.5%).

**Gate (step 4):** 2021-2024, n 10000, seed 42, paired bootstrap vs. the current engine.

| Metric | Drive engine | Current | Diff (95% CI) |
|---|---|---|---|
| CRPS margin | 7.423 | 7.404 | +0.019 [-0.008, +0.046] |
| CRPS total | 7.499 | 7.518 | -0.019 [-0.058, +0.015] |
| Brier | 0.2251 | 0.2254 | -0.0003 [-0.0012, +0.0007] |
| Log loss | 0.6432 | 0.6438 | -0.0006 [-0.0027, +0.0016] |
| Margin MAE | 10.163 | 10.169 | -0.006 |
| Total MAE | 10.587 | 10.605 | -0.018 |
| Calibration slope | 1.18 | 1.20 | |
| Margins of 3 / 7 | 7.1% / 7.3% | 9.0% / 5.2% | actual 15.2% / 7.5% |

**Decision (rule set before results): fail.** Margin CRPS did not improve, and neither did
the shape overall (more 7s, fewer 3s). It is kept as an opt-in engine.

What we learned:
- **The excess of 3-point margins does not come from drive outcomes.** Resampling real
  drives independently reproduces 7s but not 3s. The 3s come from the end of games: teams
  playing for a tying or winning field goal, protecting a lead, and overtime field goals.
  That is game script (step 5), so a later attempt should start there.
- **A first run looked better on calibration (slope 1.08, Brier -0.0004), but it was an
  artifact:** the tilt was miscalibrated (simulated totals 0.7-2.8 points above the points
  model), which stretched margins. With pilot calibration the means match (within
  0.1-0.6) and the slope is back to 1.18. The under-confidence is not an engine problem.

### Step 5: game script, only if step 4 passes

Trailing and leading behavior (pace, pass rate via drive outcome shifts, 4th-down
aggressiveness by score and time), and two-minute drives at the end of halves.

## Data gaps and assumptions

- **Rule changes inside the window:** kickoffs (2023 fair catch, 2024 new format, 2025
  touchback to the 35), overtime (2017, 2022, 2024), extra-point distance (2015, before
  the window). Anything about field position or overtime is modeled per season, and the
  newest season's rules are used when simulating it.
- **Drive results are slightly messy:** a handful of drives have no result (15 of 78,960),
  and "end of half" drives depend on the clock, which step 2 models only coarsely.
- **Drives are treated as independent** given start position and ratings (no momentum).
- **Fourth-down decisions** are inside drive outcomes until step 5; coaches have become
  more aggressive over the window, so recent seasons should weigh more if that drifts.
- **Speed:** simulating ~29 drives per game for 10,000 runs and 1,139 games must stay
  vectorized in numpy (no Python loop per drive per run).
