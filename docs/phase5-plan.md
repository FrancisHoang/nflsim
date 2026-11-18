# Phase 5 plan: drive-level engine

Status: **in progress** (2026-11-17). Step 0 is done (benchmark below).

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
end field position and time used. Coverage report by season. About 28.6 drives per game
(27.4 in 2025, falling).

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
