# Phase 6 plan: context

Status: **done** (2026-11-17). Indoor and wind are now default features (total MAE
10.69 -> 10.61); home-field options and rest showed no gain and are opt-in. Step 4 was
skipped (see results). Taken before Phase 5 because context factors are cheap to add and
have known effects, while Phase 5 rebuilds the engine.

## Goal

Add game context the team ratings cannot see: home field, rest, roof and weather, and
(if those help) travel and division games. Each is a small, well-understood effect; the
question is whether the model, which does not use Vegas lines, gains from them.

## What the data already shows (2016-2025 completed games, `schedules`)

| Factor | Raw effect | vs. Vegas |
|---|---|---|
| Home field | home margin +3.1 (2016), +0.1 (2019), +0.2 (2020, mostly no fans), +1.8 to +3.0 (2021-2025) | n/a |
| Rest, 3+ days more | margin +3.4 (home rested) / +0.8 (away rested) vs. +1.9 even | priced (-0.2 to -0.3 vs. spread) |
| Roof | dome/closed totals 47.7-48.2 vs. 44.7 outdoors | dome/closed beat the total by +1.4 |
| Wind (outdoors) | totals 46.6 at <=5 mph, 42.5 at 10-15, 42.1 at 15-20 | 10-20 mph under the total by 1.7 |
| Division game | absolute margin 11.1 vs. 11.2 | none |

**Home field is the likeliest gain.** The model fits a constant home value on the previous
5 seasons, so every test season from 2021 on is trained on a window containing the flat
2019-2020 seasons. That should bias home field low, and may be part of the
under-confidence (slope 1.20).

## Results

Backtest 2021-2024, n 10000, seed 42, 1,139 games, paired bootstrap vs. the Phase 1
baseline (Brier 0.2254, log loss 0.6437, margin MAE 10.17, total MAE 10.69).

**Step 1, home field** (`[model] home_exclude_seasons`, `home_half_life_seasons`): no gain.

| Variant | Home value per team | Brier diff | 95% CI |
|---|---|---|---|
| Constant (baseline) | +0.60 to +0.78 | 0 | |
| Leave out 2020 | +0.73 to +0.98 | +0.0002 | [-0.0004, +0.0008] |
| Half-life 2 seasons | +0.49 to +0.91 | -0.0001 | [-0.0008, +0.0006] |
| Half-life 1 season | | -0.0001 | [-0.0013, +0.0010] |
| Leave out 2020 + half-life 2 | | -0.0001 | [-0.0009, +0.0006] |

The home value rises when 2020 is left out, but predictions barely change and the
calibration slope stays at 1.18-1.19: home field is not the cause of the under-confidence.

**Steps 2-3, context features:**

| Features added | Brier diff | Total MAE diff | 95% CI (total MAE) | Seasons better (total) |
|---|---|---|---|---|
| rest, opp_rest | +0.0001 [-0.0002, +0.0004] | -0.004 | [-0.030, +0.022] | |
| indoor, wind | +0.0000 [-0.0004, +0.0005] | **-0.088** | [-0.198, +0.026] | 3 of 4 |
| all four | +0.0001 [-0.0005, +0.0006] | -0.085 | [-0.198, +0.032] | 3 of 4 |
| all four, direct margin fit (vs. direct) | +0.0003 [-0.0002, +0.0008] | -0.090 | [-0.204, +0.025] | |

- Rest: coefficients ~0.1 points per day with flipping signs. Nothing to gain.
- Wind: stable, -0.16 to -0.22 points per mph per team (10 mph lowers the total by
  ~3.5-4.5). Indoor is unstable (-0.43 to +0.62 per team).
- Indoor + wind improve total MAE in 2021-2023 (-0.13, -0.20, -0.14) and not 2024 (+0.11).
  That meets the decision rule for total factors (3 of 4 seasons, overall better), so they
  are on by default, but the interval includes 0: the evidence is weak.

**Step 4 skipped.** Travel and division games are margin factors like rest and home field,
which showed nothing; the plan made them conditional on gains in steps 1-3.

## Steps

1. **Home field.** Compare the constant 5-season fit with: (a) leaving 2020 out of the
   home estimate, (b) weighting recent seasons more (half-life in seasons). Settings in
   `[model]`. Backtest each against the Phase 1 baseline.
2. **Context features** from `schedules`, per team side, as opt-in points-model features:
   - `rest`: days of rest relative to 7, capped (cap in config), for the scoring team and
     its opponent (`rest`, `opp_rest`);
   - `indoor`: dome or closed roof (same for both teams, so it moves the total);
   - `wind`: mph above a threshold for outdoor games, 0 indoors.
   Missing outdoor wind (45% of 2022, 19% of 2023) uses the league outdoor average and
   is flagged.
3. **Backtest each family** separately and together against the baseline (and against
   the step 1 winner), 2021-2024, n 10000, seed 42, paired bootstrap.
4. **Only if steps 1-3 show gains:** travel (time zones crossed, needs a small static
   team time-zone table) and division-game familiarity.

## Decision rule (set before looking at results)

- Margin and win-probability factors (home field, rest): adopt as the default only if
  the 95% CI for the Brier difference is below 0, or Brier and log loss both improve in
  at least 3 of 4 test seasons with the overall point estimate better.
- Total factors (indoor, wind): judged on total MAE with the same rule.
- Otherwise the factor stays opt-in and is reported as no gain.

## Data gaps and assumptions

- `temp`/`wind` in `schedules` are recorded at kickoff. Forecasts a few hours earlier are
  close but not identical; using the recorded values is a mild, flagged form of
  look-ahead, acceptable for a pre-game model that would use the forecast.
- No stadium coordinates or time zones in nflverse; travel needs a hand-made 32-team
  table (and international games use the stadium, not the home team).
- Coaching (tenure, new coach) is out of scope: only coach names are available, and the
  effects are small and hard to separate from team strength.
- Rest is in days between games; it does not distinguish a bye from a mini-bye after a
  Thursday game beyond the day count.
