# Under-confidence experiments (2026-11-18)

The default model's predicted margins are too narrow: actual margin vs. predicted margin
slope 1.20 on 2021-2024 (Vegas 1.03), predicted margin SD 4.3 (Vegas 6.0). Earlier work
ruled out shrinkage and early-season noise, and found that the direct margin fit fixes
the slope without significantly improving accuracy. This round tried four more ideas.

## Gate (set before any result)

Benchmark: the current default model, `nflsim backtest --seasons 2021-2024 --n 10000
--seed 42` (1,139 games), paired game bootstrap (2,000 resamples). A variant becomes the
default only if Brier improves with a 95% CI below 0, **or** both Brier and margin MAE
improve in at least 3 of the 4 seasons. Slope and SD are reported, but better calibration
alone does not pass.

## Results (2021-2024)

| Variant | Brier | Diff (95% CI) | Log loss | Margin MAE | Slope | SD | Seasons better |
|---|---|---|---|---|---|---|---|
| Benchmark | 0.2254 | | 0.6438 | 10.17 | 1.20 | 4.3 | |
| Opponent-adjusted ratings | 0.2253 | -0.0001 [-0.0012, +0.0011] | 0.6436 | 10.18 | 1.15 | 4.4 | 1/4 |
| EPA-only features (+pace, context) | 0.2262 | +0.0008 [-0.0005, +0.0022] | 0.6452 | 10.19 | 1.19 | 4.3 | 0/4 |
| EPA-only, no pace | 0.2266 | +0.0012 [-0.0008, +0.0031] | 0.6460 | 10.23 | 1.25 | 4.1 | 0/4 |
| Coefficient ridge 0.003 / 0.01 / 0.03 / 0.1 | 0.2253-0.2256 | within +/-0.0001 | | 10.17-10.18 | 1.21-1.29 | 4.0-4.3 | 1-2/4 |
| Direct margin fit | 0.2248 | -0.0006 [-0.0029, +0.0017] | 0.6435 | 10.18 | 0.94 | 5.6 | 1/4 |
| Opponent-adjusted + direct | 0.2249 | -0.0005 [-0.0030, +0.0017] | 0.6431 | 10.19 | 0.92 | 5.6 | 1/4 |
| Walk-forward recalibration | 0.2250 | -0.0005 [-0.0010, +0.0001] | 0.6428 | 10.17 | 1.15 | 4.5 | 2/4 |
| **+ market strength (`mkt`, `opp_mkt`)** | **0.2224** | **-0.0030 [-0.0062, +0.0002]** | 0.6374 | 10.11 | 0.92 | 5.8 | **3/4** |
| Market, EPA only | 0.2226 | -0.0028 [-0.0058, +0.0005] | 0.6377 | 10.11 | 0.93 | 5.7 | 3/4 |
| Market only (no EPA ratings) | 0.2239 | -0.0015 [-0.0059, +0.0031] | 0.6400 | 10.21 | 0.93 | 5.5 | 2/4 |
| Market + direct | 0.2232 | -0.0022 [-0.0057, +0.0012] | 0.6395 | 10.15 | 0.90 | 5.9 | 3/4 |
| **Market + walk-forward recalibration** | **0.2226** | **-0.0028 [-0.0054, -0.0003]** | 0.6377 | 10.13 | 1.08 | 4.9 | **3/4** |
| Vegas close | 0.2112 | | 0.6115 | 9.79 | 1.03 | 6.0 | |

Out of the test window, 2017-2024 (8 seasons): market strength gives Brier 0.2240 ->
0.2197 (diff -0.0042, 95% CI [-0.0068, -0.0015]), better in 7 of 8 seasons on Brier (2017
and 2022 tie) and 6 of 8 on both Brier and margin MAE.

## What each idea did

- **Opponent adjustment** (`[ratings] opponent_adjust`): a joint ridge fit of offense and
  defense per week. The adjusted ratings correlate 0.97-0.99 with the raw ones: by
  mid-season each team's opponents average out, so the correction is small. No gain.
- **Fewer features / coefficient ridge** (`--features`, `[model] ridge`): dropping
  success rate or pace is worse. Ridge makes no difference: the overlapping features
  make the coefficients unstable, but not the predictions.
- **Stretching margins** (direct fit, recalibration): fixes the slope, with a Brier
  change too small to be sure of. Recalibration scales fit on 2017-2020 are 0.94-1.11,
  i.e. the base model was nearly calibrated before 2021. Per-season slopes swing from
  0.91 to 1.42 (about +/-0.15 sampling error per season), so the 1.20 is partly the
  2021-2024 window.
- **Market strength** (`mkt`, `opp_mkt`, `[market]`): team strength fit to earlier weeks'
  closing spreads, shrunk toward last season. It carries what play-by-play ratings miss
  (injuries, roster changes, quarterback quality as the market sees it). It is the only
  idea that improves accuracy, and it widens the margins as a side effect. On its own it
  slightly **over**-shoots (slope 0.92; 0.83 over 2017-2024); walk-forward recalibration
  brings that back to 1.08 at the same Brier.

## Conclusion

The under-confidence came from missing information, not from how the margins are
scaled. Adding the market's view of team strength passes the gate; nothing else does.
The gap to Vegas (Brier 0.2224 vs. 0.2112) is still large.

**Caveat:** with market strength the closing lines of earlier weeks become an input. That
is not leakage (they are public before kickoff, and the game's own line is never used),
but Vegas is then no longer an independent benchmark: beating it would partly mean
re-using it. **Decision (2026-11-18):** market strength is on by default. Better predictions for real
games outweigh a fully independent Vegas comparison, and the backtest keeps one: it also
reports `pbp_only`, the same model without `mkt`/`opp_mkt`. Simulations flag teams with no
closing lines yet (last season's strength is used, carried over) and note that market
strength lags a week (a player back from injury is still priced as out). Recalibration is
not adopted: it fixes the slope (0.92 -> 1.08) but not Brier, and CRPS gets slightly
worse (7.336 -> 7.360).

Recalibration is not built into the CLI yet (it was run as a script on top of
`walk_forward`). The market settings (`prior_games` 4, `carryover` 0.6) were chosen before
testing and not tuned.
