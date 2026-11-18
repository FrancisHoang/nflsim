# Recency weighting (2026-11-18)

## Why

Team ratings use this season's games before week W with equal weight, so a team that
changed (new QB, scheme, injuries that stuck) is still rated on its September games in
December. Down-weighting older games could follow such changes faster, at the cost of
noisier ratings.

## Change (`[ratings] half_life_weeks`)

Each of the team's games this season before week W counts with weight
`0.5 ** ((W - week) / half_life_weeks)`, in both the totals and the play counts, so the
empirical-Bayes shrinkage sees the smaller effective sample and leans more on the prior.
League averages and last season's prior are unchanged. Off (equal weights) by default.

## Gate (set before testing)

Two half-lives chosen beforehand, not tuned: 4 and 8 weeks. Benchmark = current default,
`nflsim backtest --seasons 2021-2024 --n 10000 --seed 42`, paired game bootstrap. Adopt
one as default if Brier improves with a 95% CI below 0, or Brier and margin MAE both
improve in at least 3 of 4 seasons. Because two settings are tried, the chosen one must
also improve Brier on 2017-2024.

## Results

`nflsim backtest --n 10000 --seed 42`, paired game bootstrap (5,000 resamples), benchmark =
current default (Brier 0.2195, margin MAE 10.063 on 2021-2024; 0.2171 / 10.212 on
2017-2024).

| Half-life | Seasons | Brier diff (95% CI) | Margin MAE diff (95% CI) | Total MAE diff | Brier better |
|---|---|---|---|---|---|
| 4 weeks | 2021-2024 | +0.0007 [-0.0006, +0.0019] | +0.022 [-0.026, +0.070] | +0.014 | 2 of 4 |
| 4 weeks | 2017-2024 | +0.0005 [-0.0006, +0.0017] | **+0.047 [+0.002, +0.091]** | +0.021 | 4 of 8 |
| 8 weeks | 2021-2024 | +0.0001 [-0.0007, +0.0008] | +0.004 [-0.024, +0.031] | -0.001 | 2 of 4 |
| 8 weeks | 2017-2024 | -0.0001 [-0.0007, +0.0006] | +0.020 [-0.006, +0.045] | -0.004 | 5 of 8 |

**Gate: fail** for both. Neither improves Brier on 2021-2024 (both slightly worse, 2 of 4
seasons); a 4-week half-life makes margin MAE significantly worse on 2017-2024. Default
unchanged (equal weights); `half_life_weeks` stays as an option.

## Why it does not help

The shrinkage already balances signal and noise, and down-weighting old games mostly
throws away sample: the team's true level changes less over a season than single-game
EPA varies. The changes that do matter mid-season (a new quarterback) are now handled
directly by `qb_change` and by market strength, which reacts each week.
