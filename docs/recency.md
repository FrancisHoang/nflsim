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

(filled in after the run)
