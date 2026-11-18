# Market totals (2026-11-18)

## Why

With market strength and quarterback change on, the model is still clearly behind Vegas
on totals: total MAE 10.44 vs. 10.29, CRPS total 7.422 vs. 7.302. Margins get the
market's information through `mkt`/`opp_mkt` (earlier weeks' closing spreads), totals get
none: the spread says how much better a team is, not how many points its games produce.

## Feature (`nflsim.model.market`)

Per season and week W, a ridge fit over the season's games before W of

    total_line = league + t_home + t_away

with `t` shrunk toward `carryover` x last season's values, `prior_games` games of weight
per team, and an unpenalized league intercept (it moves from season to season). `t` is a
team's scoring tendency: how many points above average its games are expected to have.

- `mkt_tot`: the scoring team's `t`; `opp_mkt_tot`: the opponent's.

With `mkt`/`opp_mkt` the points model can then recover market-implied points for and
against (`(t + s) / 2`, `(t - s) / 2`). Settings are the existing `[market]` ones
(`prior_games` 4, `carryover` 0.6, `prior_season_ridge` 0.5), not tuned. Total lines are
public before kickoff, so there is no leakage, but the market becomes even more of an
input. The new features count as market features, so `pbp_only` leaves them out too.

## Gate (set before testing)

Benchmark = current default (six team features + indoor, wind + `mkt`, `opp_mkt` +
`qb_change`), `nflsim backtest --seasons 2021-2024 --n 10000 --seed 42`, paired game
bootstrap. Adopt as default if all of:

1. total MAE and CRPS total both improve, each with a 95% CI below 0;
2. total MAE improves in at least 3 of 4 seasons;
3. Brier is not worse by more than its noise (diff < +0.0005).

Also checked on 2017-2024.

## Results

(filled in after the run)
