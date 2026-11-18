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

`nflsim backtest --n 10000 --seed 42`, paired game bootstrap (5,000 resamples), benchmark =
current default.

| Seasons | Metric | Default | + `mkt_tot`, `opp_mkt_tot` | Diff (95% CI) |
|---|---|---|---|---|
| 2021-2024 | Total MAE | 10.436 | 10.423 | -0.013 [-0.077, +0.049] |
| | CRPS total | 7.422 | 7.421 | -0.001 [-0.040, +0.036] |
| | Brier | 0.2195 | 0.2187 | -0.0007 [-0.0015, +0.0000] |
| | Log loss | 0.6307 | 0.6287 | -0.0020 [-0.0037, -0.0003] |
| | Margin MAE | 10.063 | 10.044 | -0.019 [-0.045, +0.006] |
| 2017-2024 | Total MAE | 10.762 | 10.788 | +0.025 [-0.046, +0.102] |
| | CRPS total | 7.683 | 7.716 | +0.033 [-0.014, +0.082] |
| | Brier | 0.2171 | 0.2172 | +0.0002 [-0.0006, +0.0009] |

Total MAE by season (2021-2024): -0.05, -0.07, +0.05, +0.02 (2 of 4 better).
Vegas: total MAE 10.29, CRPS total 7.302.

**Gate: fail** on conditions 1 and 2 (both CIs include 0; 2 of 4 seasons). The
small log-loss gain on 2021-2024 is not what the feature was for and reverses on
2017-2024, so it is not a reason to adopt it. The features stay opt-in
(`--features ...,mkt_tot,opp_mkt_tot`); the default is unchanged.

## Why it does not help

The fitted coefficients are small (about +0.3 per point of tendency, own and opponent):
the team ratings already explain most of what earlier totals say about a team. The gap to
Vegas on totals is real (the Vegas-minus-model total correlates 0.19 with the model's
error; gap SD 3.0 points), but it is in *this week's* line (weather at kickoff, injuries,
the specific matchup), which earlier weeks' lines cannot carry. Using this week's line
would be using Vegas as the prediction, not as an input.
