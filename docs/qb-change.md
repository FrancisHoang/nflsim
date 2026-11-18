# Quarterback change (2026-11-18)

## Why

With market strength on, the model is as good as Vegas in weeks 1-4 (Brier gap +0.0006)
but falls behind later: weeks 10-18 hold 78% of the gap. The 40 games (3.5%) where the
model and Vegas differ by 7+ points hold 30% of it, and almost all of them are a
quarterback the ratings had not seen: backups (Josh Johnson, Bryce Perkins, Brandon
Allen, Jake Haener, Joe Milton, ...) or starters resting in week 18 (14 of the 40 are in
weeks 17-18). Team ratings and market strength are built from games already played, so
they price the quarterbacks of those games.

## Feature (`nflsim.model.qb_change`)

`qb_change = dev(this week's QB) - baseline`: `dev` is a QB's dropback rating minus the
league dropback average (Phase 2 ratings), and the baseline is the dropback-weighted `dev`
of the team's QBs in its games this season before week W (last season's with none yet).
All ratings are as of week W, so only *who* plays differs.

- `qb_change`: this week's QB is the starter listed in the nflverse schedule
  (`home_qb_id`/`away_qb_id`), falling back to `qb_change_proj` when none is listed.
- `qb_change_proj`: this week's QB value is the projected `off_qb` (Phase 2 usage, or
  injury-aware with `--injuries`).

## Gate (set before testing)

As for market strength: benchmark = current default (market on), 2021-2024, n 10000,
seed 42, paired game bootstrap; adopt if Brier improves with a 95% CI below 0, or Brier and
margin MAE both improve in at least 3 of 4 seasons. Also checked on 2017-2024.

## Results

| Variant | Brier 2021-2024 | Diff (95% CI) | Margin MAE | Total MAE | Seasons | Brier 2017-2024 | Diff (95% CI) |
|---|---|---|---|---|---|---|---|
| Default | 0.2224 | | 10.11 | 10.56 | | 0.2197 | |
| + `qb_change` (listed starter) | **0.2195** | **-0.0029 [-0.0052, -0.0004]** | 10.06 | 10.44 | 3/4 | **0.2171** | **-0.0027 [-0.0047, -0.0005]** |
| + `qb_change_proj` | 0.2219 | -0.0005 [-0.0025, +0.0015] | 10.11 | 10.47 | 3/4 | 0.2186 | -0.0012 [-0.0027, +0.0003] |
| + `qb_change_proj`, injury-aware | 0.2211 | -0.0013 [-0.0037, +0.0009] | 10.08 | 10.44 | 3/4 | 0.2184 | -0.0013 [-0.0029, +0.0003] |
| Vegas close | 0.2112 | | 9.79 | 10.29 | | | |

All three pass the gate (the projections on 3 of 4 seasons only); the listed starter is the
only one with a CI below 0, in both windows. The fitted coefficient is about +26 points
per EPA/dropback (2021-2024), so a backup 0.15 EPA/dropback worse moves the margin by about
4 points. The two sources correlate only 0.53: the projection misses many changes
(it sees a new QB only after his first game, or from the injury report).

## Caveat: how clean is the listed starter?

For past games nflverse lists the **actual** starter, which is usually announced before
kickoff, but not always (surprise starts, game-time decisions). So the listed-starter
backtest is an upper bound. For upcoming games nflverse lists an expected starter (e.g.
2026 week 11) or none (later weeks: the feature falls back to the projection). The honest
real-time gain is somewhere between the injury-aware projection (-0.0013) and the listed
starter (-0.0029); nflverse's historical pre-game listings are not available to measure it.

**Decision (2026-11-18):** `qb_change` with the listed starter is on by default.
Starters are usually known before kickoff, and the closing line uses the same information.
`simulate` prints each team's listed starter in "What I used", and flags teams with none
listed (then the projection is used). Ruling out the listed starter with `--out` or
`--doubtful` switches that team to the override-adjusted projection, i.e. his backup
(BUF 2026 week 11, Josh Allen out: -7.3 points, against -4.4 from `off_qb` alone before).
Other QB overrides leave the listed starter in place. New benchmark: Brier 0.2195, log
loss 0.6307, margin MAE 10.06, total MAE 10.44 (`pbp_only` 0.2229).

Still not handled: per-run availability sampling does not resample `qb_change_proj`.
