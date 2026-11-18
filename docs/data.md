# Data sources

Everything comes from nflverse through `nflreadpy`. Run `uv run nflsim datasets` for the
live list.

| Dataset | First season | Used for (planned) |
|---|---|---|
| `pbp` | 1999 | Team EPA/success rate/pace (Phase 1), play outcomes (Phase 5) |
| `schedules` | 1999 | Games, results, rest days, roof, closing `spread_line`/`total_line` |
| `rosters`, `rosters_weekly` | 1999 / 2002 | Who is on each team, by week |
| `player_stats`, `team_stats` | 1999 | Weekly box-score stats |
| `injuries` | 2009 | Game status and practice participation (Phase 3) |
| `depth_charts` | 2001 | Backups for injured players (Phase 3) |
| `snap_counts` | 2012 | Snap-weighting player ratings (Phase 2) |
| `participation` | 2016 | Players on the field, pressure, rushers, man/zone per play (Phase 4; see [phase4-plan.md](phase4-plan.md) for coverage by season) |
| `ngs_passing`, `ngs_receiving`, `ngs_rushing` | 2016 | Next Gen Stats (Phase 4) |
| `pfr_pass`, `pfr_rush`, `pfr_rec`, `pfr_def` | 2018 | PFR advanced stats: pressures, coverage (Phase 4) |
| `players`, `teams` | n/a | ID crosswalk, team metadata |

The default fetch range is 2016 onward (`default_first_season` in the config).

## Known gaps and caveats

- **Injury reports give a status, not a severity.** "Questionable" covers everything from
  a near-certain starter to a near-certain scratch. Phase 3 will have to estimate play
  probabilities from historical status-to-outcome rates.
- **Injury reports can change before kickoff.** Late scratches and inactives may not be in
  the weekly file, so the most recent report before kickoff is the best we have.
- **PFR advanced stats start in 2018**, so 2016–2017 matchups will have less data.
- **Participation data may lag or be incomplete for recent seasons.** Later phases must
  handle missing data gracefully.
- **Closing lines** in `schedules` are consensus numbers, not a specific sportsbook's.
- **Schemas drift between seasons.** When stacking seasons, missing columns are filled with
  null rather than raising an error.
- **Data leakage is not handled at this layer.** The cache returns whole seasons. Filtering
  to "information available before kickoff of week W" is the job of each model phase.
